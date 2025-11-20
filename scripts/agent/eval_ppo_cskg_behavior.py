# scripts/agent/eval_ppo_cskg_behavior.py
# -*- coding: utf-8 -*-
"""
行为级回放 / 对比脚本：
- 载入一个 PPO checkpoint
- 在 CybORGWrapper 上跑若干回合
- 对每一步记录：
    - obs facts（suspicious_activity / host_compromised / ...）
    - 选的动作、EnvReward / TotalReward
    - legal_mask_sum / rule_mask_sum / combined_mask_sum
    - 所选动作的 prior 值 & top-3 prior
    - KB.explain_decision 的输出
- 可以只看带 CSKG，也可以额外跑 plain_no_cskg 做对比（--with-plain）
"""

import os
import sys
import time
import json
import pathlib
import argparse
from typing import Any, Dict, List

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical

# ===== 路径注入 =====
ROOT = pathlib.Path(__file__).resolve().parents[2]  # C:\cybdef
THIRD = ROOT / "third_party" / "CybORG"

for p in (ROOT, THIRD):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

# ===== 项目内 import =====
from scripts.envs.cyborg_wrapper import CybORGWrapper
from scripts.cskg.reasoner import KnowledgeBridge
import yaml

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ===== 与 train_ppo_cskg_old.py 一致的 ActorCritic =====
class ActorCritic(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden: int = 128):
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, act_dim),
        )
        self.critic = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1),
        )

    def forward(self, obs):
        logits = self.actor(obs)
        value = self.critic(obs)
        return logits, value.squeeze(-1)


def to_obs_vector(obs_raw: Any) -> np.ndarray:
    """
    兼容 CybORGWrapper.reset()/step() 返回的 dict 结构：
    - {"obs_vec": np.ndarray, "facts": {...}, "raw": ...}
    """
    if isinstance(obs_raw, dict):
        if "obs_vec" in obs_raw:
            arr = obs_raw["obs_vec"]
        else:
            for k in ["obs", "observation", "vector", "state"]:
                if k in obs_raw:
                    arr = obs_raw[k]
                    break
            else:
                raise TypeError(f"obs_raw 中找不到 obs_vec/obs 等字段: keys={list(obs_raw.keys())}")
    else:
        arr = obs_raw

    arr = np.array(arr, dtype=np.float32).reshape(-1)
    return arr


def to_serializable(obj):
    """
    把 numpy / torch 相关类型递归转成 Python 原生，方便 json.dumps
    """
    import numpy as _np

    if isinstance(obj, (_np.floating,)):
        return float(obj)
    if isinstance(obj, (_np.integer,)):
        return int(obj)
    if isinstance(obj, _np.ndarray):
        return obj.tolist()
    if isinstance(obj, torch.Tensor):
        return to_serializable(obj.detach().cpu().numpy())
    if isinstance(obj, (list, tuple)):
        return [to_serializable(v) for v in obj]
    if isinstance(obj, dict):
        return {k: to_serializable(v) for k, v in obj.items()}
    return obj


def load_ppo_config() -> Dict[str, Any]:
    """从 scripts/configs/ppo.yaml 读取 rule_coef 等参数，没有就用默认"""
    ppo_yaml = ROOT / "scripts" / "configs" / "ppo.yaml"
    if not ppo_yaml.exists():
        print(f"⚠ 未找到 {ppo_yaml}，使用 rule_coef=0.1, device=cuda/cpu 自动")
        return {
            "rule_coef": 0.1,
            "device": "cuda" if torch.cuda.is_available() else "cpu",
        }
    with open(ppo_yaml, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    return cfg


def build_env() -> CybORGWrapper:
    env_yaml = ROOT / "scripts" / "configs" / "env.yaml"
    env = CybORGWrapper(str(env_yaml))
    return env


def build_kb() -> KnowledgeBridge:
    cskg_yaml = ROOT / "scripts" / "configs" / "cskg.yaml"
    seed_graph = ROOT / "scripts" / "configs" / "seed_graph.json"
    kb = KnowledgeBridge(
        seed_graph_path=str(seed_graph),
        cskg_rules_path=str(cskg_yaml),
        recent_steps=10,
    )
    return kb


def load_actor_critic(env: CybORGWrapper, ckpt_path: str) -> ActorCritic:
    obs_dim = env.obs_dim
    act_dim = env.action_dim
    ac = ActorCritic(obs_dim, act_dim)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state_dict = ckpt.get("model", ckpt)
    ac.load_state_dict(state_dict)
    ac.to(DEVICE)
    ac.eval()
    return ac


@torch.no_grad()
def run_behavior(
    env: CybORGWrapper,
    ac: ActorCritic,
    episodes: int,
    use_cskg: bool,
    rule_coef: float,
    log_f
):
    """
    采样若干 episode，并把每一步的详细决策过程写入 jsonl：
    - mode: "cskg" 或 "plain_no_cskg"
    """
    mode_label = "cskg" if use_cskg else "plain_no_cskg"

    kb = build_kb() if use_cskg else None
    action_names = env.action_space.names
    act_dim = env.action_dim

    for ep in range(1, episodes + 1):
        obs_raw = env.reset()
        obs_vec = to_obs_vector(obs_raw)
        facts = obs_raw.get("facts", {}) if isinstance(obs_raw, dict) else {}

        if kb is not None and hasattr(kb, "reset_episode"):
            kb.reset_episode()

        done = False
        total_r_env = 0.0
        step_count = 0

        print(f"\n[{mode_label}] Episode {ep} 开始")

        while not done:
            step_count += 1

            obs_tensor = torch.from_numpy(obs_vec).to(DEVICE).unsqueeze(0)
            logits, _ = ac(obs_tensor)
            logits = logits.squeeze(0)  # [act_dim]

            # === 环境合法掩码 ===
            try:
                legal_mask_np = env._current_legal_mask().astype(np.float32)
            except Exception:
                legal_mask_np = np.ones(act_dim, dtype=np.float32)

            if legal_mask_np.shape[0] != act_dim:
                raise ValueError(f"legal_mask 维度异常: {legal_mask_np.shape[0]} vs act_dim={act_dim}")

            prior_np = np.zeros(act_dim, dtype=np.float32)
            rule_mask_np = np.ones(act_dim, dtype=np.float32)
            explain = {}

            # === 带 CSKG 分支 ===
            if use_cskg and kb is not None:
                # 用 facts 更新 KB 内部状态
                if hasattr(kb, "update_from_facts"):
                    kb.update_from_facts(facts)

                # prior logits
                prior_np = kb.prior_logits(facts, action_names)
                if isinstance(prior_np, tuple):
                    prior_np = prior_np[0]
                prior_np = np.array(prior_np, dtype=np.float32)

                # rule mask
                mask_res = kb.query_action_mask(facts, action_names)
                if isinstance(mask_res, tuple):
                    rule_mask_np = np.array(mask_res[0], dtype=np.float32)
                else:
                    rule_mask_np = np.array(mask_res, dtype=np.float32)

                if rule_mask_np.shape[0] != act_dim:
                    raise ValueError(f"rule_mask 维度异常: {rule_mask_np.shape[0]} vs act_dim={act_dim}")
                if prior_np.shape[0] != act_dim:
                    raise ValueError(f"prior 维度异常: {prior_np.shape[0]} vs act_dim={act_dim}")

                try:
                    explain = kb.explain_decision(facts, action_names)
                except Exception:
                    explain = {}

            # ==== 融合掩码 ====
            combined_mask_np = (legal_mask_np * rule_mask_np).astype(np.float32)
            if combined_mask_np.sum() <= 0:
                combined_mask_np[0] = 1.0

            # ==== logits + prior + mask ====
            logits = logits.clone()
            if use_cskg:
                prior_t = torch.from_numpy(prior_np).to(DEVICE)
                if rule_coef != 0.0:
                    logits = logits + rule_coef * prior_t
                else:
                    logits = logits + prior_t

            mask_t = torch.from_numpy(combined_mask_np).to(DEVICE)
            logits[mask_t == 0] = -1e9

            dist = Categorical(logits=logits)
            action = dist.sample()
            a_idx = int(action.item())
            a_name = action_names[a_idx]

            # === 与环境交互 ===
            next_obs_raw, r_env, done, info = env.step(a_idx)
            total_r_env += float(r_env)

            # 总奖励（如果你想看 reward_shaping 的效果）
            if use_cskg and kb is not None and hasattr(kb, "reward_shaping"):
                try:
                    r_total = kb.reward_shaping(facts, a_name, float(r_env))
                except Exception:
                    r_total = float(r_env)
            else:
                r_total = float(r_env)

            # KB 更新历史
            next_facts = next_obs_raw.get("facts", {}) if isinstance(next_obs_raw, dict) else {}
            if use_cskg and kb is not None and hasattr(kb, "step_update"):
                try:
                    kb.step_update(next_facts, a_name, r_total)
                except Exception:
                    pass

            # === 构造日志记录 ===
            top_idx = np.argsort(prior_np)[-3:][::-1] if use_cskg else []
            top_prior = [
                [action_names[i], float(prior_np[i])]
                for i in top_idx
            ]

            rec = {
                "mode": mode_label,
                "episode": ep,
                "step": step_count,
                "action_idx": a_idx,
                "action_name": a_name,
                "env_reward": float(r_env),
                "total_reward": float(r_total),
                "done": bool(done),
                "legal_mask_sum": float(legal_mask_np.sum()),
                "rule_mask_sum": float(rule_mask_np.sum()) if use_cskg else None,
                "combined_mask_sum": float(combined_mask_np.sum()),
                "prior_chosen": float(prior_np[a_idx]) if use_cskg else None,
                "top_prior": top_prior,
                "facts": facts,
                "explain": explain,
            }
            log_f.write(json.dumps(to_serializable(rec), ensure_ascii=False) + "\n")

            print(
                f"[{mode_label}] ep={ep:02d} step={step_count:02d} "
                f"a={a_name:20s} r_env={r_env:6.2f} r_tot={r_total:6.2f} "
                f"legal={legal_mask_np.sum():3.0f} rule={rule_mask_np.sum():3.0f} comb={combined_mask_np.sum():3.0f}"
            )

            # 下一步
            obs_vec = to_obs_vector(next_obs_raw)
            facts = next_facts

        print(f"[{mode_label}] Episode {ep} 结束，总 EnvReward={total_r_env:.3f}, 步数={step_count}")


def main():
    parser = argparse.ArgumentParser(
        description="对单个 PPO+CSKG checkpoint 做行为级回放（带/不带 CSKG）"
    )
    parser.add_argument(
        "--ckpt",
        required=True,
        help="模型路径，例如 scripts/runs/ppo_cskg/ppo_xxx/ac_upd225.pt",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=5,
        help="每种模式采样的回合数（默认 5）",
    )
    parser.add_argument(
        "--with-plain",
        action="store_true",
        help="除了带 CSKG 外，再跑一遍 plain_no_cskg 对比",
    )
    args = parser.parse_args()

    cfg = load_ppo_config()
    rule_coef = float(cfg.get("rule_coef", 0.1))

    # 设备设置
    dev_cfg = str(cfg.get("device", "cuda")).lower()
    global DEVICE
    if dev_cfg == "cuda" and torch.cuda.is_available():
        DEVICE = torch.device("cuda")
    else:
        DEVICE = torch.device("cpu")
    print(f"📟 使用设备: {DEVICE}, rule_coef={rule_coef}")

    ckpt_path = os.path.abspath(args.ckpt)
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"ckpt 不存在: {ckpt_path}")

    # 输出日志文件
    ckpt_tag = pathlib.Path(ckpt_path).stem  # ac_upd200
    run_dir = ROOT / "scripts" / "runs" / "ppo_cskg"
    os.makedirs(run_dir, exist_ok=True)
    ts = int(time.time())
    log_path = run_dir / f"behavior_{ckpt_tag}_{ts}.jsonl"

    env = build_env()
    ac = load_actor_critic(env, ckpt_path)

    with open(log_path, "w", encoding="utf-8") as f:
        # 1) 带 CSKG（主角）
        run_behavior(env, ac, episodes=args.episodes, use_cskg=True, rule_coef=rule_coef, log_f=f)

        # 2) 需要的话，顺便跑一遍 plain_no_cskg
        if args.with_plain:
            run_behavior(env, ac, episodes=args.episodes, use_cskg=False, rule_coef=0.0, log_f=f)

    env.close()
    print(f"\n✅ 行为回放完成，详细日志已写入：{log_path}")


if __name__ == "__main__":
    main()
