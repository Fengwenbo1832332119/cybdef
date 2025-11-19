# scripts/agent/eval_ppo_cskg_compare.py
# -*- coding: utf-8 -*-
"""
对比评估多个 PPO+CSKG checkpoint：
- 同一套环境 CybORGWrapper
- 同一套 CSKG（cskg.yaml）
- 对每个 ckpt：
    1) 带 CSKG（cskg）
    2) 关闭 CSKG（plain）
  分别跑若干 episode，统计：
    - 平均 EnvReward ± std
    - 平均步长
    - 动作使用 Top-K

新增：
- 把所有 ckpt 的评估结果汇总到一张表（终端打印对比矩阵）
- 同时导出到 CSV 文件，便于后续画图分析
"""

import os
import sys
import time
import json
import pathlib
import argparse
import csv
from collections import Counter
from typing import Dict, Any

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


# ===== 和 train_ppo_cskg.py 完全一致的网络结构 =====
class ActorCritic(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden=128):
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
def run_episodes(
    env: CybORGWrapper,
    ac: ActorCritic,
    episodes: int = 20,
    use_cskg: bool = True,
    rule_coef: float = 0.1
) -> Dict[str, Any]:
    """
    用同一网络、同一环境跑若干回合：
    - use_cskg=True  : 带 CSKG（prior + 掩码）
    - use_cskg=False : 关闭 CSKG，只用 env.legal_mask
    """
    action_names = env.action_space.names
    act_dim = env.action_dim

    kb = build_kb() if use_cskg else None

    ep_rewards = []
    ep_steps = []
    action_counter = Counter()

    for ep in range(1, episodes + 1):
        obs_raw = env.reset()
        obs_vec = to_obs_vector(obs_raw)
        facts = obs_raw.get("facts", {}) if isinstance(obs_raw, dict) else {}

        if kb is not None and hasattr(kb, "reset_episode"):
            kb.reset_episode()

        done = False
        total_r_env = 0.0
        step_count = 0

        while not done:
            step_count += 1

            obs_tensor = torch.from_numpy(obs_vec).to(DEVICE).unsqueeze(0)
            logits, _ = ac(obs_tensor)
            logits = logits.squeeze(0)  # [act_dim]

            # === 取合法掩码 ===
            try:
                legal_mask_np = env._current_legal_mask().astype(np.float32)
            except Exception:
                legal_mask_np = np.ones(act_dim, dtype=np.float32)

            if legal_mask_np.shape[0] != act_dim:
                raise ValueError(f"legal_mask 维度异常: {legal_mask_np.shape[0]} vs act_dim={act_dim}")

            # === CSKG 分支 ===
            if use_cskg and kb is not None:
                # 用 facts 更新 KB
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

                combined_mask_np = (legal_mask_np * rule_mask_np).astype(np.float32)
                if combined_mask_np.sum() <= 0:
                    combined_mask_np[0] = 1.0

                prior_t = torch.from_numpy(prior_np).to(DEVICE)
                logits = logits.clone()
                if rule_coef != 0.0:
                    logits = logits + rule_coef * prior_t
                else:
                    logits = logits + prior_t

                mask_t = torch.from_numpy(combined_mask_np).to(DEVICE)
                logits[mask_t == 0] = -1e9

            else:
                # 不用 CSKG，只用环境合法动作
                combined_mask_np = legal_mask_np
                if combined_mask_np.sum() <= 0:
                    combined_mask_np[0] = 1.0
                logits = logits.clone()
                mask_t = torch.from_numpy(combined_mask_np).to(DEVICE)
                logits[mask_t == 0] = -1e9

            dist = Categorical(logits=logits)
            action = dist.sample()
            a_idx = int(action.item())
            a_name = action_names[a_idx]
            action_counter[a_name] += 1

            next_obs_raw, r_env, done, info = env.step(a_idx)
            total_r_env += float(r_env)

            # 更新 obs / facts
            obs_vec = to_obs_vector(next_obs_raw)
            facts = next_obs_raw.get("facts", {}) if isinstance(next_obs_raw, dict) else {}

            # KB 记录历史（可选）
            if use_cskg and kb is not None and hasattr(kb, "step_update"):
                try:
                    kb.step_update(facts, a_name, float(r_env))
                except Exception:
                    pass

        ep_rewards.append(total_r_env)
        ep_steps.append(step_count)

    # 汇总结果
    ep_rewards = np.array(ep_rewards, dtype=np.float32)
    ep_steps = np.array(ep_steps, dtype=np.float32)

    summary = {
        "episodes": episodes,
        "mean_env_reward": float(ep_rewards.mean()),
        "std_env_reward": float(ep_rewards.std()),
        "mean_steps": float(ep_steps.mean()),
        "action_counter": action_counter,
    }
    return summary


def print_summary(label: str, summary: Dict[str, Any], top_k: int = 20):
    print(f"\n===== [{label}] 评估结果 =====")
    print(f"  回合数       : {summary['episodes']}")
    print(
        f"  平均 EnvReward: {summary['mean_env_reward']:.3f} ± {summary['std_env_reward']:.3f}"
    )
    print(f"  平均步长      : {summary['mean_steps']:.1f}")
    print("\n  动作使用统计（Top {}）".format(top_k))

    counter: Counter = summary["action_counter"]
    for name, cnt in counter.most_common(top_k):
        print(f"  {name:20s}: {cnt}")


def main():
    parser = argparse.ArgumentParser(
        description="对比评估多个 PPO+CSKG checkpoint（CSKG on/off）"
    )
    parser.add_argument(
        "--ckpt",
        nargs="+",
        required=True,
        help="一个或多个 ckpt 路径，例如 ac_upd025.pt ac_upd100.pt ac_upd200.pt",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=20,
        help="每个配置评估的回合数（默认20）",
    )
    parser.add_argument(
        "--no-plain",
        action="store_true",
        help="只评估带 CSKG，不跑 plain_no_cskg 模式",
    )
    parser.add_argument(
        "--no-cskg",
        action="store_true",
        help="只评估 plain_no_cskg，不跑带 CSKG 模式",
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

    # 建一个 env 复用（注意：红方在 reset 时仍然会重建）
    env = build_env()

    # 用于最后整体汇总
    all_results = []  # 每一行：{"ckpt", "mode", "mean_reward", "std_reward", "mean_steps"}

    for ckpt_path in args.ckpt:
        ckpt_path = os.path.abspath(ckpt_path)
        if not os.path.exists(ckpt_path):
            print(f"\n❌ ckpt 不存在: {ckpt_path}")
            continue

        tag = pathlib.Path(ckpt_path).stem  # 比如 ac_upd025
        print(f"\n==============================")
        print(f"🔍 评估模型: {ckpt_path}")
        print(f"==============================")

        ac = load_actor_critic(env, ckpt_path)

        # 1) 带 CSKG
        if not args.no_cskg:
            summary_cskg = run_episodes(
                env, ac, episodes=args.episodes, use_cskg=True, rule_coef=rule_coef
            )
            print_summary(f"{tag} + CSKG", summary_cskg)

            all_results.append({
                "ckpt": tag,
                "mode": "cskg",
                "mean_reward": summary_cskg["mean_env_reward"],
                "std_reward": summary_cskg["std_env_reward"],
                "mean_steps": summary_cskg["mean_steps"],
            })

        # 2) 关闭 CSKG（只保留 env.legal_mask）
        if not args.no_plain:
            summary_plain = run_episodes(
                env, ac, episodes=args.episodes, use_cskg=False, rule_coef=0.0
            )
            print_summary(f"{tag} plain_no_cskg", summary_plain)

            all_results.append({
                "ckpt": tag,
                "mode": "plain",
                "mean_reward": summary_plain["mean_env_reward"],
                "std_reward": summary_plain["std_env_reward"],
                "mean_steps": summary_plain["mean_steps"],
            })

    env.close()

    # ===== 最终汇总打印 =====
    if all_results:
        print("\n\n================ 总体对比汇总 ================")
        # 按 ckpt + mode 排一下，方便看
        all_results.sort(key=lambda x: (x["ckpt"], x["mode"]))

        # 终端表格打印
        header = f"{'ckpt':15s} {'mode':8s} {'mean_R':>10s} {'std_R':>10s} {'mean_steps':>12s}"
        print(header)
        print("-" * len(header))
        for r in all_results:
            print(
                f"{r['ckpt']:15s} "
                f"{r['mode']:8s} "
                f"{r['mean_reward']:10.3f} "
                f"{r['std_reward']:10.3f} "
                f"{r['mean_steps']:12.2f}"
            )

        # ===== 导出 CSV =====
        out_dir = ROOT / "scripts" / "runs" / "ppo_cskg"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_csv = out_dir / f"eval_compare_{int(time.time())}.csv"

        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["ckpt", "mode", "mean_reward", "std_reward", "mean_steps"])
            for r in all_results:
                writer.writerow([
                    r["ckpt"],
                    r["mode"],
                    f"{r['mean_reward']:.6f}",
                    f"{r['std_reward']:.6f}",
                    f"{r['mean_steps']:.6f}",
                ])

        print(f"\n📄 已将汇总结果写入: {out_csv}")

    print("\n✅ 对比评估完成")


if __name__ == "__main__":
    main()
