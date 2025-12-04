# scripts/agent/train_ppo_cskg.py
# -*- coding: utf-8 -*-
"""
PPO + CSKG(KnowledgeBridge) 联合训练脚本

- 使用 CybORGWrapper 包装环境
- 使用 KnowledgeBridge 注入：
    - 动作掩码（action_mask）
    - 先验 logits（prior_logits）
    - 奖励塑形（reward_shaping）
- 保留 PPO 足够自由度，让策略自己学，而不是被规则“锁死”
"""

import os
import sys
import json
import time
import pathlib
import random
import argparse
from typing import Any, Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

# ===== 路径注入 =====
ROOT = pathlib.Path(__file__).resolve().parents[2]  # C:\cybdef
THIRD = ROOT / "third_party" / "CybORG"

for p in (ROOT, THIRD):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

# ===== 项目内 import =====
# env wrapper
try:
    from envs.cyborg_wrapper import CybORGWrapper
except ImportError:
    from scripts.envs.cyborg_wrapper import CybORGWrapper

# CSKG reasoner
from scripts.cskg.reasoner import KnowledgeBridge

import yaml

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def to_serializable(obj):
    """
    递归把 numpy 类型、ndarray 等，转成 Python 原生类型，方便 json.dumps
    """
    import numpy as _np

    if isinstance(obj, (_np.floating,)):
        return float(obj)
    if isinstance(obj, (_np.integer,)):
        return int(obj)
    if isinstance(obj, _np.ndarray):
        return obj.tolist()
    if isinstance(obj, (list, tuple)):
        return [to_serializable(v) for v in obj]
    if isinstance(obj, dict):
        return {k: to_serializable(v) for k, v in obj.items()}
    return obj


def load_yaml(path: pathlib.Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def build_paths(cfg: Dict[str, Any]) -> Dict[str, pathlib.Path]:
    paths_cfg = cfg.get("paths", {})
    return {
        "env": ROOT / paths_cfg.get("env_config", "scripts/configs/env.yaml"),
        "ppo": ROOT / paths_cfg.get("ppo_config", "scripts/configs/ppo.yaml"),
        "cskg": ROOT / paths_cfg.get("cskg_config", "scripts/configs/cskg.yaml"),
        "seed_graph": ROOT / paths_cfg.get("seed_graph", "scripts/configs/seed_graph.json"),
    }

def dump_run_metadata(out_dir: pathlib.Path, exp_cfg: Dict[str, Any], ppo_cfg: Dict[str, Any]) -> None:
    """将实验与超参快照写入日志目录，方便复现。"""

    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "exp_config_snapshot.json", "w", encoding="utf-8") as f:
        json.dump(to_serializable(exp_cfg), f, ensure_ascii=False, indent=2)

    with open(out_dir / "ppo_config_snapshot.json", "w", encoding="utf-8") as f:
        json.dump(to_serializable(ppo_cfg), f, ensure_ascii=False, indent=2)


# ===== 简单 Actor-Critic 网络 =====
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


# ===== GAE 计算 =====
def compute_gae(rewards, values, dones, gamma=0.99, lam=0.95):
    """
    rewards, values, dones: np.ndarray, shape [T]
    """
    T = len(rewards)
    adv = np.zeros(T, dtype=np.float32)
    gae = 0.0
    for t in reversed(range(T)):
        next_v = values[t + 1] if t + 1 < T else 0.0
        delta = rewards[t] + gamma * next_v * (1.0 - dones[t]) - values[t]
        gae = delta + gamma * lam * (1.0 - dones[t]) * gae
        adv[t] = gae
    return adv


def to_obs_vector(obs_raw: Any) -> np.ndarray:
    """
    把 env.reset()/step() 返回的各种结构，统一转成 1D np.array(float32)

    支持几种常见形式：
    - dict:
        - 包含 "obs_vec"（BlueTableWrapper 风格）
        - 或包含 "obs"/"observation"/"vector"/"state" 之一
    - 其它：直接 np.array(...)
    """
    if isinstance(obs_raw, dict):
        # 优先走你包装好的 obs_vec
        if "obs_vec" in obs_raw:
            obs_raw = obs_raw["obs_vec"]
        else:
            # 通用兜底：兼容老版本
            for key in ["obs", "observation", "vector", "state"]:
                if key in obs_raw:
                    obs_raw = obs_raw[key]
                    break

    # 如果还是字典，说明没法拿到向量
    if isinstance(obs_raw, dict):
        raise TypeError(
            f"无法从 obs 字典中提取向量，请检查 keys: {list(obs_raw.keys())}"
        )

    # 与早期版本的 to_obs_vector 保持一致：无论输入是 list/np.ndarray/标量都展开为 1D float32
    obs_np = np.array(obs_raw, dtype=np.float32).reshape(-1)
    return obs_np

def select_action(
    ac: ActorCritic,
    kb: KnowledgeBridge | None,
    env: CybORGWrapper,
    obs_vec: np.ndarray,
    facts: Dict[str, Any],
    action_names: list[str],
    rule_coef: float,
) -> Tuple[int, float, float, Dict[str, Any]]:
    """前向、融合先验/掩码并采样动作，返回 action_idx、logp、value、调试信息。"""

    obs_tensor = torch.from_numpy(obs_vec).to(DEVICE).unsqueeze(0)
    logits, value = ac(obs_tensor)  # [1, act_dim], [1]
    logits = logits.squeeze(0)  # [act_dim]
    value = value.squeeze(0)

    # === CSKG: 先用 facts 更新内部状态（可选）===
    if kb and hasattr(kb, "update_from_facts"):
        kb.update_from_facts(facts)

    # === 从 KB 获取先验与掩码 ===
    if kb:
        prior_np = kb.prior_logits(facts, action_names)
        if isinstance(prior_np, tuple):
            prior_np = prior_np[0]
        prior_np = np.array(prior_np, dtype=np.float32)

        mask_res = kb.query_action_mask(facts, action_names)
        if isinstance(mask_res, tuple):
            rule_mask_np = np.array(mask_res[0], dtype=np.float32)
        else:
            rule_mask_np = np.array(mask_res, dtype=np.float32)
    else:
        prior_np = np.zeros(len(action_names), dtype=np.float32)
        rule_mask_np = np.ones(len(action_names), dtype=np.float32)

    # 环境自带合法掩码
    try:
        legal_mask_np = env._current_legal_mask().astype(np.float32)
    except Exception:
        legal_mask_np = np.ones(len(action_names), dtype=np.float32)

    # 融合掩码（环境 × 规则）
    combined_mask_np = (legal_mask_np * rule_mask_np).astype(np.float32)
    if combined_mask_np.sum() <= 0:
        combined_mask_np[0] = 1.0  # 避免死锁

    logits = logits.clone()
    prior = torch.from_numpy(prior_np).to(DEVICE)
    logits = logits + (rule_coef * prior if rule_coef != 0.0 else prior)

    combined_mask_t = torch.from_numpy(combined_mask_np).to(DEVICE)
    logits[combined_mask_t == 0] = -1e9

    dist = Categorical(logits=logits)
    action = dist.sample()
    logp = dist.log_prob(action)

    debug_info = {
        "prior_np": prior_np,
        "rule_mask_np": rule_mask_np,
        "legal_mask_np": legal_mask_np,
        "combined_mask_np": combined_mask_np,
    }

    return int(action.item()), float(logp.item()), float(value.item()), debug_info


# ===== 主训练函数 =====
def main(config_path: str | None = None):
    global DEVICE

    if config_path is None:
        parser = argparse.ArgumentParser(description="PPO 训练入口")
        parser.add_argument(
            "--config",
            type=str,
            default=str(ROOT / "scripts" / "configs" / "b1.yaml"),
            help="实验配置文件（B0/B1/B2）",
        )
        args = parser.parse_args()
        config_path = args.config

    exp_cfg = load_yaml(pathlib.Path(config_path))
    if not exp_cfg:
        raise FileNotFoundError(f"无法加载配置: {config_path}")

    paths = build_paths(exp_cfg)

    exp_meta = exp_cfg.get("experiment", {})
    features = exp_cfg.get("features", {})
    logging_cfg = exp_cfg.get("logging", {})

    run_prefix = logging_cfg.get("run_prefix", "ppo_cskg")
    run_id = exp_meta.get("id", "exp")

    RUN_NAME = f"{run_prefix.lower()}_{run_id.lower()}_{int(time.time())}"
    OUT_DIR = ROOT / "scripts" / "runs" / run_prefix / RUN_NAME

    os.makedirs(OUT_DIR, exist_ok=True)

    print(f"🔧 运行 ID: {RUN_NAME}")
    print(
        "   特性: CSKG={}  RAG_explain={}".format(
            features.get("enable_cskg", True), features.get("enable_rag_explain", False)
        )
    )

    # --- 从 ppo.yaml 读取超参 ---
    ppo_yaml = paths["ppo"]
    if ppo_yaml.exists():
        with open(ppo_yaml, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
    else:
        cfg = {}
        print(f"⚠ 未找到 {ppo_yaml}，将使用代码内默认超参")

    num_updates = int(cfg.get("num_updates", 100))        # 训练轮数（原 total_episodes）
    rollout_steps = int(cfg.get("horizon", 256))          # 每轮采样步数（原 rollout_steps）
    ppo_epochs = int(cfg.get("ppo_epochs", 4))
    batch_size = int(cfg.get("mini_batch_size", 64))
    gamma = float(cfg.get("gamma", 0.99))
    lam = float(cfg.get("gae_lambda", 0.95))
    clip_ratio = float(cfg.get("clip_range", 0.2))

    lr_pi = float(cfg.get("pi_lr", 3e-4))
    lr_vf = float(cfg.get("vf_lr", lr_pi))  # 目前仍共用一个 optimizer
    lr = lr_pi

    entropy_coef = float(cfg.get("entropy_coef", 0.01))
    value_coef = float(cfg.get("value_coef", 0.5))
    rule_coef = float(cfg.get("rule_coef", 0.0))          # 用于缩放 prior logits
    max_grad_norm = float(cfg.get("max_grad_norm", 0.5))

    device_cfg = str(cfg.get("device", "cuda")).lower()

    # --- 设备选择：优先按 ppo.yaml，但要保证可用 ---
    if device_cfg == "cuda" and torch.cuda.is_available():
        DEVICE = torch.device("cuda")
    else:
        DEVICE = torch.device("cpu")

    dump_run_metadata(OUT_DIR, exp_cfg, cfg)

    print(f"📋 实验配置: {config_path}")
    print(f"📋 PPO 配置来自: {ppo_yaml}")
    print(f"   num_updates={num_updates}, horizon={rollout_steps}, "
          f"mini_batch_size={batch_size}, ppo_epochs={ppo_epochs}")
    print(f"   gamma={gamma}, gae_lambda={lam}, clip_range={clip_ratio}")
    print(f"   pi_lr={lr_pi}, vf_lr={lr_vf}, entropy_coef={entropy_coef}, value_coef={value_coef}")
    print(f"   rule_coef={rule_coef}, max_grad_norm={max_grad_norm}")
    print(f"   device={DEVICE}")

    # 固定随机种子
    seed = int(exp_meta.get("seed", 42))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # --- 初始化环境 ---
    env = CybORGWrapper(str(paths["env"]))
    obs_dim = env.obs_dim
    act_dim = env.action_dim

    print(f"✅ PPO 训练初始化完成")
    print(f"   obs_dim={obs_dim}, act_dim={act_dim}")
    print(f"   日志目录: {OUT_DIR}")

    enable_cskg = bool(features.get("enable_cskg", True))
    enable_rag = bool(features.get("enable_rag_explain", False))
    kb = None
    if enable_cskg:
        kb = KnowledgeBridge(
            seed_graph_path=str(paths["seed_graph"]),
            cskg_rules_path=str(paths["cskg"]),
            recent_steps=10,
        )
    elif enable_rag:
        print("⚠ RAG 解释被启用但 CSKG 关闭，将跳过 KB 相关日志")

    # --- 初始化策略网络 ---
    ac = ActorCritic(obs_dim, act_dim).to(DEVICE)
    optimizer = optim.Adam(ac.parameters(), lr=lr)

    # --- 可解释日志：前 N 次 update 详细记录 ---
    explain_log_path = OUT_DIR / "policy_explain_upd1_5.jsonl"
    explain_log_f = None
    if enable_cskg or enable_rag:
        explain_log_f = open(explain_log_path, "w", encoding="utf-8")

    global_step = 0

    # ===== 训练主循环：以 num_updates 为外层轮数 =====
    for upd in range(1, num_updates + 1):
        # env.reset() 返回 dict: {"obs_vec", "facts", "raw", ...}
        obs_raw = env.reset()
        if kb and hasattr(kb, "reset_episode"):
            kb.reset_episode()

        # 神经网络用的向量观测
        obs_vec = to_obs_vector(obs_raw)
        # 规则引擎用的语义 facts（来自 wrapper，而不是再从 obs_vec 反推）
        facts = obs_raw.get("facts", {})

        # rollout buffer
        obs_buf = []
        act_buf = []
        logp_buf = []
        rew_buf = []
        val_buf = []
        done_buf = []

        ep_reward_env = 0.0
        ep_reward_total = 0.0


        # 一次 update 内采样 rollout_steps 步（可能跨 episode，中途 done 就重置）
        steps_collected = 0
        while steps_collected < rollout_steps:
            global_step += 1

            action_names = env.action_space.names

            action_idx, logp, value, debug_info = select_action(
                ac, kb, env, obs_vec, facts, action_names, rule_coef
            )
            action_name = action_names[action_idx]

            # === 与环境交互 ===
            next_obs_raw, reward_env, done, info = env.step(action_idx)

            # === 修正：正确的奖励塑形 ===
            env_reward = float(reward_env)  # 环境原始奖励（真实性能）

            # CSKG奖励塑形（基于环境奖励）
            if kb and hasattr(kb, "step_update"):
                shaped_reward = kb.step_update(facts, action_name, env_reward)
            else:
                shaped_reward = env_reward

            # 关键设计：训练用塑形奖励，评估用环境奖励
            if env.mode == "train":
                r_total = shaped_reward  # PPO用CSKG指导的训练信号
            else:
                r_total = env_reward  # 评估时用真实环境奖励

            # === 修正：正确的KB状态更新 ===
            next_facts = next_obs_raw.get("facts", {})
            # 如果有专门的KB状态更新方法，在这里调用（但step_update可能已经处理了）
            # 注意：我们不再重复调用step_update，因为它已经返回了塑形奖励

            last_reward_env = env_reward  # 用于_extract_facts的recent_reward

            # ==== 写入 rollout buffer（用 r_total 来训练 PPO） ====
            obs_buf.append(obs_vec.copy())
            act_buf.append(action_idx)
            logp_buf.append(logp)
            rew_buf.append(float(r_total))  # PPO用训练信号
            val_buf.append(value)
            done_buf.append(float(done))

            # 分别记录两种奖励用于分析
            ep_reward_env += env_reward  # 真实环境表现
            ep_reward_total += r_total  # 实际训练信号

            steps_collected += 1

            # === 可解释日志：前 5 次 update 详细记录 ===
            if upd <= 5 and (enable_cskg or enable_rag):
                prior_np = debug_info["prior_np"]
                top_idx = np.argsort(prior_np)[-3:][::-1]
                top_prior = [[action_names[i], float(prior_np[i])] for i in top_idx]
                try:
                    explain = kb.explain_decision(facts, action_names) if kb else {}
                except Exception:
                    explain = {}

                explain_rec = {
                    "update": upd,
                    "step": steps_collected,
                    "global_step": global_step,
                    "action_idx": action_idx,
                    "action_name": action_name,
                    "reward_env": float(env_reward),  # 记录环境奖励
                    "reward_shaped": float(shaped_reward),  # 记录塑形奖励
                    "reward_total": float(r_total),  # 记录实际训练信号
                    "legal_mask_sum": float(debug_info["legal_mask_np"].sum()),
                    "rule_mask_sum": float(debug_info["rule_mask_np"].sum()),
                    "combined_mask_sum": float(debug_info["combined_mask_np"].sum()),
                    "top_prior": top_prior,
                    "fact": facts,
                    "explain": explain,
                }
                explain_log_f.write(
                    json.dumps(to_serializable(explain_rec), ensure_ascii=False) + "\n"
                )

            # === 准备下一步 ===
            obs_vec = to_obs_vector(next_obs_raw)  # 仅用于 NN
            facts = next_facts  # 下一步规则使用的 facts

            # 如果 episode 结束，重置环境 + KB，但继续本次 update 直到收满 horizon
            if done and steps_collected < rollout_steps:
                obs_raw = env.reset()
                if hasattr(kb, "reset_episode"):
                    kb.reset_episode()
                obs_vec = to_obs_vector(obs_raw)
                facts = obs_raw.get("facts", {})
                last_reward_env = 0.0

        # ===== 一次 update 结束：PPO 更新 =====
        T = len(rew_buf)
        if T == 0:
            continue

        rewards = np.array(rew_buf, dtype=np.float32)
        values = np.array(val_buf, dtype=np.float32)
        dones = np.array(done_buf, dtype=np.float32)

        # 末值 bootstrap = 0（这里简单处理）
        values_ext = np.concatenate([values, np.array([0.0], dtype=np.float32)], axis=0)

        adv = compute_gae(rewards, values_ext, dones, gamma=gamma, lam=lam)
        returns = adv + values

        # 标准化优势
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        # 转 tensor
        obs_tensor = torch.from_numpy(np.array(obs_buf, dtype=np.float32)).to(DEVICE)
        act_tensor = torch.from_numpy(np.array(act_buf, dtype=np.int64)).to(DEVICE)
        logp_old_tensor = torch.from_numpy(np.array(logp_buf, dtype=np.float32)).to(DEVICE)
        adv_tensor = torch.from_numpy(adv).to(DEVICE)
        ret_tensor = torch.from_numpy(returns).to(DEVICE)

        # 多 epoch 打乱训练
        num_samples = T
        idxs = np.arange(num_samples)

        for _ in range(ppo_epochs):
            np.random.shuffle(idxs)
            for start in range(0, num_samples, batch_size):
                end = start + batch_size
                batch_idx = idxs[start:end]

                b_obs = obs_tensor[batch_idx]
                b_act = act_tensor[batch_idx]
                b_logp_old = logp_old_tensor[batch_idx]
                b_adv = adv_tensor[batch_idx]
                b_ret = ret_tensor[batch_idx]

                logits, values_pred = ac(b_obs)
                dist = Categorical(logits=logits)
                logp = dist.log_prob(b_act)
                entropy = dist.entropy().mean()

                ratio = torch.exp(logp - b_logp_old)
                surr1 = ratio * b_adv
                surr2 = torch.clamp(ratio, 1.0 - clip_ratio, 1.0 + clip_ratio) * b_adv
                policy_loss = -torch.min(surr1, surr2).mean()

                value_loss = ((values_pred - b_ret) ** 2).mean()

                loss = policy_loss + value_coef * value_loss - entropy_coef * entropy

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(ac.parameters(), max_grad_norm)
                optimizer.step()

        # 打印时显示两种奖励
        print(
            f"[UPD {upd:03d}] steps={T:4d}  "
            f"Env_R={ep_reward_env:.3f}  Shaped_R={ep_reward_total:.3f}"
        )

        # 简单保存 checkpoint（每 25 次 update）
        if upd % 25 == 0:
            ckpt_path = OUT_DIR / f"ac_upd{upd:03d}.pt"
            torch.save(
                {
                    "model": ac.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "update": upd,
                    "global_step": global_step,
                },
                ckpt_path,
            )
            print(f" 💾 已保存 checkpoint: {ckpt_path}")

    if explain_log_f is not None:
        explain_log_f.close()
    print("✅ 训练结束")


if __name__ == "__main__":
    main()
