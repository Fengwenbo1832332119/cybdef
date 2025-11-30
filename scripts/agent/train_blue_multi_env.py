# scripts/agent/train_blue_multi_env.py
# -*- coding: utf-8 -*-
"""
Multi-scenario PPO training for a single Blue agent.

Envs:
    - CybORG Scenario2 (through CybORGWrapper)
    - ICS (PrimAITE, ics.yaml)
    - LOT (PrimAITE, lot.yaml)
    - Robotics (PrimAITE, robotics.yaml)

One shared policy (Actor-Critic) with unified obs_dim / act_dim:
    - obs_dim = max(obs_dim of all envs)  (目前 2781)
    - act_dim = max(act_dim of all envs)  (目前 145)

This script supports two modes:
    - Pure PPO baseline (default)
    - PPO + weak CSKG on CybORG only (soft prior + reward shaping via MultiEnvKB)

用法示例（PowerShell）：

    conda activate primaite311
    cd C:\cybdef

    # 纯 PPO 多场景训练
    python scripts/agent/train_blue_multi_env.py `
        --ppo-config C:\cybdef\scripts\configs\ppo.yaml `
        --total-updates 200 `
        --horizon 256 `
        --run-prefix multi_blue_pure

    # 开启 CybORG weak-CSKG
    python scripts/agent/train_blue_multi_env.py `
        --ppo-config C:\cybdef\scripts\configs\ppo.yaml `
        --total-updates 200 `
        --horizon 256 `
        --enable-cskg `
        --run-prefix multi_blue_cskg

"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

import pathlib

import matplotlib
matplotlib.use("Agg")  # 无界面后端，方便服务器/终端跑
import matplotlib.pyplot as plt

try:
    from torch.utils.tensorboard import SummaryWriter
except Exception:  # 没装 tensorboard 也能跑
    SummaryWriter = None

# ===== 路径注入 =====
ROOT = pathlib.Path(__file__).resolve().parents[2]  # C:\cybdef
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# MultiEnvWrapper: 你之前已经创建并测试过的那个
from scripts.envs.multi_env_wrapper import MultiEnvWrapper

# MultiEnvKB：多场景 KB 管理器（目前只挂 cyborg）
from scripts.cskg.multi_kb import MultiEnvKB

import yaml

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# CSKG 配置路径（沿用你单场景时的那套）
CSKG_CFG_PATH = ROOT / "scripts" / "configs" / "cskg_cyborg_weak.yaml"
SEED_GRAPH_PATH = ROOT / "scripts" / "configs" / "seed_graph.json"
CYBORG_ENV_CFG_PATH = ROOT / "scripts" / "configs" / "env.yaml"

ICSKG_CFG_PATH = ROOT / "scripts" / "configs" / "cskg_ics_weak.yaml"
ICS_SEED_GRAPH_PATH = SEED_GRAPH_PATH
ICS_ENV_CFG_PATH = (
    ROOT
    / "third_party"
    / "PrimAITE"
    / "src"
    / "primaite"
    / "config"
    / "_package_data"
    / "ics.yaml"
)



# 一个很小的系数，让先验真的很「weak」
CSKG_PRIOR_COEF = 0.2


# ===== 工具函数 =====
def load_yaml(path: str | pathlib.Path) -> Dict[str, Any]:
    path = pathlib.Path(path)
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def to_serializable(obj):
    """便于 json.dump：把 numpy / ndarray 转成 Python 原生类型。"""
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


def to_obs_vector(obs_raw: Any) -> np.ndarray:
    """
    统一把 env.reset()/step() 返回的 obs 转成 1D np.array(float32)。

    约定：
    - 如果是 dict，优先取 obs["obs_vec"]
    - 否则直接 np.array(...).reshape(-1)
    """
    if isinstance(obs_raw, dict):
        if "obs_vec" in obs_raw:
            obs_raw = obs_raw["obs_vec"]
        else:
            # 兜底：尝试常见 key
            for k in ["obs", "observation", "vector", "state"]:
                if k in obs_raw:
                    obs_raw = obs_raw[k]
                    break

    obs_np = np.array(obs_raw, dtype=np.float32).reshape(-1)
    return obs_np


# ===== PPO 相关 =====
@dataclass
class PpoConfig:
    num_updates: int
    horizon: int
    ppo_epochs: int
    mini_batch_size: int
    gamma: float
    gae_lambda: float
    clip_range: float
    pi_lr: float
    vf_lr: float
    entropy_coef: float
    value_coef: float
    max_grad_norm: float


def load_ppo_config(path: str | pathlib.Path) -> PpoConfig:
    cfg = load_yaml(path)

    return PpoConfig(
        num_updates=int(cfg.get("num_updates", 100)),
        horizon=int(cfg.get("horizon", 256)),
        ppo_epochs=int(cfg.get("ppo_epochs", 4)),
        mini_batch_size=int(cfg.get("mini_batch_size", 64)),
        gamma=float(cfg.get("gamma", 0.99)),
        gae_lambda=float(cfg.get("gae_lambda", 0.95)),
        clip_range=float(cfg.get("clip_range", 0.2)),
        pi_lr=float(cfg.get("pi_lr", 3e-4)),
        vf_lr=float(cfg.get("vf_lr", cfg.get("pi_lr", 3e-4))),
        entropy_coef=float(cfg.get("entropy_coef", 0.01)),
        value_coef=float(cfg.get("value_coef", 0.5)),
        max_grad_norm=float(cfg.get("max_grad_norm", 0.5)),
    )


class ActorCritic(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden: int = 256):
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

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        logits = self.actor(obs)
        value = self.critic(obs).squeeze(-1)
        return logits, value


def compute_gae(
    rewards: np.ndarray,
    values: np.ndarray,
    dones: np.ndarray,
    gamma: float,
    lam: float,
) -> np.ndarray:
    """Standard GAE(λ) 计算。"""
    T = len(rewards)
    adv = np.zeros(T, dtype=np.float32)
    gae = 0.0
    for t in reversed(range(T)):
        next_v = values[t + 1] if t + 1 < T else 0.0
        delta = rewards[t] + gamma * next_v * (1.0 - dones[t]) - values[t]
        gae = delta + gamma * lam * (1.0 - dones[t]) * gae
        adv[t] = gae
    return adv


def main():
    parser = argparse.ArgumentParser(description="Multi-scenario PPO training for Blue agent")
    parser.add_argument(
        "--ppo-config",
        type=str,
        default=str(ROOT / "scripts" / "configs" / "ppo.yaml"),
        help="Path to PPO hyper-parameters YAML.",
    )
    parser.add_argument(
        "--total-updates",
        type=int,
        default=-1,
        help="Override num_updates in ppo.yaml (if >0).",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=-1,
        help="Override horizon in ppo.yaml (if >0).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )
    parser.add_argument(
        "--run-prefix",
        type=str,
        default="multi_blue",
        help="Prefix for run directory name.",
    )
    parser.add_argument(
        "--enable-cskg",
        action="store_true",
        help="If set, enable weak CSKG (prior + reward shaping) for CybORG only via MultiEnvKB.",
    )

    args = parser.parse_args()

    # --- 加载 PPO 超参 ---
    ppo_cfg = load_ppo_config(args.ppo_config)
    if args.total_updates > 0:
        ppo_cfg.num_updates = args.total_updates
    if args.horizon > 0:
        ppo_cfg.horizon = args.horizon

    # --- 设备选择 ---
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    global DEVICE
    DEVICE = device

    # --- 固定随机种子 ---
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # --- 输出目录 ---
    run_name = f"{args.run_prefix}_{int(time.time())}"
    out_dir = ROOT / "scripts" / "runs" / args.run_prefix / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"🔧 Run ID: {run_name}")
    print(f"📋 PPO config: {args.ppo_config}")
    print(
        f"   num_updates={ppo_cfg.num_updates}, horizon={ppo_cfg.horizon}, "
        f"mini_batch_size={ppo_cfg.mini_batch_size}, ppo_epochs={ppo_cfg.ppo_epochs}"
    )
    print(
        f"   gamma={ppo_cfg.gamma}, gae_lambda={ppo_cfg.gae_lambda}, clip_range={ppo_cfg.clip_range}"
    )
    print(
        f"   pi_lr={ppo_cfg.pi_lr}, vf_lr={ppo_cfg.vf_lr}, "
        f"entropy_coef={ppo_cfg.entropy_coef}, value_coef={ppo_cfg.value_coef}, "
        f"max_grad_norm={ppo_cfg.max_grad_norm}"
    )
    print(f"   device={DEVICE}")
    print(f"   enable_cskg={args.enable_cskg}")

    # --- 保存一份超参快照 ---
    with open(out_dir / "ppo_config_snapshot.json", "w", encoding="utf-8") as f:
        json.dump(to_serializable(ppo_cfg.__dict__), f, ensure_ascii=False, indent=2)

    # --- TensorBoard writer（可选） ---
    writer = SummaryWriter(log_dir=str(out_dir)) if SummaryWriter is not None else None

    # 用于画 reward 曲线
    update_ids: list[int] = []
    return_hist: list[float] = []

    # --- 初始化 MultiEnvWrapper ---
    env = MultiEnvWrapper(
        env_names=["cyborg", "ics", "lot", "robotics"],
        weights=[0.0, 1.0, 0.0, 0.0],  # 这里其实是只训练 ICS，你后面可以再调
        mode="train",
    )

    # 统一后的全局 obs_dim / act_dim
    if hasattr(env, "obs_dim"):
        obs_dim = env.obs_dim
    else:
        obs0 = env.reset()
        obs_dim = len(to_obs_vector(obs0))

    if hasattr(env, "action_dim"):
        act_dim = env.action_dim
    else:
        if hasattr(env, "action_space") and hasattr(env.action_space, "n"):
            act_dim = env.action_space.n
        else:
            raise RuntimeError("Cannot infer action_dim from MultiEnvWrapper.")

    print(f"✅ Multi-env PPO init: obs_dim={obs_dim}, act_dim={act_dim}")
    print(f"   log_dir={out_dir}")

    # --- 初始化策略网络 ---
    ac = ActorCritic(obs_dim=obs_dim, act_dim=act_dim, hidden=256).to(DEVICE)
    optimizer = optim.Adam(ac.parameters(), lr=ppo_cfg.pi_lr)

    multi_kb: MultiEnvKB | None = None

    # ====== 初始化 MultiEnvKB（cyborg + ics）======
    if args.enable_cskg:
        try:
            from scripts.envs.cyborg_wrapper import CybORGWrapper
            from scripts.envs.primaite_wrapper import PrimaiteWrapper

            # 1) CybORG 动作名 —— 用 wrapper 自己的 action_names
            tmp_cyb = CybORGWrapper(str(CYBORG_ENV_CFG_PATH))
            cyborg_action_names = list(tmp_cyb.action_names)
            tmp_cyb.close()

            # 2) ICS 动作名 —— 也是 wrapper 的 action_names（你在 PrimaiteWrapper 里已经定义了）
            tmp_ics = PrimaiteWrapper(str(ICS_ENV_CFG_PATH))
            ics_action_names = list(tmp_ics.action_names)
            tmp_ics.close()

            # 3) 按 MultiEnvKB.from_env_specs 要求组装 env_specs
            env_specs = {
                "cyborg": {
                    "seed_graph": SEED_GRAPH_PATH,
                    "cskg": CSKG_CFG_PATH,
                    "action_names": cyborg_action_names,
                },
                "ics": {
                    # 现在先共用一个 seed_graph，后面你有单独 ICS 版再换
                    "seed_graph": ICS_SEED_GRAPH_PATH,
                    "cskg": ICSKG_CFG_PATH,
                    "action_names": ics_action_names,
                },
                # 以后可以在这里继续加 "lot" / "robotics"
            }

            multi_kb = MultiEnvKB.from_env_specs(env_specs, recent_steps=10)
            print(
                "🧠 MultiEnvKB 初始化完成：挂载场景 = "
                + ", ".join(list(env_specs.keys()))
            )
        except Exception as e:
            print(f"⚠ 无法初始化 MultiEnvKB（cyborg + ics），CSKG 将被禁用: {e}")
            multi_kb = None
    else:
        multi_kb = None

    global_step = 0

    # ===== 训练主循环 =====
    for upd in range(1, ppo_cfg.num_updates + 1):
        # 重置环境（MultiEnvWrapper 会内部随机选一个场景）
        obs_raw = env.reset()
        obs_vec = to_obs_vector(obs_raw)
        facts = obs_raw.get("facts", {}) if isinstance(obs_raw, dict) else {}

        # rollout buffer
        obs_buf: List[np.ndarray] = []
        act_buf: List[int] = []
        logp_buf: List[float] = []
        rew_buf: List[float] = []
        val_buf: List[float] = []
        done_buf: List[float] = []

        ep_return = 0.0
        steps_collected = 0

        while steps_collected < ppo_cfg.horizon:
            global_step += 1

            obs_tensor = torch.from_numpy(obs_vec).to(DEVICE).unsqueeze(0)
            logits, value = ac(obs_tensor)
            logits = logits.squeeze(0)
            value = value.squeeze(0)

            # 当前子环境名
            cur_env_name = getattr(env, "current_env_name", None)

            # === CSKG 先验（weak）===
            prior_np = None
            if multi_kb is not None and cur_env_name is not None:
                prior_np = multi_kb.prior_logits(
                    env_name=cur_env_name,
                    facts=facts,
                    global_act_dim=act_dim,
                )
                prior_t = torch.from_numpy(prior_np).to(DEVICE)
                logits = logits + CSKG_PRIOR_COEF * prior_t

                if cur_env_name in ("cyborg", "ics") and steps_collected < 20:
                    tag = cur_env_name.upper()
                    print(
                        f"[DEBUG-{tag}-CSKG][prior] upd={upd} "
                        f"step={steps_collected} env={cur_env_name}, coef={CSKG_PRIOR_COEF}"
                    )
                    print("  prior_np[:10] =", prior_np[:10])

            # === 每个场景自己的合法动作掩码 ===
            try:
                mask_np = env.current_action_mask().astype(np.float32)
            except AttributeError:
                mask_np = np.ones(act_dim, dtype=np.float32)

            if mask_np.size != act_dim:
                mask_np = np.ones(act_dim, dtype=np.float32)

            mask_t = torch.from_numpy(mask_np).to(DEVICE)
            logits = logits.clone()
            logits[mask_t == 0] = -1e9

            # === PPO 采样 ===
            dist = Categorical(logits=logits)
            action = dist.sample()
            logp = dist.log_prob(action)
            a_idx = int(action.item())

            # 与多环境交互
            next_obs_raw, reward, done, info = env.step(a_idx)

            env_r = float(reward)
            d = bool(done)
            next_facts = (
                next_obs_raw.get("facts", {}) if isinstance(next_obs_raw, dict) else {}
            )

            # === CSKG 奖励塑形 ===
            r = env_r
            if multi_kb is not None and cur_env_name is not None:
                r = multi_kb.shape_reward(
                    env_name=cur_env_name,
                    facts=next_facts,
                    action_idx=a_idx,
                    env_reward=env_r,
                )
                if cur_env_name in ("cyborg", "ics") and steps_collected < 20:
                    tag = cur_env_name.upper()
                    print("[DEBUG-ICS-FACTS]", next_facts)
                    print(
                        f"[DEBUG-{tag}-CSKG][reward] upd={upd} "
                        f"step={steps_collected} env={cur_env_name}, act={a_idx}, "
                        f"env_r={env_r:.4f}, shaped_r={r:.4f}"
                    )

            # === 每个场景自己的合法动作掩码 ===
            try:
                mask_np = env.current_action_mask().astype(np.float32)
            except AttributeError:
                mask_np = np.ones(act_dim, dtype=np.float32)

            if mask_np.size != act_dim:
                mask_np = np.ones(act_dim, dtype=np.float32)

            mask_t = torch.from_numpy(mask_np).to(DEVICE)
            logits = logits.clone()
            logits[mask_t == 0] = -1e9  # 把非法动作 logit 压到极小

            # === PPO 采样 ===
            dist = Categorical(logits=logits)
            action = dist.sample()
            logp = dist.log_prob(action)

            a_idx = int(action.item())

            # 与多环境交互
            next_obs_raw, reward, done, info = env.step(a_idx)

            env_r = float(reward)  # 环境原始奖励
            d = bool(done)
            next_facts = next_obs_raw.get("facts", {}) if isinstance(next_obs_raw, dict) else {}

            # === （可选）CSKG 奖励塑形，只对有 KB 的场景生效 ===
            r = env_r
            if multi_kb is not None and cur_env_name is not None:
                r = multi_kb.shape_reward(
                    env_name=cur_env_name,
                    facts=next_facts,
                    action_idx=a_idx,
                    env_reward=env_r,
                )

                if cur_env_name in ("cyborg", "ics") and steps_collected < 20:
                    print("[DEBUG-ICS-FACTS]", next_facts)
                    tag = cur_env_name.upper()
                    print(
                        f"[DEBUG-{tag}-CSKG][reward] upd={upd} "
                        f"step={steps_collected} env={cur_env_name}, act={a_idx}, "
                        f"env_r={env_r:.4f}, shaped_r={r:.4f}"
                    )

            # buffer 记录（PPO 用的是 r = 训练信号）
            obs_buf.append(obs_vec.copy())
            act_buf.append(a_idx)
            logp_buf.append(float(logp.item()))
            rew_buf.append(r)
            val_buf.append(float(value.item()))
            done_buf.append(float(d))

            ep_return += r
            steps_collected += 1

            # 准备下一步
            obs_vec = to_obs_vector(next_obs_raw)
            facts = next_facts

            # 如果 episode 提前结束，但当前 update 还没收满 horizon，就重置继续
            if d and steps_collected < ppo_cfg.horizon:
                if multi_kb is not None and cur_env_name is not None:
                    multi_kb.reset_episode(cur_env_name)

                obs_raw = env.reset()
                obs_vec = to_obs_vector(obs_raw)
                facts = obs_raw.get("facts", {}) if isinstance(obs_raw, dict) else {}

        # ===== 一次 update 的 rollout 收集完毕，开始 PPO 更新 =====
        T = len(rew_buf)
        if T == 0:
            continue

        rewards = np.asarray(rew_buf, dtype=np.float32)
        values = np.asarray(val_buf, dtype=np.float32)
        dones = np.asarray(done_buf, dtype=np.float32)

        # bootstrap value 简单用 0（你以后可以改成用最后一个 obs 再 forward 一次）
        values_ext = np.concatenate([values, np.array([0.0], dtype=np.float32)], axis=0)

        adv = compute_gae(
            rewards=rewards,
            values=values_ext,
            dones=dones,
            gamma=ppo_cfg.gamma,
            lam=ppo_cfg.gae_lambda,
        )
        returns = adv + values

        # 标准化优势
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        # 转 tensor
        obs_tensor = torch.from_numpy(np.asarray(obs_buf, dtype=np.float32)).to(DEVICE)
        act_tensor = torch.from_numpy(np.asarray(act_buf, dtype=np.int64)).to(DEVICE)
        logp_old_tensor = torch.from_numpy(np.asarray(logp_buf, dtype=np.float32)).to(DEVICE)
        adv_tensor = torch.from_numpy(adv.astype(np.float32)).to(DEVICE)
        ret_tensor = torch.from_numpy(returns.astype(np.float32)).to(DEVICE)

        # 多 epoch + mini-batch 训练
        num_samples = T
        idxs = np.arange(num_samples)

        for _ in range(ppo_cfg.ppo_epochs):
            np.random.shuffle(idxs)
            for start in range(0, num_samples, ppo_cfg.mini_batch_size):
                end = start + ppo_cfg.mini_batch_size
                batch_idx = idxs[start:end]

                b_obs = obs_tensor[batch_idx]
                b_act = act_tensor[batch_idx]
                b_logp_old = logp_old_tensor[batch_idx]
                b_adv = adv_tensor[batch_idx]
                b_ret = ret_tensor[batch_idx]

                logits, v_pred = ac(b_obs)
                dist = Categorical(logits=logits)
                logp = dist.log_prob(b_act)
                entropy = dist.entropy().mean()

                ratio = torch.exp(logp - b_logp_old)
                surr1 = ratio * b_adv
                surr2 = torch.clamp(
                    ratio, 1.0 - ppo_cfg.clip_range, 1.0 + ppo_cfg.clip_range
                ) * b_adv
                policy_loss = -torch.min(surr1, surr2).mean()

                value_loss = ((v_pred - b_ret) ** 2).mean()

                loss = (
                    policy_loss
                    + ppo_cfg.value_coef * value_loss
                    - ppo_cfg.entropy_coef * entropy
                )

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(ac.parameters(), ppo_cfg.max_grad_norm)
                optimizer.step()

        # ===== 日志 & 曲线数据 =====
        print(
            f"[UPD {upd:03d}] steps={T:4d}  "
            f"Return={ep_return:.3f}"
        )

        update_ids.append(upd)
        return_hist.append(ep_return)

        if writer is not None:
            writer.add_scalar("reward/return_per_update", ep_return, upd)

        # 定期保存 checkpoint
        if upd % 25 == 0:
            ckpt_path = out_dir / f"ac_multi_upd{upd:03d}.pt"
            torch.save(
                {
                    "model": ac.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "update": upd,
                    "global_step": global_step,
                },
                ckpt_path,
            )
            print(f" 💾 Saved checkpoint: {ckpt_path}")

    # ===== 训练结束：画 reward 曲线 =====
    if len(update_ids) > 0:
        plt.figure(figsize=(8, 5))
        plt.plot(update_ids, return_hist, label="Return per update")
        plt.xlabel("Update")
        plt.ylabel("Sum of rewards (horizon)")
        plt.title("Multi-env Blue PPO training (return per update)")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        png_path = out_dir / "reward_curve.png"
        plt.savefig(png_path, dpi=150)
        plt.close()
        print(f"📈 Saved reward curve: {png_path}")

    if writer is not None:
        writer.close()

    print("✅ Multi-env PPO training finished.")


if __name__ == "__main__":
    main()
