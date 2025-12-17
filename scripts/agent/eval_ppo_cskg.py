# scripts/agent/eval_ppo_cskg.py
# -*- coding: utf-8 -*-
"""
评估脚本：加载已经训练好的 PPO-CSKG 策略，在当前 CybORGWrapper 上跑若干回合，统计表现。

支持：
- 从 checkpoint 中自动推断 obs_dim / act_dim（兼容旧的 178 维动作头）
- 多个 max_steps（例如 30 / 50 / 100）依次评估
- 动作维度不一致时，将老策略的动作映射到当前环境动作空间：
    * 0 <= idx < env_act_dim: 直接使用
    * idx >= env_act_dim: 映射为 0 (Sleep)

用法示例（PowerShell）：
    cd C:\cybdef
    conda activate cyborg310

    python scripts/agent/eval_ppo_cskg.py `
      --model scripts/runs/ppo_cskg/ppo_cskg_b1_1763597492/ac_upd250.pt `
      --episodes 20 `
      --num-steps 30 50 100
"""

import sys
import argparse
import pathlib
from collections import Counter
from typing import Any, Dict, List, Optional

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
try:
    from envs.cyborg_wrapper import CybORGWrapper
except ImportError:
    from scripts.envs.cyborg_wrapper import CybORGWrapper

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ===== 与当年训练 PPO-CSKG 一致的 Actor-Critic 结构（hidden=128） =====
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

    def forward(self, obs: torch.Tensor):
        logits = self.actor(obs)
        value = self.critic(obs)
        return logits, value.squeeze(-1)


# ===== 工具函数 =====
def to_obs_vector(obs_raw: Any) -> np.ndarray:
    """
    和 train_ppo_cskg_old.py 保持一致：
    支持 dict 包装结构（包含 obs_vec / obs / observation / state 等）
    """
    if isinstance(obs_raw, dict):
        if "obs_vec" in obs_raw:
            obs_raw = obs_raw["obs_vec"]
        else:
            for key in ["obs", "observation", "vector", "state"]:
                if key in obs_raw:
                    obs_raw = obs_raw[key]
                    break

    if isinstance(obs_raw, dict):
        raise TypeError(
            f"无法从 obs 字典中提取向量，请检查 keys: {list(obs_raw.keys())}"
        )

    obs_np = np.array(obs_raw, dtype=np.float32).reshape(-1)
    return obs_np


def _infer_dims_from_ckpt_state(state_dict: Dict[str, torch.Tensor]) -> (int, int):
    """
    从 ckpt 的 state_dict 里推断 obs_dim / act_dim：
    - actor.0.weight: [hidden, obs_dim]
    - actor.4.weight: [act_dim, hidden]
    """
    # 保险起见同时兼容 module 前缀
    keys = list(state_dict.keys())
    # 找第一个包含 ".0.weight" 的 actor 层
    actor0_key = None
    actor4_key = None
    for k in keys:
        if "actor.0.weight" in k:
            actor0_key = k
        if "actor.4.weight" in k:
            actor4_key = k
    if actor0_key is None or actor4_key is None:
        raise RuntimeError(
            "无法在 checkpoint state_dict 中找到 actor.0.weight / actor.4.weight，"
            "请检查网络结构是否与 ActorCritic 一致。"
        )

    obs_dim = state_dict[actor0_key].shape[1]
    act_dim = state_dict[actor4_key].shape[0]
    return int(obs_dim), int(act_dim)


def _get_action_mask(
    env: Any,
    obs_raw: Any,
    env_act_dim: int,
    net_act_dim: int,
) -> np.ndarray:
    """
    从 env / obs_raw 中尽量拿到合法动作 mask，并扩展/截断到 net_act_dim。

    返回值：float32 数组，shape=(net_act_dim,)，1=合法，0=非法。
    """
    mask_env: Optional[np.ndarray] = None

    # 1) 优先用 CybORGWrapper 内部的当前合法动作
    if hasattr(env, "_current_legal_mask"):
        try:
            m = env._current_legal_mask()
            mask_env = np.asarray(m, dtype=np.float32).reshape(-1)
        except Exception:
            mask_env = None

    # 2) 再尝试 obs_raw 里的 "legal_mask"
    if mask_env is None and isinstance(obs_raw, dict) and "legal_mask" in obs_raw:
        try:
            m = obs_raw["legal_mask"]
            mask_env = np.asarray(m, dtype=np.float32).reshape(-1)
        except Exception:
            mask_env = None

    # 3) 如果还是拿不到，默认前 env_act_dim 个动作合法
    if mask_env is None or mask_env.size != env_act_dim:
        mask_env = np.ones(env_act_dim, dtype=np.float32)

    # === 映射到 net_act_dim ===
    if net_act_dim == env_act_dim:
        mask = mask_env
    elif net_act_dim > env_act_dim:
        # 旧策略头更大（例如 178 > 145）：多出来的动作全部设为 0（非法）
        mask = np.zeros(net_act_dim, dtype=np.float32)
        mask[:env_act_dim] = mask_env[:env_act_dim]
    else:
        # 理论上用不到：策略头比当前 env 小
        mask = mask_env[:net_act_dim]

    # 保证至少有一个合法动作
    if mask.sum() <= 0:
        mask[0] = 1.0
    return mask


# ===== 主评估逻辑 =====
def evaluate_for_max_steps(
    model_path: str,
    episodes: int = 20,
    max_steps: int = 100,
):
    # --- 初始化环境 ---
    env_yaml = ROOT / "scripts" / "configs" / "env.yaml"
    env = CybORGWrapper(str(env_yaml))

    env_obs_dim = env.obs_dim
    env_act_dim = env.action_dim
    action_names = env.action_space.names

    print("======================================================================")
    print(f"▶ 开始评估：max_steps = {max_steps}")
    print("======================================================================")
    print(
        f"✅ 评估环境初始化完成: env_obs_dim={env_obs_dim}, env_act_dim={env_act_dim}"
    )
    print(f"   使用模型: {model_path}")

    # --- 先加载 ckpt，推断 ckpt 里的 obs_dim / act_dim ---
    ckpt = torch.load(model_path, map_location=DEVICE)
    if isinstance(ckpt, dict) and "model" in ckpt:
        state = ckpt["model"]
        print("   🔄 从 checkpoint 字典中加载 'model' 权重")
    else:
        state = ckpt
        print("   🔄 从纯 state_dict 中加载权重")

    if not isinstance(state, dict):
        raise RuntimeError("checkpoint 格式异常：既不是 dict(model=...) 也不是 state_dict")

    ckpt_obs_dim, ckpt_act_dim = _infer_dims_from_ckpt_state(state)
    print(
        f"   📐 ckpt_dims: obs_dim={ckpt_obs_dim}, act_dim={ckpt_act_dim} "
        f"(env_act_dim={env_act_dim})"
    )
    if ckpt_obs_dim != env_obs_dim:
        print(
            f"   ⚠ 警告：ckpt_obs_dim({ckpt_obs_dim}) != env_obs_dim({env_obs_dim})，"
            "仍然尝试继续评估，但结果可能不太可靠。"
        )

    # --- 初始化并加载模型（按 ckpt 的维度构网络） ---
    ac = ActorCritic(obs_dim=ckpt_obs_dim, act_dim=ckpt_act_dim).to(DEVICE)
    ac.load_state_dict(state)
    ac.eval()

    # 统计
    all_rewards: List[float] = []
    all_lengths: List[int] = []
    action_counter = Counter()

    for ep in range(1, episodes + 1):
        obs_raw = env.reset()
        obs_vec = to_obs_vector(obs_raw)

        done = False
        ep_reward = 0.0
        step = 0

        while not done and step < max_steps:
            step += 1
            obs_pad = obs_vec

            # 如果 ckpt_obs_dim 比当前 obs 长/短，做一下 pad/crop
            obs_pad = np.asarray(obs_pad, dtype=np.float32).reshape(-1)
            if obs_pad.size < ckpt_obs_dim:
                tmp = np.zeros(ckpt_obs_dim, dtype=np.float32)
                tmp[: obs_pad.size] = obs_pad
                obs_pad = tmp
            elif obs_pad.size > ckpt_obs_dim:
                obs_pad = obs_pad[:ckpt_obs_dim]

            obs_tensor = torch.from_numpy(obs_pad).to(DEVICE).unsqueeze(0)

            with torch.no_grad():
                logits, _ = ac(obs_tensor)
                logits = logits.squeeze(0)

                # === 合法动作 mask ===
                mask_np = _get_action_mask(
                    env=env,
                    obs_raw=obs_raw,
                    env_act_dim=env_act_dim,
                    net_act_dim=ckpt_act_dim,
                )
                mask_t = torch.from_numpy(mask_np.astype(bool)).to(DEVICE)
                logits = logits.clone()
                logits[~mask_t] = -1e9

                # 这里仍然用采样（保持和训练/旧评估一致）；如果想改为 greedy，可以用 argmax
                dist = Categorical(logits=logits)
                action_net = dist.sample()

            action_idx_net = int(action_net.item())
            # 将 178 维的动作 idx 映射回当前 145 维动作空间
            if 0 <= action_idx_net < env_act_dim:
                action_idx_env = action_idx_net
            else:
                action_idx_env = 0  # 越界动作强制当成 Sleep

            action_name = action_names[action_idx_env]
            action_counter[action_name] += 1

            next_obs_raw, reward_env, done, info = env.step(action_idx_env)

            ep_reward += float(reward_env)
            obs_raw = next_obs_raw
            obs_vec = to_obs_vector(next_obs_raw)

        all_rewards.append(ep_reward)
        all_lengths.append(step)

        print(f"[EVAL EP {ep:03d}] steps={step:3d}  R_env={ep_reward:.3f}")

    # --- 汇总统计 ---
    if len(all_rewards) > 0:
        mean_r = float(np.mean(all_rewards))
        std_r = float(np.std(all_rewards))
        mean_len = float(np.mean(all_lengths))
        print("\n===== 评估结果汇总 =====")
        print(f"  回合数         : {episodes}")
        print(f"  平均 EnvReward : {mean_r:.3f} ± {std_r:.3f}")
        print(f"  平均步长       : {mean_len:.1f}")

    print("\n===== 动作使用统计（Top 20） =====")
    for name, cnt in action_counter.most_common(20):
        print(f"  {name:<25s} : {cnt}")

    env.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="模型 checkpoint 路径，如 scripts/runs/ppo_cskg/.../ac_upd250.pt",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=20,
        help="每个 max_steps 下的评估回合数（默认 20）",
    )
    parser.add_argument(
        "--num-steps",
        type=int,
        nargs="+",
        default=[100],
        help="每回合最多步数列表，例如: --num-steps 30 50 100",
    )

    args = parser.parse_args()

    for max_steps in args.num_steps:
        evaluate_for_max_steps(
            model_path=args.model,
            episodes=args.episodes,
            max_steps=max_steps,
        )


if __name__ == "__main__":
    main()
