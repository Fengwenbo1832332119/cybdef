# scripts/agent/eval_ppo_cskg.py
# -*- coding: utf-8 -*-
"""
评估脚本：加载已经训练好的 PPO 策略，跑若干回合，统计表现

用法示例：
    cd C:\cybdef
    conda activate cyborg310
    python scripts/agent/eval_ppo_cskg.py ^
        --model scripts/runs/ppo_cskg/ppo_cskg_xxxx/ac_upd050.pt ^
        --episodes 20
"""

import os
import sys
import argparse
import pathlib
from collections import Counter

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


# ===== 与训练保持一致的 Actor-Critic 结构 =====
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


def to_obs_vector(obs_raw):
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


def evaluate(model_path: str, episodes: int = 20, max_steps: int = 100):
    # --- 初始化环境 ---
    env_yaml = ROOT / "scripts" / "configs" / "env.yaml"
    env = CybORGWrapper(str(env_yaml))

    obs_dim = env.obs_dim
    act_dim = env.action_dim
    action_names = env.action_space.names

    print(f"✅ 评估环境初始化完成: obs_dim={obs_dim}, act_dim={act_dim}")
    print(f"   使用模型: {model_path}")

    # --- 初始化并加载模型 ---
    ac = ActorCritic(obs_dim, act_dim).to(DEVICE)

    ckpt = torch.load(model_path, map_location=DEVICE)
    if isinstance(ckpt, dict) and "model" in ckpt:
        ac.load_state_dict(ckpt["model"])
        print(f"   🔄 从 checkpoint 字典中加载 'model' 权重")
    else:
        ac.load_state_dict(ckpt)
        print(f"   🔄 从纯 state_dict 中加载权重")

    ac.eval()

    # 统计
    all_rewards = []
    all_lengths = []
    action_counter = Counter()

    for ep in range(1, episodes + 1):
        obs_raw = env.reset()
        obs = to_obs_vector(obs_raw)

        done = False
        ep_reward = 0.0
        step = 0

        while not done and step < max_steps:
            step += 1
            obs_tensor = torch.from_numpy(obs).to(DEVICE).unsqueeze(0)

            with torch.no_grad():
                logits, _ = ac(obs_tensor)
                logits = logits.squeeze(0)
                dist = Categorical(logits=logits)
                action = dist.sample()

            action_idx = int(action.item())
            action_name = action_names[action_idx]
            action_counter[action_name] += 1

            next_obs_raw, reward_env, done, info = env.step(action_idx)

            ep_reward += float(reward_env)
            obs = to_obs_vector(next_obs_raw)

        all_rewards.append(ep_reward)
        all_lengths.append(step)

        print(f"[EVAL EP {ep:03d}] steps={step:3d}  R_env={ep_reward:.3f}")

    # --- 汇总统计 ---
    if len(all_rewards) > 0:
        mean_r = np.mean(all_rewards)
        std_r = np.std(all_rewards)
        mean_len = np.mean(all_lengths)
        print("\n===== 评估结果汇总 =====")
        print(f"  回合数       : {episodes}")
        print(f"  平均 EnvReward: {mean_r:.3f} ± {std_r:.3f}")
        print(f"  平均步长      : {mean_len:.1f}")

    print("\n===== 动作使用统计（Top 20） =====")
    for name, cnt in action_counter.most_common(20):
        print(f"  {name:<20s} : {cnt}")

    env.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="模型 checkpoint 路径，如 scripts/runs/ppo_cskg/.../ac_upd050.pt",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=20,
        help="评估回合数（默认 20）",
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=100,
        help="每回合最多步数（默认 100）",
    )

    args = parser.parse_args()

    evaluate(
        model_path=args.model,
        episodes=args.episodes,
        max_steps=args.max_steps,
    )


if __name__ == "__main__":
    main()
