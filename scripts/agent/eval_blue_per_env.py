# scripts/agent/eval_blue_per_env.py
# -*- coding: utf-8 -*-
"""
Evaluate a Blue PPO policy on each env separately (cyborg / ics / lot / robotics).

- 使用与训练一致的 MultiEnvWrapper 统一 obs_dim / act_dim
- 每个场景单独评估固定数量 episodes
- 适用于 baseline / weak-CSKG / 以后 GNN 版

Usage (PowerShell)：

    conda activate primaite311
    cd C:\cybdef

    python scripts/agent/eval_blue_per_env.py `
        --ckpt C:\cybdef\scripts\runs\multi_blue_cskg\multi_blue_cskg_xxx\ac_multi_upd400.pt `
        --episodes-per-env 40 `
        --max-steps 128
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# ==== 路径注入 ====
ROOT = Path(__file__).resolve().parents[2]  # C:\cybdef
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# 和训练保持一致：用 MultiEnvWrapper 拿到全局 obs_dim / act_dim
from scripts.envs.multi_env_wrapper import MultiEnvWrapper  # type: ignore
from scripts.envs.registry import (  # type: ignore
    make_cyborg,
    make_ics,
    make_lot,
    make_robotics,
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def to_obs_vector(obs_raw: Any) -> np.ndarray:
    """和 train_blue_multi_env.py 保持一致的 obs 处理逻辑。"""
    if isinstance(obs_raw, dict):
        if "obs_vec" in obs_raw:
            obs_raw = obs_raw["obs_vec"]
        else:
            for k in ["obs", "observation", "vector", "state"]:
                if k in obs_raw:
                    obs_raw = obs_raw[k]
                    break
    if isinstance(obs_raw, dict):
        raise TypeError(f"无法从 obs 字典中提取向量: keys={list(obs_raw.keys())}")
    return np.asarray(obs_raw, dtype=np.float32).reshape(-1)


class ActorCritic(nn.Module):
    """和 train_blue_multi_env.py 中保持完全一致的 MLP。"""

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


def eval_on_env(
    env_name: str,
    make_fn,
    ac: ActorCritic,
    global_obs_dim: int,
    global_act_dim: int,
    episodes: int,
    max_steps: int,
    stochastic: bool = False,
) -> Dict[str, Any]:
    """在单一场景上评估若干 episodes。"""

    returns: List[float] = []
    lengths: List[int] = []

    for ep in range(1, episodes + 1):
        env = make_fn()
        obs_raw = env.reset()
        obs_env = to_obs_vector(obs_raw)

        env_obs_dim = obs_env.shape[0]
        # 某些 env（CybORGWrapper, PrimaiteWrapper）一定有 action_space.n
        if not hasattr(env, "action_space") or not hasattr(env.action_space, "n"):
            raise RuntimeError(f"{env_name} env 没有 action_space.n，无法推断动作维度")
        env_act_dim = int(env.action_space.n)

        # pad 到 global_obs_dim
        obs_vec = np.zeros(global_obs_dim, dtype=np.float32)
        obs_vec[:env_obs_dim] = obs_env

        ep_ret = 0.0
        ep_len = 0

        for t in range(max_steps):
            ep_len += 1

            obs_t = torch.from_numpy(obs_vec).to(DEVICE).unsqueeze(0)
            with torch.no_grad():
                logits, _ = ac(obs_t)  # [1, global_act_dim]

                # 1) 先把 env 之外的动作通道屏蔽（multi-env 统一 act_dim 的 padding）
                if env_act_dim < global_act_dim:
                    logits = logits.clone()
                    logits[..., env_act_dim:] = -1e9

                # 2) 再根据「环境自己的合法动作」做一次更细粒度的 mask（可选但推荐）
                try:
                    local_mask = None

                    # 优先用 action_masks()（比如一些 PrimAITE 风格环境）
                    if hasattr(env, "action_masks"):
                        m = env.action_masks()
                        if m is not None:
                            local_mask = np.asarray(m, dtype=np.float32).reshape(-1)

                    # 退而求其次，用 wrapper 内部的 _current_legal_mask（如果有的话）
                    if local_mask is None and hasattr(env, "_current_legal_mask"):
                        m = env._current_legal_mask()
                        if m is not None:
                            local_mask = np.asarray(m, dtype=np.float32).reshape(-1)

                    # 如果拿到了合法 mask，并且长度刚好等于 env_act_dim，就拼成全局 mask
                    if local_mask is not None and local_mask.size == env_act_dim:
                        # 先默认所有 global 动作都不可用
                        global_mask = np.zeros(global_act_dim, dtype=np.float32)
                        # 前 env_act_dim 里，用 local_mask 指定哪些动作可用
                        global_mask[:env_act_dim] = local_mask

                        mask_t = torch.from_numpy(global_mask).to(DEVICE)
                        logits = logits.clone()
                        logits[mask_t == 0] = -1e9

                except Exception:
                    # eval 阶段出错就当没 mask，保证不崩
                    pass

                # 3) 按照最终 masked 之后的 logits 选动作
                if stochastic:
                    dist = Categorical(logits=logits)
                    action = dist.sample()
                else:
                    action = torch.argmax(logits, dim=-1)

                a_idx = int(action.item())

            next_obs_raw, reward, done, info = env.step(a_idx)
            r = float(reward)
            ep_ret += r

            # 下一个 obs 同样需要 pad
            obs_env_next = to_obs_vector(next_obs_raw)
            env_obs_dim_next = obs_env_next.shape[0]
            obs_vec = np.zeros(global_obs_dim, dtype=np.float32)
            obs_vec[:env_obs_dim_next] = obs_env_next

            if done:
                break

        if hasattr(env, "close"):
            env.close()

        returns.append(ep_ret)
        lengths.append(ep_len)

        print(f"[{env_name} | EP {ep:03d}] return={ep_ret:8.3f}  len={ep_len:3d}")

    arr_r = np.asarray(returns, dtype=np.float32)
    arr_l = np.asarray(lengths, dtype=np.float32)

    stats = {
        "env": env_name,
        "episodes": episodes,
        "return": {
            "mean": float(arr_r.mean()),
            "std": float(arr_r.std()),
            "min": float(arr_r.min()),
            "max": float(arr_r.max()),
        },
        "length": {
            "mean": float(arr_l.mean()),
            "std": float(arr_l.std()),
            "min": float(arr_l.min()),
            "max": float(arr_l.max()),
        },
        "per_episode": [
            {"index": i + 1, "return": float(r), "length": int(l)}
            for i, (r, l) in enumerate(zip(returns, lengths))
        ],
    }
    return stats


def main():
    parser = argparse.ArgumentParser(description="Per-env evaluation for Blue PPO policy")
    parser.add_argument("--ckpt", type=str, required=True, help="Path to ac_multi_updXXX.pt")
    parser.add_argument(
        "--episodes-per-env",
        type=int,
        default=40,
        help="Evaluation episodes per environment.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=128,
        help="Max steps per episode.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for numpy / torch.",
    )
    parser.add_argument(
        "--stochastic",
        action="store_true",
        help="Use sampling instead of greedy argmax.",
    )
    args = parser.parse_args()

    ckpt_path = Path(args.ckpt)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # === 关键：用 MultiEnvWrapper 拿到「训练时」的 obs_dim / act_dim ===
    tmp_multi = MultiEnvWrapper(
        env_names=["cyborg", "ics", "lot", "robotics"],
        weights=[0.7, 0.1, 0.1, 0.1],  # 和训练脚本保持一致就行，权重不影响维度
        mode="eval",
    )
    global_obs_dim = tmp_multi.obs_dim
    global_act_dim = tmp_multi.action_dim
    print(f"✅ Inferred global_obs_dim={global_obs_dim}, global_act_dim={global_act_dim}")
    if hasattr(tmp_multi, "close"):
        tmp_multi.close()

    # ==== 构建 ActorCritic 并加载 ckpt ====
    ac = ActorCritic(obs_dim=global_obs_dim, act_dim=global_act_dim, hidden=256).to(DEVICE)
    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    if "model" in ckpt:
        ac.load_state_dict(ckpt["model"])
    else:
        ac.load_state_dict(ckpt)
    ac.eval()
    print(f"🔍 Loaded checkpoint: {ckpt_path}")

    # ===== 逐场景评估 =====
    env_fns = {
        "cyborg": make_cyborg,
        # "ics": make_ics,
        # "lot": make_lot,
        # "robotics": make_robotics,
    }

    all_stats: Dict[str, Dict[str, Any]] = {}
    overall_returns: List[float] = []
    overall_env_labels: List[str] = []

    for name, fn in env_fns.items():
        print(f"\n===== Evaluating env: {name} =====")
        stats = eval_on_env(
            env_name=name,
            make_fn=fn,
            ac=ac,
            global_obs_dim=global_obs_dim,
            global_act_dim=global_act_dim,
            episodes=args.episodes_per_env,
            max_steps=args.max_steps,
            stochastic=args.stochastic,
        )
        all_stats[name] = stats
        for ep in stats["per_episode"]:
            overall_returns.append(ep["return"])
            overall_env_labels.append(name)

    # overall 聚合
    arr_overall = np.asarray(overall_returns, dtype=np.float32)
    overall = {
        "return": {
            "mean": float(arr_overall.mean()),
            "std": float(arr_overall.std()),
            "min": float(arr_overall.min()),
            "max": float(arr_overall.max()),
        },
        "episodes": int(arr_overall.shape[0]),
    }

    # 保存 JSON
    out_dir = ckpt_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    result = {
        "ckpt": str(ckpt_path),
        "episodes_per_env": args.episodes_per_env,
        "max_steps": args.max_steps,
        "stochastic": bool(args.stochastic),
        "overall": overall,
        "per_env": all_stats,
    }

    json_path = out_dir / f"eval_per_env_ep{args.episodes_per_env}_ms{args.max_steps}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"\n💾 Per-env eval JSON saved to: {json_path}")

    # 画一个把四个 env 混在一起的散点图
    fig, ax = plt.subplots(figsize=(9, 5))
    xs = np.arange(len(overall_returns)) + 1
    env_to_id = {name: i for i, name in enumerate(env_fns.keys())}
    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red"]

    for idx, (ret, env_name) in enumerate(zip(overall_returns, overall_env_labels), start=1):
        c = colors[env_to_id[env_name] % len(colors)]
        ax.scatter(idx, ret, c=c, s=10)

    ax.set_xlabel("Episode (concatenated over envs)")
    ax.set_ylabel("Return")
    ax.set_title("Blue PPO per-env evaluation (scatter)")
    ax.grid(True, linestyle="--", alpha=0.4)

    handles = [
        plt.Line2D(
            [0], [0], marker="o", linestyle="None", markersize=5, color=colors[i],
            label=name
        )
        for i, name in enumerate(env_fns.keys())
    ]
    ax.legend(handles=handles, title="Env")

    png_path = out_dir / f"eval_per_env_ep{args.episodes_per_env}_ms{args.max_steps}.png"
    fig.tight_layout()
    fig.savefig(png_path, dpi=150)
    plt.close(fig)
    print(f"🖼  Per-env eval plot saved to: {png_path}")

    print("✅ Per-env eval finished.")


if __name__ == "__main__":
    main()
