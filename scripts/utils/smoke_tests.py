# scripts/utils/smoke_tests.py
# -*- coding: utf-8 -*-
"""轻量级 smoke test 脚本，便于快速验证动作映射与奖励归一化。

用法示例：

```bash
# 单场景快速检查：ICS/LoT/Robotics 各跑 1 集，打印 raw/normalized reward
python -m scripts.utils.smoke_tests --single-envs ics lot robotics --steps 30 --episodes 1

# 多场景采样：重置并执行 5 步，查看 env_id / reward / normalized_reward / mapped_action_name
MULTIENV_REWARD_NORM=per_env_max python -m scripts.utils.smoke_tests --multi-samples 5
```
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Iterable

import numpy as np


# 确保无论从哪个工作目录运行，都能找到仓库内的 scripts 包。
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from scripts.envs import ENV_REGISTRY
    from scripts.envs.multi_env_wrapper import MultiEnvWrapper
except ModuleNotFoundError as exc:  # pragma: no cover - 仅在环境未安装依赖时触发
    missing = exc.name or "unknown"
    print(
        "[ERROR] 依赖缺失，无法导入模块：",
        missing,
        "\n请确认：\n"
        "  1) 命令在仓库根目录执行，或已将仓库路径加入 PYTHONPATH。\n"
        "  2) 已安装第三方依赖，例如 `pip install -e third_party/PrimAITE` 或安装 CybORG。\n"
        "  3) 若提示 scripts.* 找不到，可在命令前添加 `PYTHONPATH=.` 再试。",
    )
    sys.exit(1)


def _sample_action(env) -> int:
    """根据合法掩码随机采样一个动作索引。"""

    mask = None
    for attr in ("action_masks", "_current_legal_mask"):
        if hasattr(env, attr):
            try:
                candidate = getattr(env, attr)
                candidate = candidate() if callable(candidate) else candidate
                mask = np.asarray(candidate, dtype=np.float32).reshape(-1)
                break
            except Exception:
                continue

    if mask is not None and mask.shape[0] > 0:
        legal = np.flatnonzero(mask > 0)
        if legal.size > 0:
            return int(np.random.choice(legal))

    act_dim = getattr(env, "action_dim", None)
    if act_dim is None:
        space = getattr(env, "action_space", None)
        act_dim = int(getattr(space, "n", 1)) if space is not None else 1
    return int(np.random.randint(0, max(int(act_dim), 1)))


def run_single_env_smoke(env_names: Iterable[str], steps: int, episodes: int) -> None:
    """逐个场景跑小规模 episode，打印 raw 与归一化奖励。"""

    for env_name in env_names:
        if env_name not in ENV_REGISTRY:
            print(f"[SKIP] 未注册场景: {env_name}")
            continue

        print(f"\n===== {env_name} smoke test =====")
        make_env = ENV_REGISTRY[env_name]
        for ep in range(episodes):
            env = make_env()
            obs = env.reset()
            if isinstance(obs, tuple) and len(obs) == 2:
                obs, _ = obs

            rs = getattr(env, "reward_scale", None)
            raw_total = 0.0
            norm_total = 0.0
            for t in range(steps):
                a = _sample_action(env)
                res = env.step(a)
                if len(res) == 4:
                    obs, reward, done, info = res
                else:
                    obs, reward, terminated, truncated, info = res
                    done = bool(terminated or truncated)

                raw_total += float(reward)
                norm = None
                if rs:
                    norm = float(reward) / max(float(rs), 1e-6)
                    norm_total += norm

                mapped = info.get("mapped_action_name") if isinstance(info, dict) else None
                intent = info.get("intent_name") if isinstance(info, dict) else None
                print(
                    f"[ep {ep} step {t}] raw={reward:.3f}"
                    + (f" norm={norm:.3f}" if norm is not None else "")
                    + (f" intent={intent}" if intent else "")
                    + (f" action={mapped}" if mapped else "")
                )

                if done:
                    break

            print(
                f"--> episode {ep} done: raw_total={raw_total:.3f}"
                + (f" norm_total={norm_total:.3f}" if rs else "")
            )
            if hasattr(env, "close"):
                try:
                    env.close()
                except Exception:
                    pass


def run_multi_env_samples(num_samples: int) -> None:
    """在 MultiEnvWrapper 上打印多场景采样的奖励与动作映射。"""

    os.environ.setdefault("MULTIENV_REWARD_NORM", "per_env_max")
    env = MultiEnvWrapper()
    obs = env.reset()
    print(f"[multi] init env={obs.get('env_name') if isinstance(obs, dict) else 'unknown'}")

    for idx in range(num_samples):
        mask = env.current_action_mask()
        if mask is not None and mask.size > 0:
            legal = np.flatnonzero(mask > 0)
            action = int(np.random.choice(legal))
        else:
            action = int(np.random.randint(0, env.action_dim))

        obs, reward, done, info = env.step(action)
        env_id = info.get("env_name") if isinstance(info, dict) else "unknown"
        norm = info.get("normalized_reward") if isinstance(info, dict) else None
        mapped = info.get("mapped_action_name") if isinstance(info, dict) else None
        intent = info.get("intent_name") if isinstance(info, dict) else None
        print(
            f"[multi step {idx}] env={env_id} raw={reward:.3f}"
            + (f" norm={norm:.3f}" if norm is not None else "")
            + (f" intent={intent}" if intent else "")
            + (f" action={mapped}" if mapped else "")
        )

        if done:
            obs = env.reset()
            print(f"[multi] reset -> env={obs.get('env_name') if isinstance(obs, dict) else 'unknown'}")

    env.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke tests for multi-env training")
    parser.add_argument(
        "--single-envs",
        nargs="*",
        default=None,
        help="要跑单场景 smoke 的环境列表，例如: ics lot robotics",
    )
    parser.add_argument("--steps", type=int, default=20, help="每个 episode 步数上限")
    parser.add_argument("--episodes", type=int, default=1, help="每个场景跑的 episode 数")
    parser.add_argument(
        "--multi-samples",
        type=int,
        default=0,
        help="多场景采样步数（>0 则运行多场景采样检查）",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if args.single_envs:
        run_single_env_smoke(args.single_envs, steps=args.steps, episodes=args.episodes)

    if args.multi_samples and args.multi_samples > 0:
        run_multi_env_samples(args.multi_samples)

    if not args.single_envs and (not args.multi_samples or args.multi_samples <= 0):
        print("没有指定任务，使用 --single-envs 或 --multi-samples 运行 smoke test。")


if __name__ == "__main__":
    main()