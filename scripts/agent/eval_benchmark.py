"""
统一评测脚本：同一场景跑若干回合，重复 3 次，输出均值±方差
指标：累计奖励、收敛步数、拦截率、误杀率、MTTR
"""
import argparse
import json
import pathlib
import random
import sys
from typing import Any, Dict, List

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical

# ===== 路径注入 =====
ROOT = pathlib.Path(__file__).resolve().parents[2]
THIRD = ROOT / "third_party" / "CybORG"
for p in (ROOT, THIRD):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

try:
    from envs.cyborg_wrapper import CybORGWrapper
except ImportError:
    from scripts.envs.cyborg_wrapper import CybORGWrapper


def load_yaml(path: pathlib.Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    import yaml

    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


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
    if isinstance(obs_raw, dict):
        if "obs_vec" in obs_raw:
            obs_raw = obs_raw["obs_vec"]
        else:
            for key in ["obs", "observation", "vector", "state"]:
                if key in obs_raw:
                    obs_raw = obs_raw[key]
                    break
    if isinstance(obs_raw, dict):
        raise TypeError(f"无法从 obs 字典中提取向量，keys={list(obs_raw.keys())}")
    return np.array(obs_raw, dtype=np.float32).reshape(-1)


def evaluate_once(
    model_path: pathlib.Path,
    config_path: pathlib.Path,
    episodes: int,
    num_steps: int,
    seed_offset: int = 0,
) -> Dict[str, float]:
    exp_cfg = load_yaml(config_path)
    paths = exp_cfg.get("paths", {})
    env_yaml = ROOT / paths.get("env_config", "scripts/configs/env.yaml")

    seed = int(exp_cfg.get("experiment", {}).get("seed", 42)) + seed_offset
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    env = CybORGWrapper(str(env_yaml))
    obs_dim, act_dim = env.obs_dim, env.action_dim
    action_names = env.action_space.names

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ac = ActorCritic(obs_dim, act_dim).to(device)

    ckpt = torch.load(model_path, map_location=device)
    ac.load_state_dict(ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt)
    ac.eval()

    ep_rewards: List[float] = []
    ep_lengths: List[int] = []
    intercept_flags: List[bool] = []
    false_positive_actions = 0
    checked_actions = 0
    mttr_samples: List[int] = []

    for ep in range(episodes):
        obs_raw = env.reset()
        obs = to_obs_vector(obs_raw)
        facts = obs_raw.get("facts", {}) if isinstance(obs_raw, dict) else {}

        done = False
        step = 0
        ep_reward = 0.0
        compromised_timer = None

        while not done and step < num_steps:
            step += 1
            obs_tensor = torch.from_numpy(obs).to(device).unsqueeze(0)
            with torch.no_grad():
                logits, _ = ac(obs_tensor)
                action = Categorical(logits=logits.squeeze(0)).sample()

            action_idx = int(action.item())
            action_name = action_names[action_idx]

            if action_name.startswith("Remove") or action_name.startswith("Restore"):
                checked_actions += 1
                if not facts.get("host_compromised", False) and not facts.get("suspicious_activity", False):
                    false_positive_actions += 1

            next_obs_raw, reward_env, done, info = env.step(action_idx)
            ep_reward += float(reward_env)

            next_facts = next_obs_raw.get("facts", {}) if isinstance(next_obs_raw, dict) else {}
            if next_facts.get("host_compromised", False) and compromised_timer is None:
                compromised_timer = step
            if compromised_timer is not None and not next_facts.get("host_compromised", False):
                mttr_samples.append(step - compromised_timer)
                compromised_timer = None

            obs = to_obs_vector(next_obs_raw)
            facts = next_facts

        ep_rewards.append(ep_reward)
        ep_lengths.append(step)
        intercept_flags.append(not facts.get("critical_host_breached", False) and not facts.get("high_risk_state", False))

    result = {
        "reward_mean": float(np.mean(ep_rewards)),
        "reward_var": float(np.var(ep_rewards)),
        "length_mean": float(np.mean(ep_lengths)),
        "length_var": float(np.var(ep_lengths)),
        "intercept_rate": float(np.mean(intercept_flags)),
        "false_positive_rate": float(false_positive_actions / max(checked_actions, 1)),
        "mttr": float(np.mean(mttr_samples)) if len(mttr_samples) > 0 else 0.0,
    }
    return result


def aggregate_runs(run_metrics: List[Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    summary: Dict[str, Dict[str, float]] = {}
    for key in run_metrics[0].keys():
        arr = np.array([m[key] for m in run_metrics], dtype=np.float32)
        summary[key] = {"mean": float(np.mean(arr)), "var": float(np.var(arr))}
    return summary


def main():
    parser = argparse.ArgumentParser(description="统一评测：同场景重复 3 次")
    parser.add_argument("--model", required=True, help="待评估的策略权重")
    parser.add_argument(
        "--config",
        type=str,
        default=str(ROOT / "scripts" / "configs" / "b1.yaml"),
        help="实验配置文件（B0/B1/B2）",
    )
    parser.add_argument("--episodes", type=int, default=5, help="每次评估的回合数")
    parser.add_argument("--repeats", type=int, default=3, help="重复次数")
    parser.add_argument(
        "--num-steps",
        type=int,
        nargs="+",
        default=[30, 50, 100],
        help="单次评估的最大步数列表",
    )

    args = parser.parse_args()

    config_path = pathlib.Path(args.config)
    model_path = pathlib.Path(args.model)

    all_results: Dict[str, Dict[str, Any]] = {}
    for num_steps in args.num_steps:
        print(f"\n===== num_steps = {num_steps} =====")
        run_metrics: List[Dict[str, float]] = []
        for r in range(args.repeats):
            metrics = evaluate_once(
                model_path,
                config_path,
                episodes=args.episodes,
                num_steps=num_steps,
                seed_offset=r * 7,
            )
            print(f"[Run {r+1}/{args.repeats}] {metrics}")
            run_metrics.append(metrics)

        summary = aggregate_runs(run_metrics)
        print("-- 汇总（mean ± var） --")
        for k, v in summary.items():
            print(f"{k:20s}: {v['mean']:.4f} ± {v['var']:.4f}")

        all_results[str(num_steps)] = {"runs": run_metrics, "summary": summary}

    logs_dir = ROOT / "scripts" / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_path = logs_dir / f"eval_{config_path.stem}_{int(torch.randint(0, 1_000_000, (1,)).item())}.json"
    payload = {
        "config": str(config_path),
        "model": str(model_path),
        "episodes": args.episodes,
        "repeats": args.repeats,
        "num_steps": args.num_steps,
        "results": all_results,
    }
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"日志写入: {log_path}")


if __name__ == "__main__":
    main()