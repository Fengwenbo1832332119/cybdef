"""Utility CLI for running PrimAITE scenarios individually or in batches."""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parents[1]
PRIMAITE_SRC = ROOT / "third_party" / "PrimAITE" / "src"
if str(PRIMAITE_SRC) not in sys.path:
    sys.path.insert(0, str(PRIMAITE_SRC))

from primaite.session.environment import PrimaiteGymEnv  # noqa: E402

CONFIG_BASE = PRIMAITE_SRC / "primaite" / "config" / "_package_data"
SCENARIO_PATHS: Dict[str, Path] = {
    "data_manipulation": CONFIG_BASE / "data_manipulation.yaml",
    "data_manipulation_marl": CONFIG_BASE / "data_manipulation_marl.yaml",
    "uc7": CONFIG_BASE / "uc7_config.yaml",
    "lot": CONFIG_BASE / "lot.yaml",
    "ics": CONFIG_BASE / "ics.yaml",
    "robotics": CONFIG_BASE / "robotics.yaml",
}


def summarise_history(actions: List) -> Dict[str, float]:
    """Build quick metrics from an agent history list."""
    detection_step = None
    response_step = None
    block_attempts = 0
    block_success = 0

    for idx, item in enumerate(actions):
        if detection_step is None and item.action != "do-nothing":
            detection_step = idx
        if item.action in {
            "router-acl-add-rule",
            "router-acl-remove-rule",
            "firewall-acl-add-rule",
            "node-shutdown",
        }:
            block_attempts += 1
            if getattr(item.response, "status", "") == "success":
                block_success += 1
            if response_step is None:
                response_step = idx

    return {
        "total_actions": len(actions),
        "detection_time": detection_step if detection_step is not None else -1,
        "response_time": response_step if response_step is not None else -1,
        "block_success_rate": block_success / block_attempts if block_attempts else 0.0,
        "false_positive_rate": (
            (block_attempts - block_success) / block_attempts if block_attempts else 0.0
        ),
    }


def run_scenario(name: str, config_path: Path, episodes: int, max_steps: int, report_dir: Path) -> Dict:
    """Run a scenario for a fixed number of episodes and return metrics."""
    env = PrimaiteGymEnv(config_path)
    episode_results = []

    for episode_idx in range(episodes):
        obs, _ = env.reset()
        terminated = False
        truncated = False
        steps = 0

        while not (terminated or truncated) and steps < max_steps:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            steps += 1

        history_metrics = summarise_history(env.agent.history)
        episode_results.append(
            {
                "episode": episode_idx,
                "steps": steps,
                "total_reward": env.agent.reward_function.total_reward,
                "last_reward": env.agent.reward_function.current_reward,
                **history_metrics,
            }
        )

    aggregated = {
        "scenario": name,
        "config": str(config_path),
        "episodes": episodes,
        "max_steps": max_steps,
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "episodes_detail": episode_results,
    }

    report_dir.mkdir(parents=True, exist_ok=True)
    with open(report_dir / f"{name}_summary.json", "w") as f:
        json.dump(aggregated, f, indent=2)

    return aggregated


def main():
    parser = argparse.ArgumentParser(description="Run PrimAITE scenarios with a simple random policy")
    parser.add_argument(
        "--scenario",
        "-s",
        action="append",
        choices=sorted(SCENARIO_PATHS.keys()),
        help="Scenario key to run. Can be supplied multiple times.",
    )
    parser.add_argument(
        "--episodes", "-e", type=int, default=1, help="Number of episodes to execute per scenario"
    )
    parser.add_argument(
        "--max-steps", "-m", type=int, default=32, help="Maximum steps per episode before stopping"
    )
    parser.add_argument(
        "--report-dir",
        "-o",
        type=Path,
        default=ROOT / "reports" / "primaite_runs",
        help="Directory for JSON summaries",
    )
    parser.add_argument("--list", action="store_true", help="List available scenarios and exit")

    args = parser.parse_args()

    if args.list:
        for key, path in SCENARIO_PATHS.items():
            print(f"{key}: {path}")
        return

    scenarios = args.scenario if args.scenario else list(SCENARIO_PATHS.keys())

    all_results = []
    for scenario_name in scenarios:
        config_path = SCENARIO_PATHS[scenario_name]
        if not config_path.exists():
            raise FileNotFoundError(f"Config not found for scenario {scenario_name}: {config_path}")
        print(f"Running scenario '{scenario_name}' from {config_path}...")
        aggregated = run_scenario(
            name=scenario_name,
            config_path=config_path,
            episodes=args.episodes,
            max_steps=args.max_steps,
            report_dir=args.report_dir,
        )
        print(
            f"Completed {scenario_name}: episodes={args.episodes}, "
            f"avg_steps={sum(r['steps'] for r in aggregated['episodes_detail'])/len(aggregated['episodes_detail'])}"  # noqa: E501
        )
        all_results.append(aggregated)

    summary_path = args.report_dir / "batch_summary.json"
    args.report_dir.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"Saved batch summary to {summary_path}")


if __name__ == "__main__":
    main()