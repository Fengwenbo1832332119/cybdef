"""
B0：纯 PPO 训练入口，复用 train_ppo_cskg.py 主循环
"""
from pathlib import Path
from scripts.agent.train_ppo_cskg  import main


def cli():
    config_path = Path(__file__).resolve().parents[1] / "configs" / "b0.yaml"
    main(str(config_path))


if __name__ == "__main__":
    cli()