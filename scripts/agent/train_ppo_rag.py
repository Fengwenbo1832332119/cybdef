"""
B2：PPO + CSKG + RAG 解释 训练入口
"""
from pathlib import Path
from scripts.agent.train_ppo_cskg  import main


def cli():
    config_path = Path(__file__).resolve().parents[1] / "configs" / "b2.yaml"
    main(str(config_path))


if __name__ == "__main__":
    cli()