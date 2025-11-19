# -*- coding: utf-8 -*-
"""
debug_env_hosts.py

用途：
1）从 scripts/configs/env.yaml 中解析出 scenario_file 和 seed；
2）用 BlueTableWrapper 打印 Blue 视角的原始 observation 表格（看有哪些 Host、在哪个子网）；
3）用 CybORGWrapper 打印 PPO 实际使用的 host 顺序（如果实现了 env.host_order）
   以及前若干个动作名，方便我们后面构 seed_graph.json 和 host-aware 规则。

在 C:\cybdef 目录下运行：
    python scripts/agent/debug_env_hosts.py
"""

import pprint
import pathlib
import sys

# ===== 路径注入 =====
ROOT = pathlib.Path(__file__).resolve().parents[2]  # C:\cybdef
THIRD = ROOT / "third_party" / "CybORG"

for p in (ROOT, THIRD):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

# ===== 项目内 import =====
try:
    from scripts.envs.cyborg_wrapper import CybORGWrapper
except ImportError:
    from scripts.envs.cyborg_wrapper import CybORGWrapper

from CybORG import CybORG
from CybORG.Simulator.Scenarios import FileReaderScenarioGenerator
from CybORG.Agents import B_lineAgent, BlueReactRemoveAgent
from CybORG.Agents.Wrappers import BlueTableWrapper


def main():
    # ---------- 1. 解析 env.yaml ----------
    # 优先用 scripts/configs/env.yaml，找不到再用 configs/env.yaml
    env_yaml_candidates = [
        ROOT / "scripts" / "configs" / "env.yaml",
        ROOT / "configs" / "env.yaml",
        pathlib.Path("scripts/configs/env.yaml"),
        pathlib.Path("configs/env.yaml"),
    ]
    ENV_YAML = None
    for c in env_yaml_candidates:
        if c.exists():
            ENV_YAML = c
            break
    if ENV_YAML is None:
        raise FileNotFoundError("没有找到 env.yaml，请确认路径（scripts/configs/env.yaml）")

    print("✅ 使用的 env.yaml:", ENV_YAML)

    # 用你自己的 CybORGWrapper 先读一遍配置（里面已经做好路径解析）
    tmp_env = CybORGWrapper(str(ENV_YAML))
    env_cfg = tmp_env.config["environment"]
    scenario_file = env_cfg["scenario_file"]
    seed = env_cfg.get("seed", 42)

    print("\n=== 从 env.yaml 解析到的环境配置 ===")
    print("scenario_file:", scenario_file)
    print("seed         :", seed)

    # 如果你的 CybORGWrapper 里有 host_order，就一起打印出来
    if hasattr(tmp_env, "host_order"):
        print("\n=== CybORGWrapper.host_order （PPO 真实使用的主机顺序）===")
        print(list(tmp_env.host_order))
    else:
        print("\n⚠ CybORGWrapper 上没有 host_order 属性，如果之后需要，我们可以在包装器里加上。")

    # 打印一下动作空间前若干个动作名，看看 Remove/Restore/Decoy/Monitor 的真实命名
    try:
        print("\n=== 动作空间前 40 个动作名（env.action_space.names）===")
        for i, name in enumerate(tmp_env.action_space.names[:40]):
            print(f"[{i:02d}] {name}")
    except Exception as e:
        print("\n⚠ 无法打印动作名:", e)

    # 先关掉临时 env，避免资源占用
    tmp_env.close()

    # ---------- 2. 用 BlueTableWrapper 看“表格版”观测 ----------
    red_agent = B_lineAgent()
    blue_agent = BlueReactRemoveAgent()  # 随便用一个 Blue agent 就行

    scenario_gen = FileReaderScenarioGenerator(scenario_file)
    base_env = CybORG(
        scenario_generator=scenario_gen,
        environment="sim",
        agents={"Red": red_agent, "Blue": blue_agent},
        seed=seed,
    )

    # 不设 output_mode='vector'，拿表格版输出
    table_env = BlueTableWrapper(base_env)
    result = table_env.reset(agent="Blue")
    obs = result.observation

    print("\n=== BlueTableWrapper.reset() 原始 observation ===")
    print("类型:", type(obs))

    # 兼容几种常见结构：DataFrame / dict / 其他
    # 1) pandas.DataFrame（最常见）
    try:
        import pandas as pd  # 只是为了 isinstance 判断
        if isinstance(obs, pd.DataFrame):
            print("DataFrame 形状:", obs.shape)
            print("列名:", list(obs.columns))
            print("\n前几行：")
            print(obs.head(20))   # 多打一点行，方便你看 host 列表
        else:
            raise TypeError
    except Exception:
        # 2) dict
        if isinstance(obs, dict):
            print("字典 keys:", list(obs.keys()))
            print("\n详细结构预览（只打印前几项）：")
            pprint.pprint(obs)
        else:
            # 3) 其他类型：直接 pprint
            print("内容预览:")
            pprint.pprint(obs)

    # 最后关闭环境
    try:
        table_env.close()
    except Exception:
        pass

    print("\n✅ debug_env_hosts.py 运行结束，麻烦把上面的输出发给我，我们就可以对齐 seed_graph.json 了。")


if __name__ == "__main__":
    main()
