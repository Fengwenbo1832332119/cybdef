# C:\cybdef\scripts\cyborg_cc2_logger_plus.py
import sys, re, json, argparse, inspect
from pathlib import Path
from datetime import datetime, timezone
from collections import Counter

# ========= 让 third_party/CybORG 成为可 import 的包 =========
PROJ_ROOT   = Path(__file__).resolve().parents[1]               # C:\cybdef
CYBORG_ROOT = PROJ_ROOT / "third_party" / "CybORG"
if str(CYBORG_ROOT) not in sys.path:
    sys.path.insert(0, str(CYBORG_ROOT))

# ========= CybORG & Agents & ScenarioGenerator =========
from CybORG import CybORG
from CybORG.Simulator.Scenarios.FileReaderScenarioGenerator import FileReaderScenarioGenerator
from CybORG.Agents.SimpleAgents.Meander import RedMeanderAgent
from CybORG.Agents.SimpleAgents.B_line import B_lineAgent
from CybORG.Agents.SimpleAgents.BlueReactAgent import BlueReactRemoveAgent, BlueReactRestoreAgent

# ========= 场景路径定位（基于已导入的 CybORG 包）=========
def get_scenario2_yaml() -> Path:
    """从已导入的 CybORG 包定位 Scenario2.yaml（与你贴的 FileReaderScenarioGenerator 兼容）"""
    pkg_file = Path(inspect.getfile(CybORG))            # .../third_party/CybORG/CybORG/__init__.py
    scenario = pkg_file.parent / "Simulator" / "Scenarios" / "scenario_files" / "Scenario2.yaml"
    if not scenario.exists():
        # 兜底：在包内递归搜索
        cyborg_pkg = pkg_file.parent
        cands = list(cyborg_pkg.rglob("Scenario2.yaml"))
        if not cands:
            raise FileNotFoundError(f"找不到 Scenario2.yaml，检查 {cyborg_pkg} 下的场景文件。")
        return cands[0]
    return scenario

# ========= 构建环境（显式使用 FileReaderScenarioGenerator）=========
def make_env(red="meander", seed=123):
    scenario_yaml = str(get_scenario2_yaml())
    sg = FileReaderScenarioGenerator(scenario_yaml)
    red_agent = RedMeanderAgent() if red == "meander" else B_lineAgent()
    cyborg = CybORG(sg, 'sim', agents={'Red': red_agent}, seed=seed)
    return cyborg

# ========= 动作解析 & 时间 =========
ACTION_RE = re.compile(r"^(?P<act>\w+)(?:\s+(?P<target>.+))?$")
def parse_action(s: str):
    if not s:
        return {"type": None, "target": None}
    m = ACTION_RE.match(s.strip())
    return {"type": m.group("act"), "target": m.group("target")} if m else {"type": s.strip(), "target": None}

def utc_now_iso():
    return datetime.now(timezone.utc).isoformat()

# ========= 主流程：运行并记录 JSONL =========
def run_and_log(blue_agent, red_mode, out_path, episodes=1, max_steps=300, seed=123):
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    with out.open("w", encoding="utf-8") as f:
        # meta 行（便于下游识别）
        f.write(json.dumps({
            "meta": True,
            "t": utc_now_iso(),
            "file": str(out.resolve()),
            "episodes": episodes,
            "max_steps": max_steps,
            "seed": seed,
            "red_mode": red_mode,
            "blue_agent": type(blue_agent).__name__
        }, ensure_ascii=False) + "\n")

        for ep in range(episodes):
            cyborg = make_env(red=red_mode, seed=seed + ep)
            step_stats = Counter()
            cyborg.reset()
            obs = cyborg.get_observation('Blue')

            for t in range(max_steps):
                blue_as = cyborg.get_action_space('Blue')
                blue_action = blue_agent.get_action(obs, blue_as)

                ret = cyborg.step('Blue', blue_action)
                if isinstance(ret, tuple) and len(ret) >= 4:
                    obs, reward, done, info = ret[:4]
                else:
                    obs = cyborg.get_observation('Blue'); reward, done, info = 0.0, False, {}

                # 最近红/蓝动作（不同版本字段名可能不同，做兼容）
                try:
                    last_red = str(cyborg.environment_controller.get_last_action('Red'))
                except Exception:
                    last_red = ""
                try:
                    last_blue = str(cyborg.environment_controller.get_last_action('Blue'))
                except Exception:
                    last_blue = str(blue_action)

                pr, pb = parse_action(last_red), parse_action(last_blue)
                if pr["type"]: step_stats[f"RED::{pr['type']}"]  += 1
                if pb["type"]: step_stats[f"BLUE::{pb['type']}"] += 1

                rec = {
                    "t": utc_now_iso(),
                    "episode": ep,
                    "step": t,
                    "red_action": last_red,
                    "red_action_raw": last_red,      # 兜底字段
                    "red_type": pr["type"],
                    "red_target": pr["target"],
                    "blue_action": last_blue,
                    "blue_action_raw": last_blue,    # 兜底字段
                    "blue_type": pb["type"],
                    "blue_target": pb["target"],
                    "reward": float(reward) if reward is not None else 0.0,
                    "done": bool(done),
                    "info": info if isinstance(info, dict) else {"info": str(info)},
                    "obs_top_keys": list(obs.keys())[:12] if hasattr(obs, "keys") else str(type(obs)),
                }
                print(f"[{red_mode:7s} | {type(blue_agent).__name__[:18]:18s}] "
                      f"ep={ep:02d} t={t:03d} R={rec['reward']:.3f} done={rec['done']}  "
                      f"RED={rec['red_action']}  BLUE={rec['blue_action']}")
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                if done:
                    break

            print("\n[Episode summary]", f"red={red_mode}", f"blue={type(blue_agent).__name__}", f"ep={ep}")
            for k, v in step_stats.most_common(10):
                print(f"  {k:22s}: {v}")
            print()

    print("Log saved →", out.resolve())

# ========= CLI =========
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--max_steps", type=int, default=300)
    parser.add_argument("--seed",     type=int, default=123)
    args = parser.parse_args()

    logs_dir = PROJ_ROOT / "logs"
    run_and_log(BlueReactRemoveAgent(),  "meander", logs_dir / "cc2_remove_meander.jsonl",  args.episodes, args.max_steps, args.seed)
    run_and_log(BlueReactRestoreAgent(), "meander", logs_dir / "cc2_restore_meander.jsonl", args.episodes, args.max_steps, args.seed)
    run_and_log(BlueReactRemoveAgent(),  "bline",   logs_dir / "cc2_remove_bline.jsonl",    args.episodes, args.max_steps, args.seed)
    run_and_log(BlueReactRestoreAgent(), "bline",   logs_dir / "cc2_restore_bline.jsonl",   args.episodes, args.max_steps, args.seed)
    print("✅ logger_plus finished.")

if __name__ == "__main__":
    main()
