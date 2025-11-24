# -*- coding: utf-8 -*-
"""
根据行为重放 JSONL（behavior_ac_*.jsonl）+ seed_graph.json 构造多步 KB（A+B+C+D 全要）：

A. 环境事实（facts）
B. 规则触发（active_mask_rules / active_prior_rules）
C. 推荐动作（recommended_actions）
D. 蓝方真实动作（action_name / action_idx）

输出：一个大的 knowledge_base 列表，元素示例：

{
  "id": "ep20_step4_step4_fact_suspicious",
  "text": "Suspicious activity has been detected (episode=20, step=4).",
  "source": "env_facts",
  "citation": "env://episode20/step4/suspicious_activity",
  "score": 0.9,
  "episode": 20,
  "step": 4,
  "kind": "fact"
}
"""

from __future__ import annotations


import sys
from pathlib import Path

# ---- 加这一段 ----
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

# 项目根目录 = c:\cybdef
REPO_ROOT = Path(__file__).resolve().parents[2]

# 复用你之前的单步 KB 构造器（里面已经包含 facts + rules + recs + seed_graph 拓扑）
from scripts.utils.build_kb_from_step import build_kb_from_step


KBEntry = Dict[str, Any]


def iter_records(jsonl_path: Path) -> Iterable[Dict[str, Any]]:
    """按行读取行为重放 JSONL。"""
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except Exception:
                # 某行坏掉就跳过
                continue


def build_kb_from_episode(
    jsonl_path: Path,
    seed_graph: Dict[str, Any],
    target_episode: Optional[int] = None,
) -> List[KBEntry]:
    """
    从 behavior_ac_*.jsonl 构造 KB：

    - 若 target_episode 为 None：包含所有 episode。
    - 若指定 episode：仅构造该 episode 的 KB。
    """

    all_kb: List[KBEntry] = []

    for rec in iter_records(jsonl_path):
        if rec.get("mode") != "cskg":
            continue

        ep = int(rec.get("episode", -1))
        st = int(rec.get("step", -1))
        if ep < 0 or st < 0:
            continue

        if target_episode is not None and ep != target_episode:
            continue

        base_citation = f"env://episode{ep}/step{st}"

        # ========= 1) 先用单步构造器：A + B + C =========
        step_kb = build_kb_from_step(rec, seed_graph)

        for ev in step_kb:
            new_ev = dict(ev)
            old_id = str(new_ev.get("id", "")).strip()
            if old_id:
                new_id = f"ep{ep}_step{st}_{old_id}"
            else:
                new_id = f"ep{ep}_step{st}_ev"

            new_ev["id"] = new_id
            # 补齐这些字段，保证和 Retriever/Verifier 兼容
            new_ev.setdefault("source", "kb")
            new_ev.setdefault("citation", base_citation)
            new_ev.setdefault("score", 0.5)
            new_ev["episode"] = ep
            new_ev["step"] = st
            # 粗略标个类型（方便以后分析）
            if old_id.startswith("step") and "_fact_" in old_id:
                new_ev.setdefault("kind", "fact")
            elif old_id.startswith("rule_"):
                new_ev.setdefault("kind", "rule")
            elif "_rec_" in old_id:
                new_ev.setdefault("kind", "recommendation")
            else:
                new_ev.setdefault("kind", "mixed")

            all_kb.append(new_ev)

        # ========= 2) D：蓝方真实动作 =========
        action_name = rec.get("action_name")
        action_idx = rec.get("action_idx")
        if action_name is not None:
            action_entry: KBEntry = {
                "id": f"ep{ep}_step{st}_action",
                "text": (
                    f"At episode {ep}, step {st}, blue executed action "
                    f"{action_name} (idx={action_idx})."
                ),
                "source": "blue_action",
                "citation": base_citation + "/action",
                "score": 0.9,
                "episode": ep,
                "step": st,
                "kind": "blue_action",
            }
            all_kb.append(action_entry)

    return all_kb


def main() -> None:
    parser = argparse.ArgumentParser(
        description="从 behavior_ac_*.jsonl + seed_graph.json 构造多步 KB"
    )
    parser.add_argument(
        "--log-path",
        type=str,
        required=True,
        help="行为重放 JSONL 相对路径或绝对路径，例如 logs/behavior_ac_upd250_1763607823.jsonl",
    )
    parser.add_argument(
        "--seed-graph",
        type=str,
        default="configs/seed_graph.json",
        help="seed_graph.json 的相对或绝对路径（默认：configs/seed_graph.json）",
    )
    parser.add_argument(
        "--episode",
        type=int,
        default=None,
        help="要抽取的 episode 编号；不指定则抽取所有 episode",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="输出 KB 的 JSON 文件路径（默认：reports/kb_episodeX.json 或 kb_all_episodes.json）",
    )

    args = parser.parse_args()

    # 处理路径：支持绝对路径与相对 REPO_ROOT 的路径
    log_path = Path(args.log_path)
    if not log_path.is_absolute():
        log_path = REPO_ROOT / log_path

    seed_graph_path = Path(args.seed_graph)
    if not seed_graph_path.is_absolute():
        seed_graph_path = REPO_ROOT / seed_graph_path

    if not log_path.exists():
        raise FileNotFoundError(f"行为日志不存在: {log_path}")
    if not seed_graph_path.exists():
        raise FileNotFoundError(f"seed_graph.json 不存在: {seed_graph_path}")

    seed_graph = json.loads(seed_graph_path.read_text(encoding="utf-8"))
    kb = build_kb_from_episode(
        jsonl_path=log_path,
        seed_graph=seed_graph,
        target_episode=args.episode,
    )

    if args.out:
        out_path = Path(args.out)
        if not out_path.is_absolute():
            out_path = REPO_ROOT / out_path
    else:
        if args.episode is None:
            out_path = REPO_ROOT / "reports" / "kb_all_episodes.json"
        else:
            out_path = REPO_ROOT / "reports" / f"kb_episode{args.episode}.json"

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(kb, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"[INFO] KB 构造完成，共 {len(kb)} 条，已写入: {out_path}")


if __name__ == "__main__":
    main()
