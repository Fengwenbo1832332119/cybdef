# -*- coding: utf-8 -*-
"""
根据一条行为 step + seed_graph，生成用于 CollaborativePipeline 的 knowledge_base。

输入：
- step: 你行为日志里的单条 JSON（包含 facts / explain / action_name 等）
- seed_graph: configs/seed_graph.json 中定义的拓扑与关键资产

输出：
- knowledge_base: List[Dict]，元素形如：
  {
    "id": "step4_fact_suspicious",
    "text": "Suspicious activity has been detected (episode=4, step=4).",
    "source": "env_facts",
    "citation": "env://episode4/step4/suspicious_activity",
    "score": 0.9
  }
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List


def build_kb_from_step(step: Dict[str, Any], seed_graph: Dict[str, Any]) -> List[Dict[str, Any]]:
    """根据一条 step json + seed_graph 构造一个 knowledge_base 列表。"""

    kb: List[Dict[str, Any]] = []

    episode = step.get("episode")
    t = step.get("step")
    facts = step.get("facts", {}) or {}
    explain = step.get("explain", {}) or {}
    action_name: str = step.get("action_name", "")

    base_citation = f"env://episode{episode}/step{t}"

    # ===== 1) facts -> 环境证据 =====
    if facts.get("suspicious_activity"):
        kb.append({
            "id": f"step{t}_fact_suspicious",
            "text": f"Suspicious activity has been detected (episode={episode}, step={t}).",
            "source": "env_facts",
            "citation": base_citation + "/suspicious_activity",
            "score": 0.9,
        })

    if facts.get("host_discovered"):
        kb.append({
            "id": f"step{t}_fact_host_discovered",
            "text": "At least one host has been discovered by Blue.",
            "source": "env_facts",
            "citation": base_citation + "/host_discovered",
            "score": 0.7,
        })

    # 这里先不展开各种 *compromised=false 的情况，等后面有真的 True 再加
    recent_reward = facts.get("recent_reward")
    if recent_reward is not None:
        kb.append({
            "id": f"step{t}_fact_recent_reward",
            "text": f"Recent environment reward is {recent_reward}.",
            "source": "env_facts",
            "citation": base_citation + "/recent_reward",
            "score": 0.5,
        })

    # ===== 2) 从 action_name 解析目标主机 =====
    # 例如 "DecoyVsftpd_Enterprise0" -> Enterprise0
    target_host = None
    parts = action_name.split("_")
    if len(parts) >= 2:
        cand = parts[-1]  # "Enterprise0"
        for h in seed_graph.get("hosts", []):
            if h.get("id") == cand:
                target_host = h
                break

    if target_host is not None:
        kb.append({
            "id": f"step{t}_target_{target_host['id']}",
            "text": (
                f"Current blue action targets host {target_host['id']} "
                f"({target_host.get('role')} with criticality={target_host.get('criticality')})"
            ),
            "source": "blue_action",
            "citation": base_citation + "/action_target",
            "score": 0.9,
        })

        # ===== 3) seed_graph 里和这个 host 相关的拓扑证据 =====
        kb.append({
            "id": f"seed_host_{target_host['id']}",
            "text": (
                f"{target_host['id']} is a {target_host.get('role')} with criticality="
                f"{target_host.get('criticality')} in {target_host.get('subnet')}."
            ),
            "source": "seed_graph",
            "citation": f"seed://host/{target_host['id']}",
            "score": 1.0,
        })

        # 谁能 reach 到它 (如 User0->Enterprise0, User1->Enterprise0)
        for edge in seed_graph.get("edges", []):
            if edge.get("relation") == "reachable" and edge.get("target") == target_host["id"]:
                src = edge.get("source")
                kb.append({
                    "id": f"seed_path_{src}_{target_host['id']}",
                    "text": f"{src} can reach {target_host['id']} (relation: reachable).",
                    "source": "seed_graph",
                    "citation": f"seed://path/{src}->{target_host['id']}",
                    "score": 0.9,
                })

        # 它能 reach 到谁 (如 Enterprise0->Op_Server0)
        for edge in seed_graph.get("edges", []):
            if edge.get("relation") == "reachable" and edge.get("source") == target_host["id"]:
                dst = edge.get("target")
                kb.append({
                    "id": f"seed_path_{target_host['id']}_{dst}",
                    "text": f"{target_host['id']} can reach {dst} (relation: reachable).",
                    "source": "seed_graph",
                    "citation": f"seed://path/{target_host['id']}->{dst}",
                    "score": 1.0,
                })

    # ===== 4) explain 里的规则信息 -> 规则证据 =====
    for r in explain.get("active_mask_rules", []):
        kb.append({
            "id": f"rule_{r.get('name')}",
            "text": (
                f"Rule '{r.get('name')}' with condition {r.get('condition')} is active and applies a "
                f"{r.get('effect', {}).get('type')} on actions {r.get('effect', {}).get('actions')}."
            ),
            "source": "cskg_rules",
            "citation": f"cskg://rule/{r.get('name')}",
            "score": 0.8,
        })

    for r in explain.get("active_prior_rules", []):
        kb.append({
            "id": f"rule_{r.get('name')}",
            "text": (
                f"Rule '{r.get('name')}' with condition {r.get('condition')} increases priority of "
                f"actions {r.get('effect', {}).get('actions')} by {r.get('effect', {}).get('value')}."
            ),
            "source": "cskg_rules",
            "citation": f"cskg://rule/{r.get('name')}",
            "score": 0.8,
        })

    # 规则给出的推荐动作
    for rec in explain.get("recommended_actions", []):
        kb.append({
            "id": f"step{t}_rec_{rec.get('action')}",
            "text": f"Recommended action from prior rules: {rec.get('action')} with priority={rec.get('priority')}.",
            "source": "cskg_prior",
            "citation": base_citation + "/recommended_actions",
            "score": 0.7,
        })

    return kb


if __name__ == "__main__":
    # 按你的项目结构修改下面两行路径
    root = Path(__file__).resolve().parents[2]
    seed_graph_path = root / "scripts" / "configs" / "seed_graph.json"
    step_path = root / "scripts" /"tmp_step.json"

    seed_graph = json.loads(seed_graph_path.read_text(encoding="utf-8"))
    step = json.loads(step_path.read_text(encoding="utf-8"))

    kb = build_kb_from_step(step, seed_graph)
    print(json.dumps(kb, indent=2, ensure_ascii=False))
