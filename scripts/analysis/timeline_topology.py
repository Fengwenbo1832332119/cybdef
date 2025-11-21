# -*- coding: utf-8 -*-
"""时间线 & 拓扑热力图生成器。

特性：
- 基于 MSE case 生成风险时间线（含 evidence 节点标注）。
- 支持“若执行/不执行”回放曲线，用于可视化干预效果。
- 拓扑热力图叠加风险强度与 MSE 证据边标注。
- 生成 PolicySpeak/证据 ID 对齐映射，保证解释可溯源。
- 自动生成报告并写入 reports/.
"""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scripts.common.paths import GRAPH_DIR, REPORTS_DIR, ensure_dirs
from scripts.common.policiespeak import build_policiespeak_template

# ------------------ 数据模型 ------------------


@dataclass
class EvidenceBinding:
    case_id: str
    evidence_id: str
    policy_id: str
    text: str
    step: Optional[int]
    risk: float

    def to_row(self) -> Dict[str, object]:
        return {
            "case_id": self.case_id,
            "evidence_id": self.evidence_id,
            "policy_id": self.policy_id,
            "policy_text": self.text,
            "step": self.step,
            "risk": self.risk,
        }


# ------------------ 读取与预处理 ------------------

def load_jsonl(path: Path) -> List[dict]:
    items: List[dict] = []
    if not path.exists():
        return items
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                items.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return items


def case_identifier(case: dict) -> str:
    stem = Path(case.get("file", "")).stem
    step = case.get("blue_step")
    return f"{stem}#step{step}"


def assign_evidence_ids(cases: Iterable[dict]) -> Tuple[List[dict], Dict[str, List[str]]]:
    mapping: Dict[str, List[str]] = defaultdict(list)
    updated: List[dict] = []
    for case in cases:
        cid = case_identifier(case)
        new_case = dict(case)
        for key in ["mse_events", "mce_events"]:
            evs = []
            for idx, ev in enumerate(case.get(key) or []):
                ev = dict(ev)
                if not ev.get("evidence_id"):
                    et = ev.get("type", "evt")
                    st = ev.get("step")
                    ev["evidence_id"] = f"{cid}|{key}|{et}|{st}|{idx}"
                mapping[cid].append(ev["evidence_id"])
                evs.append(ev)
            new_case[key] = evs
        updated.append(new_case)
    return updated, mapping


def load_replay_curves(path: Optional[Path]) -> pd.DataFrame:
    if not path or not path.exists():
        return pd.DataFrame(columns=["case_id", "step", "score_do", "score_undo"])
    df = pd.read_csv(path)
    rename_map = {"do": "score_do", "undo": "score_undo"}
    for old, new in rename_map.items():
        if old in df.columns and new not in df.columns:
            df[new] = df[old]
    if "case_id" not in df.columns:
        df["case_id"] = df.get("log") or df.get("file") or "global"
    if "step" not in df.columns:
        df["step"] = df.index
    return df[["case_id", "step", "score_do", "score_undo"]]


# ------------------ 风险建模 ------------------

DEFAULT_RISK_WEIGHTS = {
    "PrivilegeEscalate": 4.0,
    "ExploitRemoteService": 3.0,
    "DiscoverNetworkServices": 1.2,
    "DiscoverRemoteSystems": 1.0,
}


def risk_for_event(ev: dict, weights: Dict[str, float]) -> float:
    t = ev.get("type")
    base = weights.get(t, 1.0)
    sev = ev.get("severity")
    if isinstance(sev, (int, float)):
        return base * float(sev)
    return float(base)


def build_timeline(cases: Sequence[dict], weights: Dict[str, float]) -> pd.DataFrame:
    rows = []
    for case in cases:
        cid = case_identifier(case)
        for ev in case.get("mse_events", []) or []:
            rows.append(
                {
                    "case_id": cid,
                    "step": ev.get("step", case.get("blue_step")),
                    "evidence_id": ev.get("evidence_id"),
                    "risk": risk_for_event(ev, weights),
                    "type": ev.get("type"),
                }
            )
    return pd.DataFrame(rows)


def build_topology(cases: Sequence[dict], weights: Dict[str, float]) -> pd.DataFrame:
    edges: List[Dict[str, object]] = []
    for case in cases:
        cid = case_identifier(case)
        events = sorted(case.get("mse_events") or [], key=lambda x: x.get("step", 0))
        for idx, ev in enumerate(events):
            if idx + 1 < len(events):
                dst = events[idx + 1]
                edges.append(
                    {
                        "case_id": cid,
                        "src": ev.get("type"),
                        "dst": dst.get("type"),
                        "src_id": ev.get("evidence_id"),
                        "dst_id": dst.get("evidence_id"),
                        "weight": math.fabs(risk_for_event(ev, weights)) + math.fabs(risk_for_event(dst, weights)),
                    }
                )
    return pd.DataFrame(edges)


# ------------------ PolicySpeak 对齐 ------------------

def align_policiespeak(cases: Sequence[dict], weights: Dict[str, float]) -> List[EvidenceBinding]:
    bindings: List[EvidenceBinding] = []
    for case in cases:
        cid = case_identifier(case)
        for ev in case.get("mse_events") or []:
            policy = build_policiespeak_template(ev, []) or {}
            policy_id = policy.get("id") or f"ps::{ev.get('evidence_id')}"
            text = policy.get("statement") or policy.get("policy") or ""
            bindings.append(
                EvidenceBinding(
                    case_id=cid,
                    evidence_id=str(ev.get("evidence_id")),
                    policy_id=str(policy_id),
                    text=text,
                    step=ev.get("step"),
                    risk=risk_for_event(ev, weights),
                )
            )
    return bindings


# ------------------ 绘图 ------------------

def plot_risk_timeline(df: pd.DataFrame, replay: pd.DataFrame, out_png: Path):
    if df.empty:
        return
    plt.figure(figsize=(9, 4))
    grouped = df.groupby("case_id")
    for cid, sub in grouped:
        plt.plot(sub["step"], sub["risk"], marker="o", label=f"{cid} risk")
        if not replay.empty:
            replay_sub = replay[replay["case_id"] == cid]
            if not replay_sub.empty:
                plt.plot(replay_sub["step"], replay_sub["score_do"], linestyle="--", label=f"{cid} do")
                plt.plot(replay_sub["step"], replay_sub["score_undo"], linestyle=":", label=f"{cid} undo")
    plt.xlabel("Step")
    plt.ylabel("Risk / Score")
    plt.title("风险时间线与回放曲线 (do vs undo)")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out_png, dpi=160)
    plt.close()


def plot_topology_heatmap(edges: pd.DataFrame, out_png: Path):
    if edges.empty:
        return
    pivot = edges.pivot_table(index="src", columns="dst", values="weight", aggfunc="sum").fillna(0.0)
    plt.figure(figsize=(max(6, 0.6 * len(pivot.columns)), 4 + 0.2 * len(pivot.index)))
    im = plt.imshow(pivot, cmap="OrRd", aspect="auto")
    plt.colorbar(im, label="风险强度")
    plt.xticks(range(len(pivot.columns)), pivot.columns, rotation=45, ha="right")
    plt.yticks(range(len(pivot.index)), pivot.index)
    for i, src in enumerate(pivot.index):
        for j, dst in enumerate(pivot.columns):
            val = pivot.loc[src, dst]
            if val <= 0:
                continue
            plt.text(j, i, f"{val:.1f}", ha="center", va="center", color="black", fontsize=7)
    plt.title("MSE 拓扑热力图（含节点/边风险）")
    plt.tight_layout()
    plt.savefig(out_png, dpi=160)
    plt.close()


# ------------------ 报告 ------------------

def generate_report(out_dir: Path, timeline_png: Path, topo_png: Path, bindings: List[EvidenceBinding], replay_path: Optional[Path]):
    lines = ["# 时间线 & 拓扑热力图", ""]
    lines.append(f"- 时间线：{timeline_png.name}")
    lines.append(f"- 拓扑热力图：{topo_png.name}")
    if replay_path:
        lines.append(f"- 回放曲线：{replay_path.name}")
    lines.append("- PolicySpeak/证据对齐：evidence_policy_map.csv")
    lines.append("")
    if bindings:
        lines.append("## 核心解释可溯源示例")
        for b in bindings[:10]:
            lines.append(f"- {b.case_id} @ step{b.step}: {b.policy_id} ⇄ {b.evidence_id} (risk={b.risk:.2f})")
    (out_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")


# ------------------ CLI ------------------

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mse-cases", type=Path, default=GRAPH_DIR / "mse_cases.jsonl")
    ap.add_argument("--replay", type=Path, default=None, help="包含 do/undo 得分的回放曲线 CSV")
    ap.add_argument("--out", type=Path, default=REPORTS_DIR / "timeline_topology")
    ap.add_argument("--risk-json", type=Path, default=None, help="可选风险权重 JSON {type: weight}")
    return ap.parse_args()


def main():
    args = parse_args()
    ensure_dirs()
    out_dir = args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    weights = DEFAULT_RISK_WEIGHTS.copy()
    if args.risk_json and args.risk_json.exists():
        try:
            weights.update(json.loads(args.risk_json.read_text(encoding="utf-8")))
        except Exception:
            pass

    cases_raw = load_jsonl(args.mse_cases)
    cases, _ = assign_evidence_ids(cases_raw)
    timeline_df = build_timeline(cases, weights)
    replay_df = load_replay_curves(args.replay)
    edges_df = build_topology(cases, weights)
    bindings = align_policiespeak(cases, weights)

    timeline_png = out_dir / "risk_timeline.png"
    plot_risk_timeline(timeline_df, replay_df, timeline_png)

    topo_png = out_dir / "topology_heatmap.png"
    plot_topology_heatmap(edges_df, topo_png)

    binding_csv = out_dir / "evidence_policy_map.csv"
    pd.DataFrame([b.to_row() for b in bindings]).to_csv(binding_csv, index=False, encoding="utf-8-sig")

    generate_report(out_dir, timeline_png, topo_png, bindings, args.replay)
    print(f"✅ 可视化已生成：{out_dir}")


if __name__ == "__main__":
    main()