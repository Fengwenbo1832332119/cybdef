# -*- coding: utf-8 -*-
"""
解释/反事实评测汇总脚本
- 干预一致性曲线（含 λ=0 对照）
- SHAP / LIME-Graph 重要性与 MSE 证据重合度
- 最小修改反事实验证
- 一键生成报告（时间线、拓扑热力图、回放引用），并将结果写入 reports/

用法示例：
python -m scripts.analysis.explain.counterfactual \
  --mse-cases graph/mse_cases.jsonl \
  --counterfactual reports/counterfactual_v4/counterfactual_results.csv \
  --intervention reports/intervention_offline_nomodel_v3/cumulative_deletion.csv \
  --lambda0-intervention reports/intervention_offline_nomodel_v3/lambda0_cumulative_deletion.csv \
  --shap shap_attribution.jsonl --lime lime_attribution.jsonl \
  --pred reports/online_suite/predictions.jsonl \
  --out reports/explain_counterfactual
"""
import argparse
import json
import math
import pathlib
from collections import defaultdict
from typing import Dict, Iterable, List, Optional, Tuple


import matplotlib.pyplot as plt  # 原有的代码
import matplotlib

# === 添加这两行代码 ===
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签 (SimHei 是黑体)
plt.rcParams['axes.unicode_minus'] = False    # 用来正常显示负号
# ======================

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scripts.common.paths import GRAPH_DIR, REPORTS_DIR, ensure_dirs

# ------------------ 通用 I/O ------------------

def load_jsonl(path: pathlib.Path) -> List[dict]:
    items = []
    if not path or not path.exists():
        return items
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    items.append(json.loads(line))
                except json.JSONDecodeError:
                    # 允许带 BOM/隐藏字符的 jsonl
                    line = line.encode("utf-8", errors="ignore").decode("utf-8", errors="ignore")
                    items.append(json.loads(line))
    return items


def ensure_out_dir(out_dir: pathlib.Path) -> pathlib.Path:
    ensure_dirs()
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


# ------------------ 数据准备：MSE 证据索引 ------------------

def case_identifier(case: dict) -> str:
    stem = pathlib.Path(case.get("file", "")).stem
    step = case.get("blue_step")
    return f"{stem}#step{step}"


def assign_evidence_ids(cases: Iterable[dict]) -> Tuple[List[dict], Dict[str, List[str]]]:
    """为 MSE/MCE 事件补充 evidence_id，并返回 case_id -> evidence_id 列表映射。"""
    mapping: Dict[str, List[str]] = defaultdict(list)
    updated = []
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


# ------------------ 干预一致性曲线 ------------------

def compute_consistency_curve(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["k", "strategy", "delta_mean", "consistency"])
    agg = df.groupby(["strategy", "k"])['delta'].mean().reset_index(name="delta_mean")
    max_delta = max(agg["delta_mean"].max(), 1e-6)
    agg["consistency"] = 1.0 - (agg["delta_mean"] / max_delta)
    return agg


def compare_lambda0(baseline: pd.DataFrame, lambda0: Optional[pd.DataFrame]) -> pd.DataFrame:
    base_curve = compute_consistency_curve(baseline)
    if lambda0 is None or lambda0.empty:
        base_curve["lambda0_consistency"] = np.nan
        base_curve["consistency_drop"] = np.nan
        return base_curve
    l0_curve = compute_consistency_curve(lambda0)
    merged = pd.merge(
        base_curve,
        l0_curve[["strategy", "k", "consistency"]].rename(columns={"consistency": "lambda0_consistency"}),
        on=["strategy", "k"],
        how="left",
    )
    merged["consistency_drop"] = merged["consistency"] - merged["lambda0_consistency"]
    return merged


# ------------------ 重要性重合度 ------------------

def _top_ids(importance: Dict[str, float], topk: int) -> List[str]:
    if not importance:
        return []
    items = sorted(importance.items(), key=lambda kv: abs(kv[1]), reverse=True)
    return [k for k, _ in items[:topk]]


def load_importances(path: Optional[pathlib.Path]) -> Dict[str, Dict[str, float]]:
    """读取 shap/lime 图重要性：返回 sample_id -> {evidence_id: score}"""
    if not path or not path.exists():
        return {}
    data: Dict[str, Dict[str, float]] = {}
    for row in load_jsonl(path):
        sid = str(row.get("id") or row.get("case_id") or row.get("name"))
        imp = row.get("importances") or row.get("importance") or {}
        if isinstance(imp, list):
            # 支持 [ {"id":..., "score":...} ] 结构
            imp = {str(x.get("id")): float(x.get("score", 0.0)) for x in imp if x}
        if isinstance(imp, dict):
            data[sid] = {str(k): float(v) for k, v in imp.items()}
    return data


def compute_overlap(
    mse_mapping: Dict[str, List[str]],
    shap: Dict[str, Dict[str, float]],
    lime: Dict[str, Dict[str, float]],
    topk: int = 5,
) -> pd.DataFrame:
    rows = []
    keys = set(mse_mapping) | set(shap) | set(lime)
    for cid in sorted(keys):
        mse_ids = set(mse_mapping.get(cid, []))
        shap_ids = set(_top_ids(shap.get(cid, {}), topk))
        lime_ids = set(_top_ids(lime.get(cid, {}), topk))
        for name, ids in [("shap", shap_ids), ("lime", lime_ids)]:
            inter = mse_ids & ids
            union = mse_ids | ids if (mse_ids or ids) else set()
            jaccard = len(inter) / len(union) if union else np.nan
            cover = len(inter) / len(mse_ids) if mse_ids else np.nan
            rows.append(
                {
                    "case_id": cid,
                    "method": name,
                    "topk": topk,
                    "overlap": len(inter),
                    "mse_size": len(mse_ids),
                    "jaccard": jaccard,
                    "coverage": cover,
                    "evidence_ids": ",".join(sorted(inter)),
                }
            )
    return pd.DataFrame(rows)


# ------------------ 反事实最小修改 ------------------

def minimal_counterfactuals(cf_csv: pathlib.Path) -> pd.DataFrame:
    if not cf_csv.exists():
        return pd.DataFrame(columns=["file", "mode", "best_op", "delta"])
    df = pd.read_csv(cf_csv)
    if df.empty:
        return pd.DataFrame(columns=["file", "mode", "best_op", "delta"])
    rows = []
    for (file, mode), sub in df.groupby(["file", "mode"]):
        positive = sub[sub["delta"] > 0]
        if positive.empty:
            rows.append({"file": file, "mode": mode, "best_op": None, "delta": 0.0})
            continue
        best = positive.loc[positive["delta"].idxmin()]
        rows.append(
            {
                "file": file,
                "mode": mode,
                "best_op": best["op"],
                "delta": float(best["delta"]),
            }
        )
    return pd.DataFrame(rows)


# ------------------ 校准与一致性 ------------------

def load_predictions(pred_path: Optional[pathlib.Path]) -> pd.DataFrame:
    rows = []
    if pred_path and pred_path.exists():
        for row in load_jsonl(pred_path):
            y_true = row.get("label") if "label" in row else row.get("y_true")
            prob = row.get("prob")
            if prob is None and isinstance(row.get("probs"), (list, tuple)):
                prob = row.get("probs")[1] if len(row.get("probs")) > 1 else row.get("probs")[0]
            if y_true is None or prob is None:
                continue
            rows.append({"id": row.get("id") or row.get("case_id"), "y_true": float(y_true), "prob": float(prob)})
    return pd.DataFrame(rows)


def calibration_metrics(pred_df: pd.DataFrame) -> Tuple[float, float]:
    if pred_df.empty:
        return math.nan, math.nan
    probs = np.clip(pred_df["prob"].values, 1e-8, 1 - 1e-8)
    labels = pred_df["y_true"].values
    brier = float(np.mean((probs - labels) ** 2))
    nll = float(-np.mean(labels * np.log(probs) + (1 - labels) * np.log(1 - probs)))
    return brier, nll


def consistency_from_cf(cf_df: pd.DataFrame) -> float:
    if cf_df.empty:
        return math.nan
    # 认为 delta>0 表示“干预削弱解释” → 一致
    return float((cf_df["delta"] > 0).mean())


# ------------------ 可视化 ------------------

def plot_consistency(curve: pd.DataFrame, out_png: pathlib.Path):
    if curve.empty:
        return
    plt.figure(figsize=(7, 4))
    for strat, sub in curve.groupby("strategy"):
        plt.plot(sub["k"], sub["consistency"], marker="o", label=f"{strat}")
        if "lambda0_consistency" in sub:
            plt.plot(sub["k"], sub["lambda0_consistency"], marker="x", linestyle="--", label=f"{strat} (λ=0)")
    plt.xlabel("k (删除/干预步数)")
    plt.ylabel("consistency (1-Δ/Δmax)")
    plt.title("干预一致性曲线")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=160)
    plt.close()


def plot_heatmap(cf_df: pd.DataFrame, out_png: pathlib.Path):
    if cf_df.empty:
        return
    pivot = cf_df.pivot_table(index="type", columns="op", values="delta", aggfunc="mean").fillna(0.0)
    plt.figure(figsize=(max(6, 0.4 * len(pivot.columns)), 4))
    plt.imshow(pivot, aspect="auto", cmap="viridis")
    plt.yticks(range(len(pivot.index)), pivot.index)
    plt.xticks(range(len(pivot.columns)), pivot.columns, rotation=45, ha="right")
    plt.colorbar(label="Δscore")
    plt.title("反事实拓扑热力图")
    plt.tight_layout()
    plt.savefig(out_png, dpi=160)
    plt.close()


def plot_overlap_timeline(overlap_df: pd.DataFrame, out_png: pathlib.Path):
    if overlap_df.empty:
        return
    plt.figure(figsize=(7, 3.5))
    for method, sub in overlap_df.groupby("method"):
        plt.plot(range(len(sub)), sub["coverage"], marker="o", label=method)
    plt.xlabel("case 序号")
    plt.ylabel("MSE 覆盖率")
    plt.title("解释重要性-证据重合度时间线")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=160)
    plt.close()


# ------------------ 报告生成 ------------------

def generate_report(
    out_dir: pathlib.Path,
    curve_csv: pathlib.Path,
    overlap_csv: pathlib.Path,
    minimal_csv: pathlib.Path,
    calibration_txt: pathlib.Path,
    visual_map_csv: pathlib.Path,
    plots: Dict[str, pathlib.Path],
):
    lines = ["# 反事实/解释报告", ""]
    lines.append(f"- 干预一致性曲线：{curve_csv.name}")
    lines.append(f"- 重要性重合度：{overlap_csv.name}")
    lines.append(f"- 最小修改反事实：{minimal_csv.name}")
    lines.append(f"- 校准指标：{calibration_txt.name}")
    lines.append(f"- 证据/可视 ID 映射：{visual_map_csv.name}")
    lines.append("")
    for label, path in plots.items():
        if path.exists():
            lines.append(f"![{label}]({path.name})")
    (out_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")


# ------------------ 主流程 ------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mse-cases", type=pathlib.Path, default=GRAPH_DIR / "mse_cases.jsonl")
    ap.add_argument("--counterfactual", type=pathlib.Path, default=REPORTS_DIR / "counterfactual_v4" / "counterfactual_results.csv")
    ap.add_argument("--intervention", type=pathlib.Path, default=REPORTS_DIR / "intervention_offline_nomodel_v3" / "cumulative_deletion.csv")
    ap.add_argument("--lambda0-intervention", type=pathlib.Path, default=None)
    ap.add_argument("--shap", type=pathlib.Path, default=None)
    ap.add_argument("--lime", type=pathlib.Path, default=None)
    ap.add_argument("--pred", type=pathlib.Path, default=None, help="带 label/prob 的预测 jsonl，用于 Brier/NLL")
    ap.add_argument("--out", type=pathlib.Path, default=REPORTS_DIR / "explain_counterfactual")
    ap.add_argument("--topk", type=int, default=5, help="计算 SHAP/LIME 重合度时的 top-k")
    args = ap.parse_args()

    out_dir = ensure_out_dir(args.out)

    # 1) MSE 证据索引
    cases, mse_mapping = assign_evidence_ids(load_jsonl(args.mse_cases))
    visual_map = []
    for cid, ids in mse_mapping.items():
        for evid in ids:
            visual_map.append({"case_id": cid, "evidence_id": evid, "visual_id": f"viz::{evid}"})
    visual_df = pd.DataFrame(visual_map)
    visual_map_csv = out_dir / "evidence_visual_map.csv"
    visual_df.to_csv(visual_map_csv, index=False, encoding="utf-8-sig")

    # 2) 干预一致性（含 λ=0 对照）
    inter_df = pd.read_csv(args.intervention) if args.intervention.exists() else pd.DataFrame()
    l0_df = pd.read_csv(args.lambda0_intervention) if args.lambda0_intervention and args.lambda0_intervention.exists() else None
    curve = compare_lambda0(inter_df, l0_df)
    curve_csv = out_dir / "intervention_consistency.csv"
    curve.to_csv(curve_csv, index=False, encoding="utf-8-sig")
    curve_png = out_dir / "intervention_consistency.png"
    plot_consistency(curve, curve_png)

    # 3) 重要性重合度
    shap_imp = load_importances(args.shap)
    lime_imp = load_importances(args.lime)
    overlap = compute_overlap(mse_mapping, shap_imp, lime_imp, topk=args.topk)
    overlap_csv = out_dir / "importance_overlap.csv"
    overlap.to_csv(overlap_csv, index=False, encoding="utf-8-sig")
    timeline_png = out_dir / "overlap_timeline.png"
    plot_overlap_timeline(overlap, timeline_png)

    # 4) 反事实最小修改
    cf_df = pd.read_csv(args.counterfactual) if args.counterfactual.exists() else pd.DataFrame()
    minimal_df = minimal_counterfactuals(args.counterfactual)
    minimal_csv = out_dir / "minimal_counterfactual.csv"
    minimal_df.to_csv(minimal_csv, index=False, encoding="utf-8-sig")
    heatmap_png = out_dir / "counterfactual_heatmap.png"
    plot_heatmap(cf_df, heatmap_png)

    # 5) 校准 + 一致性指标
    pred_df = load_predictions(args.pred)
    brier, nll = calibration_metrics(pred_df)
    cf_consistency = consistency_from_cf(cf_df)
    with (out_dir / "calibration.txt").open("w", encoding="utf-8") as w:
        w.write(f"Calibration-E (Brier): {brier}\n")
        w.write(f"Calibration-E (NLL): {nll}\n")
        w.write(f"Consistency (counterfactual Δ>0 ratio): {cf_consistency}\n")
    calibration_txt = out_dir / "calibration.txt"

    # 6) 汇总报告
    generate_report(
        out_dir,
        curve_csv=curve_csv,
        overlap_csv=overlap_csv,
        minimal_csv=minimal_csv,
        calibration_txt=calibration_txt,
        visual_map_csv=visual_map_csv,
        plots={
            "干预一致性": curve_png,
            "重要性重合时间线": timeline_png,
            "反事实拓扑热力图": heatmap_png,
        },
    )

    print("✅ explain/counterfactual 报告已生成：", out_dir)


if __name__ == "__main__":
    main()