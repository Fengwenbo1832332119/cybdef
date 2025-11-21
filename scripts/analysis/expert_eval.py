# -*- coding: utf-8 -*-
"""专家理解度 & Root-Cause 定位评估脚本。

- 读取专家标注（JSONL/CSV），字段至少包含：case_id、expert_id、understanding_seconds、root_cause_correct。
- 计算整体/分 case/分专家的理解时间和定位成功率。
- 将结果写入 reports/expert_eval，并自动向主评测报告（reports/policiespeak_eval.md）附加摘要。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, Tuple

import pandas as pd

from scripts.common.paths import REPORTS_DIR, ensure_dirs

MAIN_REPORT = Path(__file__).resolve().parents[2] / "reports" / "policiespeak_eval.md"


# ------------------ I/O ------------------

def load_feedback(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=["case_id", "expert_id", "understanding_seconds", "root_cause_correct", "notes"])
    if path.suffix.lower() == ".jsonl":
        rows = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        return pd.DataFrame(rows)
    return pd.read_csv(path)


# ------------------ 统计 ------------------

def _boolify(value) -> bool:
    if isinstance(value, str):
        return value.lower() in {"1", "true", "yes", "y"}
    return bool(value)


def normalize_df(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["case_id", "expert_id", "understanding_seconds", "root_cause_correct", "notes"])
    out = df.copy()
    u_col = out["understanding_seconds"] if "understanding_seconds" in out else pd.Series([float("nan")] * len(out))
    out["understanding_seconds"] = pd.to_numeric(u_col, errors="coerce")
    r_col = out["root_cause_correct"] if "root_cause_correct" in out else pd.Series([False] * len(out))
    out["root_cause_correct"] = r_col.apply(_boolify)
    out["case_id"] = out.get("case_id").astype(str)
    out["expert_id"] = out.get("expert_id").astype(str)
    return out.dropna(subset=["understanding_seconds", "root_cause_correct"])


def aggregate(df: pd.DataFrame) -> Tuple[Dict[str, float], pd.DataFrame, pd.DataFrame]:
    if df.empty:
        return {}, pd.DataFrame(), pd.DataFrame()
    overall = {
        "avg_understanding_seconds": float(df["understanding_seconds"].mean()),
        "median_understanding_seconds": float(df["understanding_seconds"].median()),
        "root_cause_success_rate": float(df["root_cause_correct"].mean()),
        "num_sessions": int(len(df)),
        "num_cases": int(df["case_id"].nunique()),
        "num_experts": int(df["expert_id"].nunique()),
    }
    per_case = (
        df.groupby("case_id")
        .agg(
            avg_understanding_seconds=("understanding_seconds", "mean"),
            median_understanding_seconds=("understanding_seconds", "median"),
            root_cause_success_rate=("root_cause_correct", "mean"),
            sessions=("expert_id", "count"),
        )
        .reset_index()
    )
    per_expert = (
        df.groupby("expert_id")
        .agg(
            avg_understanding_seconds=("understanding_seconds", "mean"),
            median_understanding_seconds=("understanding_seconds", "median"),
            root_cause_success_rate=("root_cause_correct", "mean"),
            cases=("case_id", "nunique"),
        )
        .reset_index()
    )
    return overall, per_case, per_expert


# ------------------ 报告 ------------------

def write_report(out_dir: Path, overall: Dict[str, float], per_case: pd.DataFrame, per_expert: pd.DataFrame, source: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(overall, ensure_ascii=False, indent=2), encoding="utf-8")
    per_case.to_csv(out_dir / "per_case.csv", index=False, encoding="utf-8-sig")
    per_expert.to_csv(out_dir / "per_expert.csv", index=False, encoding="utf-8-sig")

    lines = ["# 专家理解度与 Root-Cause 评测", ""]
    lines.append(f"- 数据源: {source.name}")
    if overall:
        lines.extend(
            [
                f"- 平均理解时间: {overall['avg_understanding_seconds']:.2f}s",
                f"- 中位理解时间: {overall['median_understanding_seconds']:.2f}s",
                f"- Root-Cause 成功率: {overall['root_cause_success_rate']:.2%}",
                f"- 会话数: {overall['num_sessions']} / 专家: {overall['num_experts']} / case: {overall['num_cases']}",
            ]
        )
    else:
        lines.append("- 暂无数据")
    lines.append("")
    if not per_case.empty:
        lines.append("## 分 case 摘要 (Top 5)")
        for _, row in per_case.head(5).iterrows():
            lines.append(
                f"- {row['case_id']}: 成功率 {row['root_cause_success_rate']:.2%}, 平均理解 {row['avg_understanding_seconds']:.2f}s"
            )
    (out_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")


def inject_into_main_report(overall: Dict[str, float], report_path: Path = MAIN_REPORT):
    if not overall:
        return
    if report_path.exists():
        content = report_path.read_text(encoding="utf-8").splitlines()
    else:
        content = ["# PolicySpeak 评测报告", ""]
    content.append("## 专家理解/Root-Cause 评测")
    content.append(f"- 平均理解时间: {overall['avg_understanding_seconds']:.2f}s")
    content.append(f"- Root-Cause 成功率: {overall['root_cause_success_rate']:.2%}")
    content.append(f"- 样本: {overall['num_sessions']} 会话 / {overall['num_cases']} case / {overall['num_experts']} 专家")
    content.append("")
    report_path.write_text("\n".join(content), encoding="utf-8")


# ------------------ CLI ------------------

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, required=True, help="专家标注文件 (jsonl/csv)")
    ap.add_argument("--out", type=Path, default=REPORTS_DIR / "expert_eval")
    ap.add_argument("--skip-main-report", action="store_true", help="不写入主评测报告")
    return ap.parse_args()


def main():
    args = parse_args()
    ensure_dirs()
    df = normalize_df(load_feedback(args.input))
    overall, per_case, per_expert = aggregate(df)
    write_report(args.out, overall, per_case, per_expert, args.input)
    if not args.skip_main_report:
        inject_into_main_report(overall)
    print(f"✅ 专家评测结果已写入: {args.out}")


if __name__ == "__main__":
    main()