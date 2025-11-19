# -*- coding: utf-8 -*-
"""
faithfulness_eval
- 读取 causal_replay_results.csv（或目录内所有同名CSV合并）
- 对每条日志内部：计算各变体相对 baseline 的 Δrule_score、Δreward（Red/Blue）
- 给出相关性（Pearson/Spearman），可视化散点 + 拟合线，并输出 summary.md

用法（单行）：
python C:\cybdef\scripts\metrics\faithfulness_eval.py --csv C:\cybdef\reports\causal_replay\causal_replay_results.csv --out C:\cybdef\reports\faithfulness
"""

import os, sys, glob, argparse, pathlib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr

# ==== 根目录注入（可选） ====
ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.common.paths import REPORTS_DIR, ensure_dirs

def load_results(csv_or_dir: str) -> pd.DataFrame:
    p = pathlib.Path(csv_or_dir)
    if p.is_file():
        return pd.read_csv(p)
    files = sorted(pathlib.Path(csv_or_dir).glob("**/causal_replay_results.csv"))
    if not files:
        raise FileNotFoundError(f"No causal_replay_results.csv under: {csv_or_dir}")
    dfs = [pd.read_csv(f) for f in files]
    return pd.concat(dfs, ignore_index=True)

def compute_deltas(df: pd.DataFrame) -> pd.DataFrame:
    out_rows=[]
    for log_name, sub in df.groupby("log"):
        sub = sub.copy()
        if "baseline" not in set(sub["variant"]):
            # 没有 baseline 无法做差
            continue
        base = sub[sub["variant"]=="baseline"].iloc[0]
        base_rs = float(base["rule_score"])
        base_rr = float(base["reward_red"])
        base_rb = float(base["reward_blue"])
        for _, r in sub.iterrows():
            if r["variant"] == "baseline":
                continue
            out_rows.append({
                "log": log_name,
                "variant": r["variant"],
                "delta_rule": float(r["rule_score"]) - base_rs,
                "delta_red":  float(r["reward_red"]) - base_rr,
                "delta_blue": float(r["reward_blue"]) - base_rb
            })
    return pd.DataFrame(out_rows)

def scatter_with_fit(x, y, title, out_png):
    if len(x) == 0:
        return
    plt.figure(figsize=(5.5,5))
    plt.scatter(x, y, alpha=0.7)
    # 拟合线（最小二乘）
    try:
        m, b = np.polyfit(x, y, 1)
        xx = np.linspace(min(x), max(x), 100)
        yy = m * xx + b
        plt.plot(xx, yy)
        sub = f" y ≈ {m:.3f} x + {b:.3f}"
    except Exception:
        sub = ""
    plt.title(title + sub)
    plt.xlabel("Δ rule_score")
    plt.ylabel("Δ reward")
    plt.tight_layout()
    plt.savefig(out_png, dpi=160)
    plt.close()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="causal_replay_results.csv 或其所在目录")
    ap.add_argument("--out", type=str, default=str(REPORTS_DIR / "faithfulness"))
    args = ap.parse_args()

    ensure_dirs()
    out_dir = pathlib.Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)

    df = load_results(args.csv)
    d = compute_deltas(df)
    if d.empty:
        print("No deltas computed (missing baseline or empty data).")
        return

    # 相关性：Red & Blue
    pr_red = pearsonr(d["delta_rule"], d["delta_red"]) if len(d)>=2 else (np.nan, np.nan)
    pr_blu = pearsonr(d["delta_rule"], d["delta_blue"]) if len(d)>=2 else (np.nan, np.nan)
    sp_red = spearmanr(d["delta_rule"], d["delta_red"]) if len(d)>=2 else (np.nan, np.nan)
    sp_blu = spearmanr(d["delta_rule"], d["delta_blue"]) if len(d)>=2 else (np.nan, np.nan)

    # 散点图
    scatter_with_fit(d["delta_rule"].values, d["delta_red"].values,
                     "Δrule vs Δreward_red", out_dir/"scatter_delta_red.png")
    scatter_with_fit(d["delta_rule"].values, d["delta_blue"].values,
                     "Δrule vs Δreward_blue", out_dir/"scatter_delta_blue.png")

    # 汇总表
    summary = pd.DataFrame({
        "metric": ["pearson_red_r","pearson_red_p","pearson_blue_r","pearson_blue_p",
                   "spearman_red_r","spearman_red_p","spearman_blue_r","spearman_blue_p"],
        "value":  [pr_red[0],pr_red[1],pr_blu[0],pr_blu[1],sp_red.correlation,sp_red.pvalue,sp_blu.correlation,sp_blu.pvalue]
    })
    summary.to_csv(out_dir/"faithfulness_summary.csv", index=False, encoding="utf-8-sig")

    # Markdown 报告
    with (out_dir/"faithfulness_summary.md").open("w", encoding="utf-8") as w:
        w.write("# Faithfulness Summary\n\n")
        w.write(f"- N variants (excl. baseline): **{len(d)}**\n")
        w.write(f"- Pearson (Δrule ↔ Δreward_red): r={pr_red[0]:.4f}, p={pr_red[1]:.4g}\n")
        w.write(f"- Pearson (Δrule ↔ Δreward_blue): r={pr_blu[0]:.4f}, p={pr_blu[1]:.4g}\n")
        w.write(f"- Spearman (Δrule ↔ Δreward_red): ρ={sp_red.correlation:.4f}, p={sp_red.pvalue:.4g}\n")
        w.write(f"- Spearman (Δrule ↔ Δreward_blue): ρ={sp_blu.correlation:.4f}, p={sp_blu.pvalue:.4g}\n")
        w.write("\n![Δrule vs Δreward_red](scatter_delta_red.png)\n\n")
        w.write("\n![Δrule vs Δreward_blue](scatter_delta_blue.png)\n")

    print("✅ Done")
    print("CSV :", out_dir/"faithfulness_summary.csv")
    print("MD  :", out_dir/"faithfulness_summary.md")
    print("PNG :", out_dir/"scatter_delta_red.png", "and", out_dir/"scatter_delta_blue.png")

if __name__ == "__main__":
    main()
