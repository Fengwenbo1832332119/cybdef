# C:\cybdef\scripts\mse_stats.py
import json
import pathlib
from collections import Counter

import pandas as pd
import matplotlib.pyplot as plt

MSE_PATH = pathlib.Path(r"/graph/mse_cases.jsonl")
OUT_DIR  = pathlib.Path(r"/graph")
VIZ_DIR  = OUT_DIR / "viz"
VIZ_DIR.mkdir(parents=True, exist_ok=True)

if not MSE_PATH.exists():
    raise FileNotFoundError(f"找不到 {MSE_PATH} ，请先运行 cc2_mse_extractor.py 生成 mse_cases.jsonl")

# ---------- 读取与扁平化（含 MSE + MCE） ----------
rows_mse, rows_mce = [], []
with MSE_PATH.open("r", encoding="utf-8") as f:
    for line in f:
        case = json.loads(line)
        blue = case.get("blue_action", "")
        blue_kind = blue.split()[0] if blue else None
        blue_step = case.get("blue_step")
        window    = case.get("window")
        btnorm    = case.get("blue_target_norm")
        if isinstance(btnorm, (list, tuple)) and len(btnorm) == 2:
            blue_ttype, blue_tval = btnorm
        else:
            blue_ttype, blue_tval = None, None

        # MSE
        mse_events = case.get("mse_events", [])
        if not mse_events:
            rows_mse.append({
                "file": case.get("file"), "blue_step": blue_step, "blue_action": blue,
                "blue_kind": blue_kind, "blue_ttype": blue_ttype, "blue_tval": blue_tval,
                "window": window, "ev_type": None, "ev_step": None, "delta_steps": None
            })
        else:
            for ev in mse_events:
                ev_type = ev.get("type"); ev_step = ev.get("step")
                delta   = blue_step - ev_step if (blue_step is not None and ev_step is not None) else None
                rows_mse.append({
                    "file": case.get("file"), "blue_step": blue_step, "blue_action": blue,
                    "blue_kind": blue_kind, "blue_ttype": blue_ttype, "blue_tval": blue_tval,
                    "window": window, "ev_type": ev_type, "ev_step": ev_step, "delta_steps": delta
                })

        # MCE
        mce_events = case.get("mce_events", [])
        if not mce_events:
            rows_mce.append({
                "file": case.get("file"), "blue_step": blue_step, "blue_action": blue,
                "blue_kind": blue_kind, "blue_ttype": blue_ttype, "blue_tval": blue_tval,
                "window": window, "ev_type": None, "ev_step": None, "delta_steps": None
            })
        else:
            for ev in mce_events:
                ev_type = ev.get("type"); ev_step = ev.get("step")
                delta   = blue_step - ev_step if (blue_step is not None and ev_step is not None) else None
                rows_mce.append({
                    "file": case.get("file"), "blue_step": blue_step, "blue_action": blue,
                    "blue_kind": blue_kind, "blue_ttype": blue_ttype, "blue_tval": blue_tval,
                    "window": window, "ev_type": ev_type, "ev_step": ev_step, "delta_steps": delta
                })

df_mse = pd.DataFrame(rows_mse)
df_mce = pd.DataFrame(rows_mce)
df_mse.to_csv(OUT_DIR / "mse_flatten.csv", index=False, encoding="utf-8-sig")
df_mce.to_csv(OUT_DIR / "mce_flatten.csv", index=False, encoding="utf-8-sig")

# ---------- 概览统计 ----------
summary = {}
summary["num_blue_actions"]   = int(df_mse["blue_step"].nunique())
summary["num_rows_mse_flat"]  = int(len(df_mse))
summary["num_rows_mce_flat"]  = int(len(df_mce))
summary["num_cases_with_mse"] = int(df_mse.groupby(["blue_step","blue_action"])["ev_type"].count().gt(0).sum())
summary["num_cases_no_mse"]   = int(df_mse.groupby(["blue_step","blue_action"])["ev_type"].count().eq(0).sum())
summary["num_cases_with_mce"] = int(df_mce.groupby(["blue_step","blue_action"])["ev_type"].count().gt(0).sum())
summary["num_cases_no_mce"]   = int(df_mce.groupby(["blue_step","blue_action"])["ev_type"].count().eq(0).sum())

# MSE/MCE 大小分布
mse_size = (df_mse.groupby(["blue_step","blue_action"])
            .agg(mse_size=("ev_type", lambda s: int(s.notna().sum())))
            .reset_index())
mse_size["blue_kind"] = mse_size["blue_action"].str.split().str[0]
mse_size.to_csv(OUT_DIR / "mse_size_by_blue.csv", index=False, encoding="utf-8-sig")

mce_size = (df_mce.groupby(["blue_step","blue_action"])
            .agg(mce_size=("ev_type", lambda s: int(s.notna().sum())))
            .reset_index())
mce_size["blue_kind"] = mce_size["blue_action"].str.split().str[0]
mce_size.to_csv(OUT_DIR / "mce_size_by_blue.csv", index=False, encoding="utf-8-sig")

# 证据类型分布
ev_type_counts_all_mse = df_mse["ev_type"].value_counts(dropna=True).rename_axis("ev_type").reset_index(name="count")
ev_type_counts_kind_mse = (df_mse.dropna(subset=["ev_type"])
                           .groupby(["blue_kind","ev_type"])
                           .size().reset_index(name="count"))

ev_type_counts_all_mce = df_mce["ev_type"].value_counts(dropna=True).rename_axis("ev_type").reset_index(name="count")
ev_type_counts_kind_mce = (df_mce.dropna(subset=["ev_type"])
                           .groupby(["blue_kind","ev_type"])
                           .size().reset_index(name="count"))

# delta 分布
delta_series_mse = df_mse["delta_steps"].dropna().astype(int)
delta_by_kind_mse = (df_mse.dropna(subset=["delta_steps"])
                     .assign(delta_steps=lambda x: x["delta_steps"].astype(int))
                     .groupby("blue_kind")["delta_steps"])

delta_series_mce = df_mce["delta_steps"].dropna().astype(int)
delta_by_kind_mce = (df_mce.dropna(subset=["delta_steps"])
                     .assign(delta_steps=lambda x: x["delta_steps"].astype(int))
                     .groupby("blue_kind")["delta_steps"])

# Top 目标（按 MSE）
if df_mse.empty:
    raise RuntimeError("mse_flatten 为空：请先确认 mse_cases.jsonl 是否有内容。")
top_targets = (
    df_mse.assign(target=df_mse["blue_ttype"].fillna("") + ":" + df_mse["blue_tval"].fillna(""))
         .query("target != ':'")
         .groupby("target").size()
         .sort_values(ascending=False)
         .head(15)
         .reset_index(name="count")
)

# 保存汇总
with (OUT_DIR / "mse_stats_summary.txt").open("w", encoding="utf-8") as w:
    for k, v in summary.items():
        w.write(f"{k}: {v}\n")

ev_type_counts_all_mse.to_csv(OUT_DIR / "mse_evtype_counts_all.csv", index=False, encoding="utf-8-sig")
ev_type_counts_kind_mse.to_csv(OUT_DIR / "mse_evtype_counts_by_kind.csv", index=False, encoding="utf-8-sig")
ev_type_counts_all_mce.to_csv(OUT_DIR / "mce_evtype_counts_all.csv", index=False, encoding="utf-8-sig")
ev_type_counts_kind_mce.to_csv(OUT_DIR / "mce_evtype_counts_by_kind.csv", index=False, encoding="utf-8-sig")
top_targets.to_csv(OUT_DIR / "mse_top_targets.csv", index=False, encoding="utf-8-sig")

# ---------- 可视化 ----------
# MSE
plt.figure()
ev_type_counts_all_mse.set_index("ev_type")["count"].plot(kind="bar", title="Evidence Type Counts (MSE, All)")
plt.xlabel("ev_type"); plt.ylabel("count"); plt.tight_layout()
plt.savefig(VIZ_DIR / "ev_type_counts_all_mse.png", dpi=160)

for kind, sub in ev_type_counts_kind_mse.groupby("blue_kind"):
    plt.figure()
    sub.set_index("ev_type")["count"].plot(kind="bar", title=f"Evidence Type Counts (MSE, {kind})")
    plt.xlabel("ev_type"); plt.ylabel("count"); plt.tight_layout()
    plt.savefig(VIZ_DIR / f"ev_type_counts_mse_{kind}.png", dpi=160)

plt.figure()
mse_size["mse_size"].value_counts().sort_index().plot(kind="bar", title="MSE Size Distribution")
plt.xlabel("mse_size (events per Blue action)"); plt.ylabel("count"); plt.tight_layout()
plt.savefig(VIZ_DIR / "mse_size_distribution.png", dpi=160)

plt.figure()
delta_series_mse.plot(kind="hist", bins=range(1, max(delta_series_mse.max(), 2)+2), title="Delta (MSE) Blue - Evidence")
plt.xlabel("delta_steps"); plt.ylabel("frequency"); plt.tight_layout()
plt.savefig(VIZ_DIR / "delta_hist_all_mse.png", dpi=160)

for kind, g in delta_by_kind_mse:
    plt.figure()
    g.plot(kind="hist", bins=range(1, max(g.max(), 2)+2), title=f"Delta Histogram (MSE, {kind})")
    plt.xlabel("delta_steps"); plt.ylabel("frequency"); plt.tight_layout()
    plt.savefig(VIZ_DIR / f"delta_hist_mse_{kind}.png", dpi=160)

# MCE
plt.figure()
ev_type_counts_all_mce.set_index("ev_type")["count"].plot(kind="bar", title="Evidence Type Counts (MCE, All)")
plt.xlabel("ev_type"); plt.ylabel("count"); plt.tight_layout()
plt.savefig(VIZ_DIR / "ev_type_counts_all_mce.png", dpi=160)

for kind, sub in ev_type_counts_kind_mce.groupby("blue_kind"):
    plt.figure()
    sub.set_index("ev_type")["count"].plot(kind="bar", title=f"Evidence Type Counts (MCE, {kind})")
    plt.xlabel("ev_type"); plt.ylabel("count"); plt.tight_layout()
    plt.savefig(VIZ_DIR / f"ev_type_counts_mce_{kind}.png", dpi=160)

plt.figure()
mce_size["mce_size"].value_counts().sort_index().plot(kind="bar", title="MCE Size Distribution")
plt.xlabel("mce_size (events per Blue action)"); plt.ylabel("count"); plt.tight_layout()
plt.savefig(VIZ_DIR / "mce_size_distribution.png", dpi=160)

plt.figure()
if len(delta_series_mce) > 0:
    delta_series_mce.plot(kind="hist", bins=range(1, max(delta_series_mce.max(), 2)+2), title="Delta (MCE) Blue - Evidence")
    plt.xlabel("delta_steps"); plt.ylabel("frequency"); plt.tight_layout()
    plt.savefig(VIZ_DIR / "delta_hist_all_mce.png", dpi=160)

for kind, g in delta_by_kind_mce:
    plt.figure()
    g.plot(kind="hist", bins=range(1, max(g.max(), 2)+2), title=f"Delta Histogram (MCE, {kind})")
    plt.xlabel("delta_steps"); plt.ylabel("frequency"); plt.tight_layout()
    plt.savefig(VIZ_DIR / f"delta_hist_mce_{kind}.png", dpi=160)

# Top Targets（沿用 MSE）
plt.figure()
top_targets.set_index("target")["count"].plot(kind="barh", title="Top Targets (by Blue actions, MSE)")
plt.xlabel("count"); plt.ylabel("target"); plt.tight_layout()
plt.savefig(VIZ_DIR / "top_targets.png", dpi=160)

print("✅ 统计完成：")
print(f"  - 扁平数据：{OUT_DIR / 'mse_flatten.csv'} / {OUT_DIR / 'mce_flatten.csv'}")
print(f"  - MSE规模：{OUT_DIR / 'mse_size_by_blue.csv'}；MCE规模：{OUT_DIR / 'mce_size_by_blue.csv'}")
print(f"  - 证据分布（MSE/MCE）：{OUT_DIR / 'mse_evtype_counts_all.csv'} / {OUT_DIR / 'mce_evtype_counts_all.csv'}")
print(f"  - Top 目标：{OUT_DIR / 'mse_top_targets.csv'}")
print(f"  - 可视化图片在：{VIZ_DIR}")
