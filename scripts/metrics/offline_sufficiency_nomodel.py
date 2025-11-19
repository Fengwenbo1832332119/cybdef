
# -*- coding: utf-8 -*-
"""
Offline sufficiency (no model): minimal subset needed to retain the same priority-top evidence type
under a recency-prioritized selector.
"""
import os, sys, json, pathlib, itertools
import pandas as pd
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

THIS = pathlib.Path(__file__).resolve()
PKG  = THIS.parents[1]
if str(PKG) not in sys.path:
    sys.path.insert(0, str(PKG))

from common.paths import GRAPH_DIR, REPORTS_DIR, ensure_dirs

MSE_PATH = GRAPH_DIR / "mse_cases.jsonl"
OUT_DIR  = REPORTS_DIR / "sufficiency_offline_nomodel"
OUT_DIR.mkdir(parents=True, exist_ok=True)

PRIORITY={"PrivilegeEscalate":4,"ExploitRemoteService":3,"DiscoverNetworkServices":2,"DiscoverRemoteSystems":2}

def select_minimal(cands):
    if not cands:
        return []
    kept={}
    for c in sorted(cands, key=lambda x: x.get("step",0), reverse=True):
        t = c.get("type","")
        if t not in kept:
            kept[t] = c
    return sorted(kept.values(), key=lambda x: PRIORITY.get(x.get("type",""),0), reverse=True)

def main():
    ensure_dirs()
    rows=[]
    with MSE_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            case = json.loads(line)
            mse = case.get("mse_events",[]) or []
            base = select_minimal(mse)
            base_top = base[0]["type"] if base else None
            found = 0
            if mse and base_top:
                for r in range(1, len(mse)+1):
                    ok=False
                    for comb in itertools.combinations(mse, r):
                        out=select_minimal(list(comb))
                        if out and out[0]["type"]==base_top:
                            found=r; ok=True; break
                    if ok: break
            rows.append({
                "file": pathlib.Path(case.get("file","")).stem,
                "episode": case.get("episode"),
                "blue_step": case.get("blue_step"),
                "blue_action": case.get("blue_action"),
                "min_set": int(found)
            })

    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR/"sufficiency_minset_nomodel.csv", index=False, encoding="utf-8-sig")
    s = df["min_set"].replace(0, np.nan).dropna()
    stats = {
        "count": int(s.shape[0]),
        "mean": float(s.mean()) if not s.empty else 0.0,
        "median": float(s.median()) if not s.empty else 0.0,
        "p90": float(s.quantile(0.9)) if not s.empty else 0.0,
    }
    with (OUT_DIR/"summary.txt").open("w", encoding="utf-8") as w:
        for k,v in stats.items():
            w.write(f"{k}: {v}\n")

    plt.figure(figsize=(8,4))
    df["min_set"].value_counts().sort_index().plot(kind="bar")
    plt.title("Minimal sufficient evidence size (no model)")
    plt.xlabel("k"); plt.ylabel("count"); plt.tight_layout()
    plt.savefig(OUT_DIR/"sufficiency_hist.png", dpi=160)

    print("✅ Saved:", OUT_DIR)

if __name__ == "__main__":
    main()
