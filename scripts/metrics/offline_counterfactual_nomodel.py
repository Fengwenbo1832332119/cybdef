# -*- coding: utf-8 -*-
"""
offline_counterfactual_v4.py
统一的“语义反事实 + 充分性检验”脚本（离线、无模型版）

输入：GRAPH_DIR/mse_cases.jsonl
输出：reports/counterfactual_v4/*

功能：
1) 语义反事实（counterfactual）
   - 类型反事实：upgrade/downgrade/random_cross/freq_weighted_cross
   - 时序反事实：same_type_step_shift（±k 步）
   - 目标反事实：swap_target（将最近一条事件的目标换成同 case 中其他目标）
2) 充分性检验（sufficiency）
   - 仅保留 related-only 解释证据，检查 rule_score 是否仍支撑原判

注意：
- 依赖 scripts.common.paths / scripts.common.feat_utils（已在你仓库里）
- 与 v3 保持同权重与 recency decay 口径
"""

import os, sys, json, copy, random, pathlib, argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# 让 python 能找到 scripts.* 包（用相对位置推导）
THIS = pathlib.Path(__file__).resolve()
PKG  = THIS.parents[1]  # .../scripts
if str(PKG) not in sys.path:
    sys.path.insert(0, str(PKG))

from scripts.common.paths import GRAPH_DIR, REPORTS_DIR, ensure_dirs
from scripts.common.feat_utils import build_feats_related  # 我们只用它来做“related-only”过滤

# ----------------- 常量与权重口径 -----------------
OUT_DIR = REPORTS_DIR / "counterfactual_v4"
MSE_PATH = GRAPH_DIR / "mse_cases.jsonl"

DEFAULT_SEED = 1234
DEFAULT_REC_DECAY = 0.98

ALL_TYPES = ["PrivilegeEscalate","ExploitRemoteService","DiscoverNetworkServices","DiscoverRemoteSystems"]
WEIGHT_DEFAULT = {
    "PrivilegeEscalate": 4.0,
    "ExploitRemoteService": 3.0,
    "DiscoverNetworkServices": 1.5,
    "DiscoverRemoteSystems": 1.2,
    "Sleep": 0.2,  # 噪声/无关证据
}
TYPE_PRIORITY = {"PrivilegeEscalate":4,"ExploitRemoteService":3,"DiscoverNetworkServices":2,"DiscoverRemoteSystems":2}

# 类型的“任务阶段”语义（便于做升级/降级）
PHASE = {
    "DiscoverRemoteSystems":      "Recon",
    "DiscoverNetworkServices":    "Recon",
    "ExploitRemoteService":       "Access",
    "PrivilegeEscalate":          "Control",
}
UPGRADE_MAP = {  # 语义升级（Recon→Access→Control）
    "DiscoverRemoteSystems":   "ExploitRemoteService",
    "DiscoverNetworkServices": "ExploitRemoteService",
    "ExploitRemoteService":    "PrivilegeEscalate",
}
DOWNGRADE_MAP = {  # 语义降级
    "PrivilegeEscalate":       "DiscoverNetworkServices",
    "ExploitRemoteService":    "DiscoverNetworkServices",
    "DiscoverNetworkServices": "DiscoverRemoteSystems",
}

# ----------------- 工具函数 -----------------
def deep(x): return copy.deepcopy(x)
def rng(seed): random.seed(seed); np.random.seed(seed)

def rule_score(evs, blue_step, weight, rec_decay):
    """与 v3 一致的可解释打分：按类型权重 + 距离蓝方步数的衰减"""
    if not evs: return 0.0
    if blue_step is None:
        blue_step = max([e.get("step", 0) or 0 for e in evs] + [0])
    s = 0.0
    for e in evs:
        et = e.get("type","")
        st = e.get("step", None)
        w = weight.get(et, 0.5)
        decay = (rec_decay ** max(0, int(blue_step) - int(st))) if st is not None else 1.0
        s += w * decay
    return float(s)

def latest_idx_of_type(evs, t):
    idx=-1; best=-10**9
    for i,e in enumerate(evs):
        if e.get("type")==t:
            st = e.get("step"); st=-10**6 if st is None else int(st)
            if st>=best: best=st; idx=i
    return idx

def all_targets(evs):
    """容错地收集 'target' 字段（如果有）"""
    tgs=[]
    for e in evs or []:
        tg = e.get("target") or e.get("ip") or e.get("hostname")
        if tg is not None: tgs.append(str(tg))
    return list(dict.fromkeys(tgs))  # 去重保序

def global_type_freq(cases):
    cnt = {t:0 for t in ALL_TYPES}
    for c in cases:
        for key in ["mse_events","mce_events"]:
            for e in (c.get(key) or []):
                t = e.get("type")
                if t in cnt: cnt[t]+=1
    total = sum(cnt.values()) or 1
    return {t: cnt[t]/total for t in ALL_TYPES}

# ----------------- 反事实：类型/时序/目标 -----------------
def cf_type_change(evs, idx, mode, freq_map=None):
    """将 evs[idx] 的 type 做‘升级/降级/随机跨类/按频次跨类’"""
    evs = deep(evs)
    if not (0 <= idx < len(evs)): return evs
    cur = evs[idx].get("type","")
    if mode == "upgrade":
        evs[idx]["type"] = UPGRADE_MAP.get(cur, cur)
    elif mode == "downgrade":
        evs[idx]["type"] = DOWNGRADE_MAP.get(cur, cur)
    elif mode == "random_cross":
        cands = [t for t in ALL_TYPES if t != cur]
        if cands: evs[idx]["type"] = random.choice(cands)
    elif mode == "freq_weighted_cross":
        items = [(t, freq_map.get(t,0.0)) for t in ALL_TYPES if t != cur]
        if items:
            types, probs = zip(*items)
            probs = np.array(probs, dtype=float); probs = probs/probs.sum() if probs.sum()>0 else np.ones_like(probs)/len(probs)
            evs[idx]["type"] = str(np.random.choice(types, p=probs))
    return evs

def cf_time_shift(evs, idx, shift):
    """保持 type 不变，step ± shift"""
    evs = deep(evs)
    if 0 <= idx < len(evs):
        st = int(evs[idx].get("step",0) or 0)
        evs[idx]["step"] = st + int(shift)
    return evs

def cf_target_swap(evs, idx):
    """将 evs[idx] 的 target 替换成同 case 内的其他目标（如果有）"""
    evs = deep(evs)
    if not (0 <= idx < len(evs)): return evs
    tgs = all_targets(evs)
    if len(tgs) < 2: return evs
    cur = evs[idx].get("target") or evs[idx].get("ip") or evs[idx].get("hostname")
    cands = [x for x in tgs if str(x) != str(cur)]
    if cands:
        evs[idx]["target"] = str(random.choice(cands))
    return evs

def run_counterfactual(cases, out_dir, weight, rec_decay, cf_modes, time_shifts, do_target):
    """
    cf_modes: ['upgrade','downgrade','random_cross','freq_weighted_cross'] 的子集
    time_shifts: e.g. [-2,-1,1,2]
    do_target: bool
    """
    freq_map = global_type_freq(cases)
    rows = []

    for c in cases:
        blue_step = c.get("blue_step")
        file_stem = pathlib.Path(c.get("file","")).stem
        for mode_name in ["MSE","MCE"]:
            base = copy.deepcopy(c.get("mse_events") if mode_name=="MSE" else c.get("mce_events"))
            if not base: continue
            s0 = rule_score(base, blue_step, weight, rec_decay)

            # 逐类型拿“最近一条”做反事实
            for t in ALL_TYPES:
                idx = latest_idx_of_type(base, t)
                if idx < 0:
                    rows.append({"mode":mode_name,"file":file_stem,"type":t,"op":"none","delta":0.0})
                    continue

                # 类型反事实
                for op in cf_modes:
                    evs = cf_type_change(base, idx, op, freq_map=freq_map)
                    s1 = rule_score(evs, blue_step, weight, rec_decay)
                    rows.append({"mode":mode_name,"file":file_stem,"type":t,"op":f"type_{op}","delta":s0-s1})

                # 时序反事实
                for sh in time_shifts:
                    evs = cf_time_shift(base, idx, shift=sh)
                    s1 = rule_score(evs, blue_step, weight, rec_decay)
                    rows.append({"mode":mode_name,"file":file_stem,"type":t,"op":f"time_shift_{sh:+d}","delta":s0-s1})

                # 目标反事实
                if do_target:
                    evs = cf_target_swap(base, idx)
                    s1 = rule_score(evs, blue_step, weight, rec_decay)
                    rows.append({"mode":mode_name,"file":file_stem,"type":t,"op":"target_swap","delta":s0-s1})

    df = pd.DataFrame(rows)
    df.to_csv(out_dir/"counterfactual_results.csv", index=False, encoding="utf-8-sig")

    # 聚合可视化：按 op 汇总
    for mode_name in ["MSE","MCE"]:
        sub = df[df["mode"]==mode_name]
        if sub.empty: continue
        g = sub.groupby("op")["delta"].agg(["mean","std"]).sort_values("mean", ascending=False)
        plt.figure(figsize=(10,4))
        plt.bar(g.index, g["mean"], yerr=g["std"], capsize=4)
        plt.xticks(rotation=65, ha="right")
        plt.ylabel("Δ score (mean±std)")
        plt.title(f"Counterfactual Δ — {mode_name}")
        plt.tight_layout()
        plt.savefig(out_dir/f"counterfactual_{mode_name}.png", dpi=160); plt.close()

    return df

# ----------------- 充分性检验 -----------------
def run_sufficiency(cases, out_dir, weight, rec_decay):
    """
    仅保留 related-only 证据（feat_utils.build_feats_related 的筛选口径）对应的事件子集，
    观察打分是否仍能维持原判：输出“保真度曲线”
    """
    rows=[]
    for c in cases:
        blue_step = c.get("blue_step")
        file_stem = pathlib.Path(c.get("file","")).stem

        for mode_name in ["MSE","MCE"]:
            base = copy.deepcopy(c.get("mse_events") if mode_name=="MSE" else c.get("mce_events"))
            if not base:
                rows.append({"mode":mode_name,"file":file_stem,"orig":0.0,"kept":0.0,"delta":0.0})
                continue

            s0 = rule_score(base, blue_step, weight, rec_decay)

            # related-only：利用 feat_utils 的口径做“相关证据过滤”
            feats = build_feats_related({"mse_events":base, "mce_events":[], "blue_step":blue_step}, disc_cap=2)
            # 根据 feats 里保留下来的统计还原出需要的类型集合
            keep_types=[]
            if feats.get("mse_priv_any",0)>0: keep_types.append("PrivilegeEscalate")
            if feats.get("mse_expl_any",0)>0: keep_types.append("ExploitRemoteService")
            if feats.get("mse_disc",0)>0:     keep_types += ["DiscoverNetworkServices","DiscoverRemoteSystems"]

            kept=[]
            for e in base:
                if e.get("type") in set(keep_types):
                    kept.append(e)

            s1 = rule_score(kept, blue_step, weight, rec_decay)
            rows.append({"mode":mode_name,"file":file_stem,"orig":s0,"kept":s1,"delta":s0-s1,"kept_ratio": (len(kept)/(len(base) or 1))})

    df = pd.DataFrame(rows)
    df.to_csv(out_dir/"sufficiency_results.csv", index=False, encoding="utf-8-sig")

    # 可视化：orig vs kept 的散点 + 45° 线
    for mode_name in ["MSE","MCE"]:
        sub = df[df["mode"]==mode_name]
        if sub.empty: continue
        plt.figure(figsize=(5,5))
        plt.scatter(sub["orig"], sub["kept"], s=12)
        lim = [0, max(1e-6, sub[["orig","kept"]].values.max())*1.05]
        plt.plot(lim, lim, 'k--', lw=1)
        plt.xlabel("score (original)"); plt.ylabel("score (kept related-only)")
        plt.title(f"Sufficiency — {mode_name}")
        plt.tight_layout(); plt.savefig(out_dir/f"sufficiency_{mode_name}.png", dpi=160); plt.close()

    return df

# ----------------- CLI -----------------
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=int(os.environ.get("CYBDEF_SEED", DEFAULT_SEED)))
    ap.add_argument("--rec-decay", type=float, default=float(os.environ.get("CYBDEF_REC_DECAY", DEFAULT_REC_DECAY)))
    ap.add_argument("--do", choices=["counterfactual","sufficiency","all"], default="all")

    # counterfactual 相关参数
    ap.add_argument("--cf-modes", type=str,
                    default=os.environ.get("CYBDEF_CF_MODES","upgrade,downgrade,random_cross,freq_weighted_cross"),
                    help="逗号分隔：upgrade,downgrade,random_cross,freq_weighted_cross 的子集")
    ap.add_argument("--time-shifts", type=str, default=os.environ.get("CYBDEF_TIME_SHIFTS","-2,-1,1,2"),
                    help="时序反事实的 step 偏移列表，如 -2,-1,1,2")
    ap.add_argument("--cf-target", action="store_true", help="是否启用 target_swap 反事实")

    return ap.parse_args()

def _parse_list_of_ints(s: str):
    s = s.strip()
    if s.startswith("[") and s.endswith("]"): s = s[1:-1]
    out=[]
    for tok in s.split(","):
        tok = tok.strip()
        if tok: out.append(int(tok))
    return out

def main():
    args = parse_args()
    rng(args.seed)
    ensure_dirs()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # 权重口径
    weight = dict(WEIGHT_DEFAULT)
    rec_decay = float(args.rec_decay)

    # 数据加载
    if not MSE_PATH.exists():
        raise FileNotFoundError(f"not found: {MSE_PATH}")
    cases=[]
    with MSE_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            line=line.strip()
            if line:
                cases.append(json.loads(line))

    # 任务分发
    if args.do in ("counterfactual","all"):
        cf_modes = [x.strip() for x in args.cf_modes.split(",") if x.strip()]
        time_shifts = _parse_list_of_ints(args.time_shifts)
        run_counterfactual(cases, OUT_DIR, weight, rec_decay,
                           cf_modes=cf_modes, time_shifts=time_shifts, do_target=args.cf_target)

    if args.do in ("sufficiency","all"):
        run_sufficiency(cases, OUT_DIR, weight, rec_decay)

    with (OUT_DIR/"report.txt").open("w", encoding="utf-8") as w:
        w.write("counterfactual_v4 finished.\n")
        w.write(f"do={args.do}\n")
        w.write(f"recency_decay={rec_decay}\n")
        w.write(f"cf_modes={args.cf_modes}\n")
        w.write(f"time_shifts={args.time_shifts}\n")
        w.write(f"cf_target={args.cf_target}\n")

    print("✅ Done ->", OUT_DIR.resolve())

if __name__ == "__main__":
    main()
