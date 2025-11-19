# -*- coding: utf-8 -*-
"""
离线删证/换证（无模型）— v3 增强版
新增：
1) 跨类型换证策略：
   - same_type_diffstep：同类不同步（保留 type，改变 step）
   - random_cross：随机跨类（均匀从其他类型中选）
   - freq_weighted_cross：按全局类型频次加权抽样跨类
2) 无关证据注入（Sleep）的时序敏感性曲线：
   - 在 blue_step + offsets 的一组相对步数注入噪声，统计 Δ 并画曲线
输出：CSV + 可视化 + 简要报告
"""
import os, sys, json, copy, random, pathlib, itertools, argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---- 项目路径 ----
THIS = pathlib.Path(__file__).resolve()
PKG  = THIS.parents[1]
if str(PKG) not in sys.path:
    sys.path.insert(0, str(PKG))
from scripts.common.paths import GRAPH_DIR, REPORTS_DIR, ensure_dirs

# ---- I/O & 参数默认 ----
MSE_PATH = GRAPH_DIR / "mse_cases.jsonl"
OUT_DIR  = REPORTS_DIR / "intervention_offline_nomodel_v3"

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
PRIORITY = {"PrivilegeEscalate":4,"ExploitRemoteService":3,"DiscoverNetworkServices":2,"DiscoverRemoteSystems":2}

# ------------------ 基础工具 ------------------
def deep(x): return copy.deepcopy(x)
def rng(seed):
    random.seed(seed); np.random.seed(seed)

def rule_score(evs, blue_step, weight, rec_decay):
    if not evs: return 0.0
    if blue_step is None:
        blue_step = max([e.get("step", 0) or 0 for e in evs] + [0])
    s = 0.0
    for e in evs:
        et = e.get("type","")
        w = weight.get(et, 0.5)
        st = e.get("step", None)
        decay = (rec_decay ** max(0, int(blue_step) - int(st))) if st is not None else 1.0
        s += w * decay
    return float(s)

def contrib(e, blue_step, weight, rec_decay):
    et = e.get("type",""); w = weight.get(et, 0.5)
    st = e.get("step", None)
    decay = (rec_decay ** max(0, int(blue_step) - int(st))) if st is not None else 1.0
    return float(w * decay)

def latest_idx_of_type(evs, t):
    idx=-1; best=-10**9
    for i,e in enumerate(evs):
        if e.get("type")==t:
            st = e.get("step"); st=-10**6 if st is None else int(st)
            if st>=best: best=st; idx=i
    return idx

def select_minimal(cands):
    if not cands: return []
    kept={}
    for c in sorted(cands, key=lambda x: x.get("step",0), reverse=True):
        t = c.get("type","")
        if t not in kept:
            kept[t]=c
    return sorted(kept.values(), key=lambda x: PRIORITY.get(x.get("type",""),0), reverse=True)

# ------------------ 数据加载 ------------------
def load_cases(mse_path: pathlib.Path):
    if not mse_path.exists():
        raise FileNotFoundError(f"mse_cases.jsonl not found: {mse_path}")
    cases=[]
    with mse_path.open("r", encoding="utf-8") as f:
        for line in f:
            line=line.strip()
            if line:
                cases.append(json.loads(line))
    return cases

def global_type_freq(cases):
    cnt = {t:0 for t in ALL_TYPES}
    for c in cases:
        for key in ["mse_events","mce_events"]:
            for e in (c.get(key) or []):
                t = e.get("type")
                if t in cnt: cnt[t]+=1
    total = sum(cnt.values()) or 1
    freq = {t: cnt[t]/total for t in ALL_TYPES}
    return freq

# ------------------ 删证曲线（随机/贪心） ------------------
def rand_delete(evs):
    if not evs: return evs, False
    evs = deep(evs)
    i = random.randrange(len(evs))
    del evs[i]
    return evs, True

def greedy_delete(evs, blue_step, weight, rec_decay):
    if not evs: return evs, False
    evs = deep(evs)
    scores=[contrib(e, blue_step, weight, rec_decay) for e in evs]
    i=int(np.argmax(scores))
    del evs[i]
    return evs, True

def run_cumulative_deletion(cases, out_dir, K=5, trials=10, weight=None, rec_decay=DEFAULT_REC_DECAY):
    rows=[]
    for c in cases:
        blue_step = c.get("blue_step")
        for mode in ["MSE","MCE"]:
            base = deep(c.get("mse_events") if mode=="MSE" else c.get("mce_events"))
            if not base:
                for k in range(1, K+1):
                    rows.append({"mode":mode,"k":k,"strategy":"random","delta":0.0})
                    rows.append({"mode":mode,"k":k,"strategy":"greedy","delta":0.0})
                continue
            s0 = rule_score(base, blue_step, weight, rec_decay)
            n=len(base); kmax=min(K,n)
            # random
            for k in range(1, kmax+1):
                vals=[]
                for _ in range(trials):
                    evs = deep(base)
                    for _del in range(k):
                        evs, ok = rand_delete(evs)
                        if not ok: break
                    vals.append(s0 - rule_score(evs, blue_step, weight, rec_decay))
                rows.append({"mode":mode,"k":k,"strategy":"random","delta":float(np.mean(vals))})
            # greedy
            evs = deep(base)
            for k in range(1, kmax+1):
                evs, ok = greedy_delete(evs, blue_step, weight, rec_decay)
                ds = s0 - rule_score(evs, blue_step, weight, rec_decay)
                rows.append({"mode":mode,"k":k,"strategy":"greedy","delta":ds})
    df = pd.DataFrame(rows)
    df.to_csv(out_dir/"cumulative_deletion.csv", index=False, encoding="utf-8-sig")
    # plot
    for mode in ["MSE","MCE"]:
        sub = df[df["mode"]==mode]
        if sub.empty: continue
        plt.figure(figsize=(7,4))
        for strat in ["random","greedy"]:
            s2 = sub[sub["strategy"]==strat].groupby("k")["delta"].mean()
            plt.plot(s2.index, s2.values, marker="o", label=strat)
        plt.title(f"Cumulative Deletion Curve — {mode}")
        plt.xlabel("k deletions"); plt.ylabel("Δ score (mean)"); plt.legend(); plt.tight_layout()
        plt.savefig(out_dir/f"cumu_del_curve_{mode}.png", dpi=160); plt.close()
    return df

# ------------------ 跨类型换证策略 ------------------
def replace_same_type_diffstep(evs, idx, jitter=1):
    """保持 type 不变，仅把 step±jitter（模拟同类不同步替换）"""
    evs = deep(evs)
    if 0<=idx<len(evs):
        st = evs[idx].get("step", 0) or 0
        evs[idx]["step"] = int(st) + int(jitter)
    return evs

def replace_random_cross(evs, idx, cur_type):
    """把 evs[idx] 的 type 换成“其他类型”中的随机一个（均匀）"""
    evs = deep(evs)
    if 0<=idx<len(evs):
        candidates = [t for t in ALL_TYPES if t != cur_type]
        evs[idx]["type"] = random.choice(candidates) if candidates else cur_type
    return evs

def replace_freq_weighted_cross(evs, idx, cur_type, freq_map):
    """把 type 换为按全局频次加权抽样的其他类型"""
    evs = deep(evs)
    if 0<=idx<len(evs):
        items = [(t, freq_map.get(t,0.0)) for t in ALL_TYPES if t != cur_type]
        if not items:
            return evs
        types, probs = zip(*items)
        probs = np.array(probs, dtype=float)
        probs = probs/ probs.sum() if probs.sum() > 0 else np.ones_like(probs)/len(probs)
        evs[idx]["type"] = str(np.random.choice(types, p=probs))
    return evs

def run_cross_type_swaps(cases, out_dir, strategy="random_cross", jitter=1, weight=None, rec_decay=DEFAULT_REC_DECAY):
    """对各类型最近一条执行“替换”，统计 Δ"""
    rows=[]
    freq_map = global_type_freq(cases)
    for c in cases:
        blue_step = c.get("blue_step")
        for mode in ["MSE","MCE"]:
            base = deep(c.get("mse_events") if mode=="MSE" else c.get("mce_events"))
            if not base:
                continue
            s0 = rule_score(base, blue_step, weight, rec_decay)
            for t in ALL_TYPES:
                i = latest_idx_of_type(base, t)
                if i < 0:
                    rows.append({"mode":mode,"type":t,"delta":0.0,"strategy":strategy,"file":pathlib.Path(c.get("file","")).stem})
                    continue
                if strategy == "same_type_diffstep":
                    evs = replace_same_type_diffstep(base, i, jitter=jitter)
                elif strategy == "freq_weighted_cross":
                    evs = replace_freq_weighted_cross(base, i, cur_type=t, freq_map=freq_map)
                else:  # random_cross
                    evs = replace_random_cross(base, i, cur_type=t)
                s1 = rule_score(evs, blue_step, weight, rec_decay)
                rows.append({
                    "mode":mode,"type":t,"delta":s0-s1,"strategy":strategy,
                    "file":pathlib.Path(c.get("file","")).stem
                })
    df = pd.DataFrame(rows)
    df.to_csv(out_dir/f"cross_type_{strategy}.csv", index=False, encoding="utf-8-sig")
    # plot
    for mode in ["MSE","MCE"]:
        sub = df[df["mode"]==mode]
        if sub.empty: continue
        g = sub.groupby("type")["delta"].agg(["mean","std"]).reindex(ALL_TYPES)
        plt.figure(figsize=(8,4))
        plt.bar(g.index, g["mean"], yerr=g["std"], capsize=5)
        plt.title(f"Cross-Type Replacement — {strategy} — {mode}")
        plt.ylabel("Δ score (mean±std)"); plt.xticks(rotation=15); plt.tight_layout()
        plt.savefig(out_dir/f"cross_type_{strategy}_{mode}.png", dpi=160); plt.close()
    return df

# ------------------ 无关证据时序敏感性 ------------------
def inject_unrelated_at(evs, step):
    evs = deep(evs)
    evs.append({"type":"Sleep", "step": int(step)})
    return evs

def run_noise_timing_sensitivity(cases, out_dir, offsets, weight=None, rec_decay=DEFAULT_REC_DECAY):
    """在 blue_step + offset 注入 Sleep，统计 Δ 并画曲线"""
    rows=[]
    for c in cases:
        blue_step = c.get("blue_step")
        for mode in ["MSE","MCE"]:
            base = deep(c.get("mse_events") if mode=="MSE" else c.get("mce_events"))
            if not base:
                continue
            s0 = rule_score(base, blue_step, weight, rec_decay)
            ref = int(blue_step) if isinstance(blue_step, int) else max([e.get("step",0) or 0 for e in base]+[0])
            for off in offsets:
                evs = inject_unrelated_at(base, ref + int(off))
                s1 = rule_score(evs, blue_step, weight, rec_decay)
                rows.append({
                    "mode":mode, "offset": int(off),
                    "delta": s0 - s1,
                    "file":pathlib.Path(c.get("file","")).stem
                })
    df = pd.DataFrame(rows)
    df.to_csv(out_dir/"noise_timing_sensitivity.csv", index=False, encoding="utf-8-sig")
    # plot: 每个模式画一条均值曲线
    for mode in ["MSE","MCE"]:
        sub = df[df["mode"]==mode]
        if sub.empty: continue
        g = sub.groupby("offset")["delta"].mean().sort_index()
        plt.figure(figsize=(8,4))
        plt.plot(g.index, g.values, marker="o")
        plt.axvline(0, ls="--", lw=1, color="k")
        plt.title(f"Noise Timing Sensitivity — {mode}")
        plt.xlabel("offset (relative to blue_step)"); plt.ylabel("Δ score (mean)")
        plt.tight_layout(); plt.savefig(out_dir/f"noise_timing_{mode}.png", dpi=160); plt.close()
    return df

# ------------------ 主流程 ------------------
def parse_offsets(s: str):
    """
    将形如 "-5,-3,0,1,3" 或 "[ -5, -3, 0, 1, 3 ]" 的字符串解析为整数列表
    """
    s = s.strip()
    s = s[1:-1] if (s.startswith("[") and s.endswith("]")) else s
    vals=[]
    for tok in s.split(","):
        tok=tok.strip()
        if tok:
            vals.append(int(tok))
    return vals

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=int(os.environ.get("CYBDEF_SEED", DEFAULT_SEED)))
    ap.add_argument("--rec-decay", type=float, default=float(os.environ.get("CYBDEF_REC_DECAY", DEFAULT_REC_DECAY)))
    ap.add_argument("--swap-strategy", choices=["same_type_diffstep","random_cross","freq_weighted_cross"], default="random_cross")
    ap.add_argument("--jitter", type=int, default=1, help="same_type_diffstep 的 step 偏移量")
    ap.add_argument("--cumu-k", type=int, default=int(os.environ.get("CYBDEF_CUMU_K","5")))
    ap.add_argument("--cumu-trials", type=int, default=int(os.environ.get("CYBDEF_CUMU_TRIALS","10")))
    ap.add_argument("--noise-offsets", type=str, default=os.environ.get("CYBDEF_NOISE_OFFSETS","-5,-3,-1,0,1,3,5"),
                    help='相对 blue_step 的偏移列表，例如 "-5,-3,0,1,3" 或 "[ -5, -3, 0, 1, 3 ]"')
    args = ap.parse_args()

    ensure_dirs()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rng(args.seed)

    # 权重（可按需改为从环境变量读 JSON）
    weight = dict(WEIGHT_DEFAULT)
    rec_decay = args.rec_decay

    # 数据
    cases = load_cases(MSE_PATH)

    # 1) 累积删证曲线
    run_cumulative_deletion(cases, OUT_DIR, K=args.cumu_k, trials=args.cumu_trials, weight=weight, rec_decay=rec_decay)

    # 2) 跨类型换证
    run_cross_type_swaps(cases, OUT_DIR, strategy=args.swap_strategy, jitter=args.jitter, weight=weight, rec_decay=rec_decay)

    # 3) 无关证据时序敏感性
    offsets = parse_offsets(args.noise_offsets)
    run_noise_timing_sensitivity(cases, OUT_DIR, offsets=offsets, weight=weight, rec_decay=rec_decay)

    # 简要报告
    with (OUT_DIR/"report_v3.md").open("w", encoding="utf-8") as w:
        w.write("# Offline Intervention v3 (No-Model)\n")
        w.write(f"- swap-strategy: {args.swap_strategy}, jitter={args.jitter}\n")
        w.write(f"- cumulative K={args.cumu_k}, trials={args.cumu_trials}\n")
        w.write(f"- noise offsets={offsets}\n")
        w.write(f"- recency decay={rec_decay}\n")
    print("✅ Done:", OUT_DIR)

if __name__ == "__main__":
    main()
