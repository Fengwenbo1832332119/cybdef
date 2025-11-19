# -*- coding: utf-8 -*-
"""
离线删证/换证（无模型）— v3 增强版（related-only + 发现类去重/限幅）
"""
import os, sys, json, copy, random, pathlib, argparse
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
DEFAULT_REC_DECAY = 0.90
ALL_TYPES = ["PrivilegeEscalate","ExploitRemoteService","DiscoverNetworkServices","DiscoverRemoteSystems"]
DISC_TYPES = {"DiscoverRemoteSystems","DiscoverNetworkServices"}

WEIGHT_DEFAULT = {
    "PrivilegeEscalate": 4.0,
    "ExploitRemoteService": 3.0,
    "DiscoverNetworkServices": 1.2,
    "DiscoverRemoteSystems": 1.0,
    "Sleep": 0.2,  # 噪声/无关证据（仅在噪声敏感性实验中可选计入）
}
PRIORITY = {"PrivilegeEscalate":4,"ExploitRemoteService":3,"DiscoverNetworkServices":2,"DiscoverRemoteSystems":2}

# ------------------ 工具 ------------------
def deep(x): return copy.deepcopy(x)
def rng(seed): random.seed(seed); np.random.seed(seed)

def _latest_by_type_target(evs):
    """同(type,target)仅保留最近一步"""
    latest = {}
    for e in evs or []:
        k = (e.get("type"), e.get("target"))
        st = e.get("step", -10**9) or -10**9
        if (k not in latest) or (st > (latest[k].get("step", -10**9) or -10**9)):
            latest[k] = e
    return list(latest.values())

def preprocess_events(evs, disc_cap=2, allow_unrelated=False):
    """
    仅相关证据 + 发现类去重/限幅；
    allow_unrelated=True 时，会把 is_related=False 的事件也纳入（仅供噪声曲线实验）。
    """
    evs = evs or []
    if allow_unrelated:
        base = evs
    else:
        base = [e for e in evs if e.get("is_related")]

    # 同(type,target)只留最近一步
    base = _latest_by_type_target(base)

    # 发现类限幅：只保留最近 disc_cap 条
    disc = [e for e in base if e.get("type") in DISC_TYPES]
    non_disc = [e for e in base if e.get("type") not in DISC_TYPES]
    disc_sorted = sorted(disc, key=lambda x: x.get("step", -10**9) or -10**9, reverse=True)[:int(disc_cap)]
    return non_disc + disc_sorted

def rule_score(evs, blue_step, weight, rec_decay, disc_cap=2, allow_unrelated=False):
    """对事件打分前做预处理：related-only + 发现类去重/限幅"""
    clean = preprocess_events(evs, disc_cap=disc_cap, allow_unrelated=allow_unrelated)
    if not clean: return 0.0
    if blue_step is None:
        blue_step = max([e.get("step", 0) or 0 for e in clean] + [0])
    s = 0.0
    for e in clean:
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
                if t in cnt and e.get("is_related", True):  # 频次也仅统计相关
                    cnt[t]+=1
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
    # 贪心要在“清洗后”空间上做，避免把被限幅剔除的条目算进来
    evs = preprocess_events(evs)
    if not evs: return evs, False
    scores=[contrib(e, blue_step, weight, rec_decay) for e in evs]
    i=int(np.argmax(scores))
    del evs[i]
    return evs, True

def run_cumulative_deletion(cases, out_dir, K=5, trials=10, weight=None, rec_decay=DEFAULT_REC_DECAY, disc_cap=2):
    rows=[]
    for c in cases:
        blue_step = c.get("blue_step")
        for mode in ["MSE","MCE"]:
            raw = c.get("mse_events") if mode=="MSE" else c.get("mce_events")
            base = preprocess_events(raw, disc_cap=disc_cap)
            if not base:
                for k in range(1, K+1):
                    rows.append({"mode":mode,"k":k,"strategy":"random","delta":0.0})
                    rows.append({"mode":mode,"k":k,"strategy":"greedy","delta":0.0})
                continue
            s0 = rule_score(base, blue_step, weight, rec_decay, disc_cap=disc_cap)
            n=len(base); kmax=min(K,n)
            # random
            for k in range(1, kmax+1):
                vals=[]
                for _ in range(trials):
                    evs = deep(base)
                    for _del in range(k):
                        evs, ok = rand_delete(evs)
                        if not ok: break
                    vals.append(s0 - rule_score(evs, blue_step, weight, rec_decay, disc_cap=disc_cap))
                rows.append({"mode":mode,"k":k,"strategy":"random","delta":float(np.mean(vals))})
            # greedy
            evs = deep(base)
            for k in range(1, kmax+1):
                evs, ok = greedy_delete(evs, blue_step, weight, rec_decay)
                ds = s0 - rule_score(evs, blue_step, weight, rec_decay, disc_cap=disc_cap)
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
    evs = deep(evs)
    if 0<=idx<len(evs):
        st = evs[idx].get("step", 0) or 0
        evs[idx]["step"] = int(st) + int(jitter)
    return evs

def replace_random_cross(evs, idx, cur_type):
    evs = deep(evs)
    if 0<=idx<len(evs):
        candidates = [t for t in ALL_TYPES if t != cur_type]
        evs[idx]["type"] = random.choice(candidates) if candidates else cur_type
    return evs

def replace_freq_weighted_cross(evs, idx, cur_type, freq_map):
    evs = deep(evs)
    if 0<=idx<len(evs):
        items = [(t, freq_map.get(t,0.0)) for t in ALL_TYPES if t != cur_type]
        if not items: return evs
        types, probs = zip(*items)
        probs = np.array(probs, dtype=float); probs = probs/ probs.sum() if probs.sum() > 0 else np.ones_like(probs)/len(probs)
        evs[idx]["type"] = str(np.random.choice(types, p=probs))
    return evs

def run_cross_type_swaps(cases, out_dir, strategy="random_cross", jitter=1, weight=None, rec_decay=DEFAULT_REC_DECAY, disc_cap=2):
    rows=[]
    freq_map = global_type_freq(cases)
    for c in cases:
        blue_step = c.get("blue_step")
        for mode in ["MSE","MCE"]:
            raw = c.get("mse_events") if mode=="MSE" else c.get("mce_events")
            base = preprocess_events(raw, disc_cap=disc_cap)
            if not base: continue
            s0 = rule_score(base, blue_step, weight, rec_decay, disc_cap=disc_cap)
            # 从“清洗后”的 base 里找“最近一次该类型”
            for t in ALL_TYPES:
                # 注意：base 是清洗后的；这里一致性更好
                idx = -1; best=-10**9
                for i,e in enumerate(base):
                    if e.get("type")==t:
                        st = e.get("step", -10**9) or -10**9
                        if st > best: best=st; idx=i
                if idx < 0:
                    rows.append({"mode":mode,"type":t,"delta":0.0,"strategy":strategy,"file":pathlib.Path(c.get("file","")).stem})
                    continue
                if strategy == "same_type_diffstep":
                    evs = replace_same_type_diffstep(base, idx, jitter=jitter)
                elif strategy == "freq_weighted_cross":
                    evs = replace_freq_weighted_cross(base, idx, cur_type=t, freq_map=freq_map)
                else:
                    evs = replace_random_cross(base, idx, cur_type=t)
                s1 = rule_score(evs, blue_step, weight, rec_decay, disc_cap=disc_cap)
                rows.append({
                    "mode": mode, "type": t, "delta": s0 - s1,
                    "delta_rel": (s0 - s1) / max(s0, 1e-6),
                    "strategy": strategy, "file": pathlib.Path(c.get("file", "")).stem
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
    # 显式无关：不设置 is_related 或设为 False
    evs.append({"type":"Sleep", "step": int(step), "is_related": False})
    return evs

def run_noise_timing_sensitivity(cases, out_dir, offsets, weight=None, rec_decay=DEFAULT_REC_DECAY, disc_cap=2):
    """在 blue_step + offset 注入 Sleep，统计 Δ 并画曲线；这里 allow_unrelated=True"""
    rows=[]
    for c in cases:
        blue_step = c.get("blue_step")
        for mode in ["MSE","MCE"]:
            raw = c.get("mse_events") if mode=="MSE" else c.get("mce_events")
            base = preprocess_events(raw, disc_cap=disc_cap)  # 基线仍是“相关+去偏”
            if not base: continue
            s0 = rule_score(base, blue_step, weight, rec_decay, disc_cap=disc_cap)
            ref = int(blue_step) if isinstance(blue_step, int) else max([e.get("step",0) or 0 for e in base]+[0])
            for off in offsets:
                # 注入无关后，打分时 allow_unrelated=True，让无关事件参与打分
                evs = inject_unrelated_at(base, ref + int(off))
                s1 = rule_score(evs, blue_step, weight, rec_decay, disc_cap=disc_cap, allow_unrelated=True)
                rows.append({"mode":mode, "offset": int(off), "delta": s0 - s1, "file":pathlib.Path(c.get("file","")).stem})
    df = pd.DataFrame(rows)
    df.to_csv(out_dir/"noise_timing_sensitivity.csv", index=False, encoding="utf-8-sig")
    # plot
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
    ap.add_argument("--noise-offsets", type=str, default=os.environ.get("CYBDEF_NOISE_OFFSETS","-5,-3,-1,0,1,3,5"))
    ap.add_argument("--disc-cap", type=int, default=2, help="发现类上限（仅取最近 K 条）")
    args = ap.parse_args()

    ensure_dirs()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rng(args.seed)

    weight = dict(WEIGHT_DEFAULT)
    rec_decay = args.rec_decay
    disc_cap = int(args.disc_cap)

    # 数据
    cases = load_cases(MSE_PATH)

    # 1) 累积删证曲线（related-only）
    run_cumulative_deletion(cases, OUT_DIR, K=args.cumu_k, trials=args.cumu_trials, weight=weight, rec_decay=rec_decay, disc_cap=disc_cap)

    # 2) 跨类型换证（related-only）
    run_cross_type_swaps(cases, OUT_DIR, strategy=args.swap_strategy, jitter=args.jitter, weight=weight, rec_decay=rec_decay, disc_cap=disc_cap)

    # 3) 无关证据时序敏感性（注入无关 + 允许计入打分）
    offsets = parse_offsets(args.noise_offsets)
    run_noise_timing_sensitivity(cases, OUT_DIR, offsets=offsets, weight=weight, rec_decay=rec_decay, disc_cap=disc_cap)

    # 3) 在 main() 末尾写报告前，增加证据规模分布与直方图
    mse_sizes = [len(c.get("mse_events") or []) for c in cases]
    mce_sizes = [len(c.get("mce_events") or []) for c in cases]

    np.save(OUT_DIR / "mse_sizes.npy", np.array(mse_sizes))
    np.save(OUT_DIR / "mce_sizes.npy", np.array(mce_sizes))

    plt.figure(figsize=(6, 4));
    plt.hist(mse_sizes, bins=range(0, 10));
    plt.title("MSE size distribution");
    plt.xlabel("|MSE|");
    plt.tight_layout()
    plt.savefig(OUT_DIR / "hist_mse_size.png", dpi=150);
    plt.close()

    plt.figure(figsize=(6, 4));
    plt.hist(mce_sizes, bins=range(0, 10));
    plt.title("MCE size distribution");
    plt.xlabel("|MCE|");
    plt.tight_layout()
    plt.savefig(OUT_DIR / "hist_mce_size.png", dpi=150);
    plt.close()

    # 简要报告
    with (OUT_DIR/"report_v3.md").open("w", encoding="utf-8") as w:
        w.write("# Offline Intervention v3 (No-Model)\n")
        w.write(f"- swap-strategy: {args.swap_strategy}, jitter={args.jitter}\n")
        w.write(f"- cumulative K={args.cumu_k}, trials={args.cumu_trials}\n")
        w.write(f"- noise offsets={offsets}\n")
        w.write(f"- recency decay={rec_decay}\n")
        w.write(f"- disc_cap={disc_cap}\n")
        w.write("\n## Evidence size\n")
        w.write(f"- |MSE| mean±std: {np.mean(mse_sizes):.2f} ± {np.std(mse_sizes):.2f}\n")
        w.write(f"- |MCE| mean±std: {np.mean(mce_sizes):.2f} ± {np.std(mce_sizes):.2f}\n")
        w.write("- See histograms: hist_mse_size.png, hist_mce_size.png\n")
    print("✅ Done:", OUT_DIR)

if __name__ == "__main__":
    main()
