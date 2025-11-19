# -*- coding: utf-8 -*-
"""
离线删证/换证（无模型）增强版：
1) 证据类型敏感性：逐类删除最近一条，统计 Δ
2) 累积删证曲线：k=1..K，(a) 随机删除均值；(b) 贪心删除“贡献最大”的事件
3) 跨类型换证：把某类型事件替换为另一类型；无关证据注入（Sleep）
输出：CSV + 可视化 + 总结报告
"""
import os, sys, json, copy, random, pathlib, itertools
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# --- 项目路径 ---
THIS = pathlib.Path(__file__).resolve()
PKG  = THIS.parents[1]
if str(PKG) not in sys.path:
    sys.path.insert(0, str(PKG))
from scripts.common.paths import GRAPH_DIR, REPORTS_DIR, ensure_dirs

# --- I/O & 参数 ---
MSE_PATH = GRAPH_DIR / "mse_cases.jsonl"
OUT_DIR  = REPORTS_DIR / "intervention_offline_nomodel_v2"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SEED = int(os.environ.get("CYBDEF_SEED", "1234"))
random.seed(SEED); np.random.seed(SEED)

# 评分权重与衰减（可按需调整或暴露为 CLI）
WEIGHT = {
    "PrivilegeEscalate": 4.0,
    "ExploitRemoteService": 3.0,
    "DiscoverNetworkServices": 1.5,
    "DiscoverRemoteSystems": 1.2,
    "Sleep": 0.2,           # 无关/噪声证据
}
REC_DECAY = 0.98
ALL_TYPES = ["PrivilegeEscalate","ExploitRemoteService","DiscoverNetworkServices","DiscoverRemoteSystems"]

# ---------- 评分 ----------
def rule_score(evs, blue_step):
    if not evs: return 0.0
    if blue_step is None:
        blue_step = max([e.get("step", 0) or 0 for e in evs] + [0])
    s = 0.0
    for e in evs:
        et = e.get("type","")
        w = WEIGHT.get(et, 0.5)
        st = e.get("step", None)
        decay = (REC_DECAY ** max(0, int(blue_step) - int(st))) if st is not None else 1.0
        s += w * decay
    return float(s)

def contrib(e, blue_step):
    """单事件对分数的边际贡献（当前权重*衰减），用于贪心删证"""
    et = e.get("type",""); w = WEIGHT.get(et, 0.5)
    st = e.get("step", None)
    decay = (REC_DECAY ** max(0, int(blue_step) - int(st))) if st is not None else 1.0
    return float(w * decay)

# ---------- 辅助 ----------
def deep(evs): return copy.deepcopy(evs or [])
def latest_idx_of_type(evs, t):
    # 最近一条 = 最大step；若缺step则取最后一个出现的
    idx = -1; best_step = -1
    for i, e in enumerate(evs):
        if e.get("type") == t:
            st = e.get("step")
            st = -1 if st is None else int(st)
            if st >= best_step:
                best_step = st; idx = i
    return idx

def rand_delete(evs):
    if not evs: return evs, False
    evs = deep(evs)
    i = random.randrange(len(evs))
    del evs[i]
    return evs, True

def greedy_delete(evs, blue_step):
    if not evs: return evs, False
    evs = deep(evs)
    scores = [contrib(e, blue_step) for e in evs]
    i = int(np.argmax(scores))
    del evs[i]
    return evs, True

def cross_type_replace(evs, target_idx, new_type):
    """把 evs[target_idx] 的 type 替换为 new_type（step 不变）"""
    evs = deep(evs)
    if 0 <= target_idx < len(evs):
        evs[target_idx]["type"] = new_type
    return evs

def inject_unrelated(evs, blue_step):
    """注入一条无关证据（Sleep），放在 blue_step-1，模拟近因噪声"""
    evs = deep(evs)
    step = (blue_step-1) if isinstance(blue_step,int) else (max([e.get("step",0) or 0 for e in evs]+[0]) + 1)
    evs.append({"type":"Sleep","step": step})
    return evs

# =========================================================
# 1) 基线 + 类型敏感性（删除某类型最近一条）
# =========================================================
def run_type_sensitivity(cases):
    rows=[]
    for c in cases:
        blue_step = c.get("blue_step")
        for mode in ["MSE","MCE"]:
            evs = c.get("mse_events") if mode=="MSE" else c.get("mce_events")
            evs = evs or []
            s0 = rule_score(evs, blue_step)
            for t in ALL_TYPES:
                i = latest_idx_of_type(evs, t)
                if i < 0:
                    rows.append({"mode":mode,"type":t,"delta":0.0,"had_type":0,
                                 "file":pathlib.Path(c.get("file","")).stem})
                    continue
                vv = deep(evs); del vv[i]
                s1 = rule_score(vv, blue_step)
                rows.append({"mode":mode,"type":t,"delta":s0-s1,"had_type":1,
                             "file":pathlib.Path(c.get("file","")).stem})
    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR/"type_sensitivity.csv", index=False, encoding="utf-8-sig")
    # 可视化：按类型分组的均值±std
    for mode in ["MSE","MCE"]:
        sub = df[df["mode"]==mode]
        if sub.empty: continue
        g = sub.groupby("type")["delta"].agg(["mean","std"]).reindex(ALL_TYPES)
        plt.figure(figsize=(8,4))
        plt.bar(g.index, g["mean"], yerr=g["std"], capsize=5)
        plt.title(f"Type Sensitivity (delete latest one) — {mode}")
        plt.ylabel("Δ score (mean±std)"); plt.xticks(rotation=15); plt.tight_layout()
        plt.savefig(OUT_DIR/f"type_sensitivity_{mode}.png", dpi=160); plt.close()
    return df

# =========================================================
# 2) 累积删证曲线（随机/贪心）
# =========================================================
def run_cumulative_deletion(cases, K=5, trials=10):
    rows=[]
    for c in cases:
        blue_step = c.get("blue_step")
        for mode in ["MSE","MCE"]:
            base = deep(c.get("mse_events") if mode=="MSE" else c.get("mce_events"))
            if not base:
                for k in range(1,K+1):
                    rows.append({"mode":mode,"k":k,"strategy":"random","delta":0.0})
                    rows.append({"mode":mode,"k":k,"strategy":"greedy","delta":0.0})
                continue
            s0 = rule_score(base, blue_step)
            n = len(base); kmax = min(K,n)

            # random: 多次试验取均值
            for k in range(1, kmax+1):
                vals=[]
                for _ in range(trials):
                    evs = deep(base)
                    for _del in range(k):
                        evs, ok = rand_delete(evs)
                        if not ok: break
                    vals.append(s0 - rule_score(evs, blue_step))
                rows.append({"mode":mode,"k":k,"strategy":"random","delta":float(np.mean(vals))})

            # greedy: 每次删贡献最大的
            evs = deep(base)
            for k in range(1, kmax+1):
                evs, ok = greedy_delete(evs, blue_step)
                ds = s0 - rule_score(evs, blue_step)
                rows.append({"mode":mode,"k":k,"strategy":"greedy","delta":ds})
    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR/"cumulative_deletion.csv", index=False, encoding="utf-8-sig")

    # 画曲线
    for mode in ["MSE","MCE"]:
        sub = df[df["mode"]==mode]
        if sub.empty: continue
        plt.figure(figsize=(7,4))
        for strat in ["random","greedy"]:
            s2 = sub[sub["strategy"]==strat].groupby("k")["delta"].mean()
            plt.plot(s2.index, s2.values, marker="o", label=strat)
        plt.title(f"Cumulative Deletion Curve — {mode}")
        plt.xlabel("k deletions"); plt.ylabel("Δ score (mean)"); plt.legend(); plt.tight_layout()
        plt.savefig(OUT_DIR/f"cumu_del_curve_{mode}.png", dpi=160); plt.close()
    return df

# =========================================================
# 3) 跨类型换证 & 无关证据注入
# =========================================================
def run_cross_type_and_noise(cases):
    rows=[]
    for c in cases:
        blue_step = c.get("blue_step")
        for mode in ["MSE","MCE"]:
            base = deep(c.get("mse_events") if mode=="MSE" else c.get("mce_events"))
            if not base:
                rows.append({"mode":mode,"op":"inject_unrelated","delta":0.0})
                continue
            s0 = rule_score(base, blue_step)

            # 3.1 对每个类型：把该类型最近一条换成另一类型（优先降级为 Discover…; 若本身已是 Discover 则换 Exploit）
            for t in ALL_TYPES:
                i = latest_idx_of_type(base, t)
                if i < 0:
                    rows.append({"mode":mode,"op":f"swap_{t}_none","delta":0.0})
                    continue
                if t in ("PrivilegeEscalate","ExploitRemoteService"):
                    new_t = "DiscoverNetworkServices"
                else:
                    new_t = "ExploitRemoteService"
                evs = cross_type_replace(base, i, new_t)
                s1 = rule_score(evs, blue_step)
                rows.append({"mode":mode,"op":f"swap_{t}_to_{new_t}","delta":s0-s1})

            # 3.2 注入无关证据（Sleep）
            evs2 = inject_unrelated(base, blue_step)
            s2 = rule_score(evs2, blue_step)
            rows.append({"mode":mode,"op":"inject_unrelated","delta":s0-s2})

    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR/"cross_type_and_noise.csv", index=False, encoding="utf-8-sig")

    # 可视化：每类 swap 的均值柱图 + noise
    for mode in ["MSE","MCE"]:
        sub = df[df["mode"]==mode]
        if sub.empty: continue
        # 只画 swap_* 和 inject_unrelated
        show = sub[sub["op"].str.startswith("swap_") | (sub["op"]=="inject_unrelated")]
        g = show.groupby("op")["delta"].mean().sort_values(ascending=False)
        plt.figure(figsize=(10,5))
        g.plot(kind="bar"); plt.ylabel("Δ score (mean)")
        plt.title(f"Cross-type Replacement & Unrelated Injection — {mode}")
        plt.tight_layout(); plt.savefig(OUT_DIR/f"cross_type_{mode}.png", dpi=160); plt.close()
    return df

# =========================================================
# 主流程 & 报告
# =========================================================
def load_cases():
    if not MSE_PATH.exists():
        raise FileNotFoundError(f"mse_cases.jsonl not found: {MSE_PATH}")
    cases=[]
    with MSE_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                cases.append(json.loads(line))
    return cases

def write_report():
    rpt = OUT_DIR / "report.md"
    with rpt.open("w", encoding="utf-8") as w:
        w.write("# Offline Intervention (No-Model) — Extended\n\n")
        w.write("## 1) Type Sensitivity\n")
        w.write("- 输出：type_sensitivity.csv / type_sensitivity_MSE.png / type_sensitivity_MCE.png\n\n")
        w.write("## 2) Cumulative Deletion Curve\n")
        w.write("- 输出：cumulative_deletion.csv / cumu_del_curve_MSE.png / cumu_del_curve_MCE.png\n\n")
        w.write("## 3) Cross-type Replacement & Unrelated Injection\n")
        w.write("- 输出：cross_type_and_noise.csv / cross_type_MSE.png / cross_type_MCE.png\n\n")
        w.write("**解读建议：**\n- 删除特定类型的Δ越大，说明该类型对解释越关键。\n")
        w.write("- 贪心曲线若显著高于随机曲线，表示系统主要依赖少数“高贡献”证据。\n")
        w.write("- 注入无关证据(Sleep)若Δ≈0，说明对噪声鲁棒；若Δ>0，说明被噪声误导（应调低该类权重/衰减）。\n")

def main():
    ensure_dirs()
    cases = load_cases()
    df1 = run_type_sensitivity(cases)
    df2 = run_cumulative_deletion(cases, K=int(os.environ.get("CYBDEF_CUMU_K","5")),
                                  trials=int(os.environ.get("CYBDEF_CUMU_TRIALS","10")))
    df3 = run_cross_type_and_noise(cases)
    write_report()
    print("✅ Done:", OUT_DIR)

if __name__ == "__main__":
    main()
