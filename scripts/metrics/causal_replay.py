# -*- coding: utf-8 -*-
"""
批量回放对比（causal_replay）- 纯离线CSKG验证版
- 读取一条或多条 cc2 日志（.jsonl）
- 对 baseline + 多种"环境级干预"（删证/换证/时序/充分性）进行离线规则得分计算
- 结果写入 CSV，并绘制简单柱状图
用法示例：
python causal_replay.py --logs "path_to_your_logs/*.jsonl" --out "reports/causal_replay" --del-types "PrivilegeEscalate,ExploitRemoteService" --sufficiency --score-mode offline
"""

import os
import sys
import glob
import argparse
import pathlib
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ==== 根目录设置与路径添加 ====
ROOT = pathlib.Path(__file__).resolve().parents[2]
paths_to_add = [str(ROOT), str(ROOT / "scripts"), str(ROOT / "scripts" / "metrics")]
for p in paths_to_add:
    if p not in sys.path:
        sys.path.insert(0, p)

# ==== 项目内部依赖 ====
try:
    from common.paths import REPORTS_DIR, ensure_dirs

    print("✅ 成功导入 common.paths")
except ImportError as e:
    print(f"❌ 导入 common.paths 失败: {e}")
    REPORTS_DIR = ROOT / "reports"


    def ensure_dirs():
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# 导入离线回放工具
try:
    from online.replay_core import load_episode_actions
    from online.replay_interventions import (
        deep, parse_csv_list,
        delete_first_of_type, delete_latest_of_type,
        replace_type_at, random_cross_type, freq_weighted_cross_type,
        keep_related_only, time_shift, find_index, build_freq_map
    )

    print("✅ 成功导入 online 工具")
except ImportError as e:
    print(f"❌ 导入 online 工具失败: {e}")
    sys.exit(1)

# ==== 规则评分函数（保持不变） ====
ALL_TYPES = ["PrivilegeEscalate", "ExploitRemoteService", "DiscoverNetworkServices", "DiscoverRemoteSystems", "Sleep"]
WEIGHT = {"PrivilegeEscalate": 4.0, "ExploitRemoteService": 3.0, "DiscoverNetworkServices": 1.2,
          "DiscoverRemoteSystems": 1.0}
REC_DECAY = 0.90
IS_DISC = {"DiscoverNetworkServices", "DiscoverRemoteSystems"}
DISC_CAP = 2


def classify_name_to_type(name: str):
    if not name: return None
    if name in ALL_TYPES: return name
    if "Privilege" in name: return "PrivilegeEscalate"
    if "Exploit" in name: return "ExploitRemoteService"
    if "DiscoverNetworkServices" in name: return "DiscoverNetworkServices"
    if "DiscoverRemoteSystems" in name: return "DiscoverRemoteSystems"
    if "Sleep" in name: return "Sleep"
    return None


# 离线口径对齐（调用 build_feats_related）
try:
    from scripts.common.feat_utils import build_feats_related


    def rule_score_offline(red_actions, disc_cap=DISC_CAP, weight=WEIGHT, rec_decay=REC_DECAY, debug=False):
        case = {"mse_events": [], "mce_events": [], "blue_step": len(red_actions)}
        for i, a in enumerate(red_actions, start=1):
            t = classify_name_to_type(a.get("name", ""))
            if not t: continue
            case["mse_events"].append({"type": t, "step": i})
        try:
            feat = build_feats_related(case, disc_cap=disc_cap)
        except Exception as e:
            if debug: print(f"[rule_score_offline] build_feats_related failed: {e}")
            return rule_score_simple(red_actions, weight=weight, rec_decay=rec_decay)
        s = 0.0
        if isinstance(feat, dict) and isinstance(feat.get("per_type"), dict):
            per = feat["per_type"]
            for k, w in weight.items():
                s += w * float(per.get(k, 0.0))
        elif isinstance(feat, dict) and all(k in feat for k in weight):
            for k, w in weight.items():
                s += w * float(feat.get(k, 0.0))
        elif isinstance(feat, dict) and isinstance(feat.get("score"), (int, float)):
            s = float(feat["score"])
        else:
            eff = {t: 0.0 for t in weight}
            disc_seen = {t: 0 for t in IS_DISC}
            blue_step = len(red_actions)
            for i, a in enumerate(red_actions, start=1):
                t = classify_name_to_type(a.get("name", ""))
                if not t or t not in weight: continue
                decay = rec_decay ** max(0, (blue_step - i))
                if t in IS_DISC:
                    if disc_seen[t] >= disc_cap: continue
                    disc_seen[t] += 1
                eff[t] += decay
            s = sum(weight[t] * eff[t] for t in eff)
        if debug: print(f"[rule_score_offline] score={s:.6f}")
        return float(s)
except Exception as e:
    print(f"⚠️ 导入 feat_utils 失败: {e}，使用 simple 评分")


    def rule_score_offline(red_actions, disc_cap=DISC_CAP, weight=WEIGHT, rec_decay=REC_DECAY, debug=False):
        return rule_score_simple(red_actions, weight=weight, rec_decay=rec_decay)


def rule_score_simple(red_actions, weight=WEIGHT, rec_decay=REC_DECAY):
    if not red_actions: return 0.0
    blue_step = len(red_actions)
    s = 0.0
    for i, a in enumerate(red_actions, start=1):
        t = classify_name_to_type(a.get("name", ""))
        if not t or t not in weight: continue
        decay = rec_decay ** max(0, (blue_step - i))
        s += weight[t] * decay
    return float(s)


# ------- 干预工具（与 online_intervention_suite 一致） -------
def apply_deletions(red_seq, del_types, which):
    cur = deep(red_seq)
    for t in del_types:
        cur = delete_first_of_type(cur, t) if which == "first" else delete_latest_of_type(cur, t)
    return cur


def parse_swap_map(s: str):
    s = (s or "").strip()
    if not s: return {}
    out = {}
    for pair in s.split(","):
        if ":" in pair:
            a, b = pair.split(":", 1)
            a, b = a.strip(), b.strip()
            if a and b: out[a] = b
    return out


def apply_swaps(red_seq, swap_strategy, target_type, which, swap_map):
    cur = deep(red_seq)
    if swap_map:
        for src, dst in swap_map.items():
            while True:
                idx = find_index(cur, src, which)
                if idx is None: break
                cur = replace_type_at(cur, idx, dst)
        return cur
    if swap_strategy and swap_strategy != "none":
        idx = find_index(cur, target_type, which)
        if idx is not None:
            freq = build_freq_map(cur)
            new_t = random_cross_type(target_type) if swap_strategy == "random_cross" else freq_weighted_cross_type(
                target_type, freq)
            cur = replace_type_at(cur, idx, new_t)
    return cur


def apply_time_shift_spec(red_seq, time_shift_spec: str):
    cur = deep(red_seq)
    if not time_shift_spec: return cur
    items = [x.strip() for x in time_shift_spec.split(",") if x.strip()]
    for it in items:
        if ":" not in it: continue
        t, s = it.split(":", 1)
        t, s = t.strip(), s.strip()
        if not t or not s: continue
        try:
            sh = int(s)
        except:
            sh = int(s.replace("+", "")) if s.replace("+", "").lstrip("-").isdigit() else 0
        idx = find_index(cur, t, which="last")
        if idx is not None: cur = time_shift(cur, idx, sh)
    return cur


# ================= CLI & 主流程 =================
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs", required=True, help="日志路径或通配符，如 C:\\cybdef\\logs\\*.jsonl")
    ap.add_argument("--out", type=str, default=str(REPORTS_DIR / "causal_replay"))
    ap.add_argument("--del-types", type=str, default="", help="要删除的动作类型，逗号分隔")
    ap.add_argument("--del-which", choices=["first", "last"], default="last", help="删除第一个还是最后一个")
    ap.add_argument("--swap-strategy", choices=["none", "random_cross", "freq_weighted"], default="none",
                    help="动作替换策略")
    ap.add_argument("--swap-target-type", type=str, default="PrivilegeEscalate", help="要替换的目标动作类型")
    ap.add_argument("--swap-which", choices=["first", "last"], default="first", help="替换第一个还是最后一个")
    ap.add_argument("--swap-map", type=str, default="", help="动作映射，格式为 '源类型:目标类型,源类型:目标类型'")
    ap.add_argument("--time-shift", type=str, default="", help="时间偏移，格式为 '动作类型:偏移步数,动作类型:偏移步数'")
    ap.add_argument("--sufficiency", action="store_true", help="是否进行充分性分析")
    ap.add_argument("--score-mode", choices=["simple", "offline"], default="offline", help="规则得分计算模式")
    ap.add_argument("--debug", action="store_true", help="是否开启调试模式")
    return ap.parse_args()


def main():
    args = parse_args()
    ensure_dirs()
    out_dir = pathlib.Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    score_fn = rule_score_offline if args.score_mode == "offline" else rule_score_simple
    paths = sorted([pathlib.Path(p) for p in glob.glob(args.logs)])
    if not paths: raise FileNotFoundError(f"No logs matched: {args.logs}")

    all_rows = []
    for log_path in paths:
        print(f"处理日志: {log_path.name}")
        try:
            red0, blue0 = load_episode_actions(log_path)
        except Exception as e:
            print(f"[WARN] skip {log_path.name}: {e}"); continue

        baseline_score = score_fn(red0, debug=args.debug)
        all_rows.append({"log": log_path.name, "variant": "baseline", "reward_red": 0.0, "reward_blue": 0.0,
                         "rule_score": baseline_score})

        del_types = parse_csv_list(args.del_types)
        if del_types:
            red_del = apply_deletions(red0, del_types, args.del_which)
            score_del = score_fn(red_del, debug=args.debug)
            all_rows.append(
                {"log": log_path.name, "variant": f"delete[{args.del_which}]_{'+'.join(del_types)}", "reward_red": 0.0,
                 "reward_blue": 0.0, "rule_score": score_del})

        swap_map = parse_swap_map(args.swap_map)
        if args.swap_strategy != "none" or swap_map:
            red_sw = apply_swaps(red0, args.swap_strategy, args.swap_target_type, args.swap_which, swap_map)
            score_sw = score_fn(red_sw, debug=args.debug)
            tag = args.swap_strategy if args.swap_strategy != "none" else "map"
            all_rows.append({"log": log_path.name, "variant": f"swap[{tag}]", "reward_red": 0.0, "reward_blue": 0.0,
                             "rule_score": score_sw})

        if args.time_shift:
            red_ts = apply_time_shift_spec(red0, args.time_shift)
            score_ts = score_fn(red_ts, debug=args.debug)
            all_rows.append({"log": log_path.name, "variant": f"time_shift({args.time_shift})", "reward_red": 0.0,
                             "reward_blue": 0.0, "rule_score": score_ts})

        if args.sufficiency:
            red_suf = keep_related_only(red0)
            score_suf = score_fn(red_suf, debug=args.debug)
            all_rows.append(
                {"log": log_path.name, "variant": "sufficiency_related_only", "reward_red": 0.0, "reward_blue": 0.0,
                 "rule_score": score_suf})

    df = pd.DataFrame(all_rows)
    csv_out = out_dir / "causal_replay_results.csv"
    df.to_csv(csv_out, index=False, encoding="utf-8-sig")
    print(f"✅ 纯离线CSKG验证完成！结果已保存至: {csv_out}")

    for log_name, sub in df.groupby("log"):
        plt.figure(figsize=(10, 4))
        plt.bar(sub["variant"], sub["rule_score"])
        plt.xticks(rotation=25, ha="right")
        plt.ylabel("rule_score")
        plt.title(f"Rule Score by Variant — {log_name} (纯离线CSKG验证)")
        plt.tight_layout()
        plt.savefig(out_dir / f"rule_score_{pathlib.Path(log_name).stem}.png", dpi=160)
        plt.close()
    print(f"📊 图表已生成至: {out_dir}/rule_score_*.png")


if __name__ == "__main__":
    main()