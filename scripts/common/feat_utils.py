# -*- coding: utf-8 -*-
# 所有特征只看相关证据；
# 发现类去重（同 type+target 只记一步，优先取最近），
# 并限幅（cap），delta_mean 只在相关上计算。
import numpy as np

DISC_TYPES = {"DiscoverRemoteSystems", "DiscoverNetworkServices"}
PRIV = "PrivilegeEscalate"
EXPL = "ExploitRemoteService"

def _dedup_keep_latest(evs):
    """按 (type, target) 去重，仅保留最近一步；仅处理标注 is_related=True 的记录。"""
    latest = {}
    for e in evs or []:
        if not e.get("is_related"):
            continue
        k = (e.get("type"), e.get("target"))
        if k not in latest or (e.get("step", -1) > latest[k].get("step", -1)):
            latest[k] = e
    return list(latest.values())

def _cap_count(n, cap):
    return float(min(int(n), int(cap)))

def _delta_mean_related(evs, blue_step):
    ds = []
    for e in evs or []:
        if not e.get("is_related"):
            continue
        st = e.get("step")
        if st is not None and blue_step is not None:
            ds.append(max(0, blue_step - st))
        elif "delta" in e and e["delta"] is not None:
            ds.append(max(0, int(e["delta"])))
    return float(np.mean(ds)) if ds else 0.0

def build_feats_related(case, disc_cap=2):
    """
    仅相关证据特征 + 发现类去重/限幅。
    返回一个干净的特征 dict，可直接用于你现有的离线指标或后续训练。
    """
    mse = case.get("mse_events") or []
    mce = case.get("mce_events") or []
    blue = case.get("blue_step")

    # 只保留相关 & 发现类去重（同 type+target 仅留最近一步）
    mse_rel = _dedup_keep_latest(mse)
    mce_rel = _dedup_keep_latest(mce)

    types_mse = [e.get("type","") for e in mse_rel]
    types_mce = [e.get("type","") for e in mce_rel]

    disc_mce_cnt = sum(t in DISC_TYPES for t in types_mce)
    disc_mse_cnt = sum(t in DISC_TYPES for t in types_mse)

    feat = {
        # 仅相关的规模
        "mse_size_rel": float(len(mse_rel)),
        "mce_size_rel": float(len(mce_rel)),

        # 关键类型（仅相关）
        "mce_priv": float(sum(t == PRIV for t in types_mce)),
        "mce_expl": float(sum(t == EXPL for t in types_mce)),

        # 发现类（仅相关）+ 限幅（cap）
        "mce_disc": _cap_count(disc_mce_cnt, disc_cap),
        "mse_disc": _cap_count(disc_mse_cnt, disc_cap),

        # 仅相关的时序均值
        "delta_mean_rel": _delta_mean_related(mse_rel + mce_rel, blue),

        # 是否存在关键证据（仅相关）
        "mse_priv_any": float(any(t == PRIV for t in types_mse)),
        "mse_expl_any": float(any(t == EXPL for t in types_mse)),
    }
    return feat
