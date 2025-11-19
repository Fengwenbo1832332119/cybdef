# -*- coding: utf-8 -*-
# scripts/online/replay_interventions.py
from typing import List, Dict, Any, Tuple, Optional
import copy, random, numpy as np

ALL_TYPES = ["PrivilegeEscalate","ExploitRemoteService","DiscoverNetworkServices","DiscoverRemoteSystems"]

def deep(x): return copy.deepcopy(x)

# ---------- 基础编辑算子 ----------
def delete_latest_of_type(seq: List[Dict], t: str) -> List[Dict]:
    seq = deep(seq)
    for i in range(len(seq)-1, -1, -1):
        if seq[i].get("name") == t:
            del seq[i]; break
    return seq

def delete_first_of_type(seq: List[Dict], t: str) -> List[Dict]:
    seq = deep(seq)
    for i in range(len(seq)):
        if seq[i].get("name") == t:
            del seq[i]; break
    return seq

def delete_at_index(seq: List[Dict], idx: int) -> List[Dict]:
    seq = deep(seq)
    if 0 <= idx < len(seq): del seq[idx]
    return seq

def replace_type_at(seq: List[Dict], idx: int, new_type: str) -> List[Dict]:
    """仅替换 name，参数原样保留；不合法时由 validate_or_sleep 兜底"""
    seq = deep(seq)
    if 0 <= idx < len(seq):
        seq[idx]["name"] = new_type
    return seq

def time_shift(seq: List[Dict], idx: int, shift: int) -> List[Dict]:
    seq = deep(seq)
    if 0 <= idx < len(seq):
        j = max(0, min(len(seq)-1, idx + shift))
        act = seq.pop(idx)
        seq.insert(j, act)
    return seq

def keep_related_only(seq: List[Dict]) -> List[Dict]:
    RELATED = {"PrivilegeEscalate","ExploitRemoteService"}
    return [a for a in seq if a.get("name") in RELATED]

# ---------- 索引选择 ----------
def find_index(seq: List[Dict], t: str, which: str) -> Optional[int]:
    if which == "first":
        for i,a in enumerate(seq):
            if a.get("name")==t: return i
    elif which == "last":
        for i in range(len(seq)-1, -1, -1):
            if seq[i].get("name")==t: return i
    return None

# ---------- 跨类型策略 ----------
def random_cross_type(t_now: str) -> str:
    cands = [t for t in ALL_TYPES if t != t_now]
    return random.choice(cands) if cands else t_now

def freq_weighted_cross_type(t_now: str, freq_map: dict) -> str:
    items = [(t, freq_map.get(t, 1e-9)) for t in ALL_TYPES if t != t_now]
    if not items: return t_now
    types, probs = zip(*items)
    probs = np.array(probs, dtype=float)
    probs = probs / probs.sum() if probs.sum() > 0 else np.ones_like(probs)/len(probs)
    return str(np.random.choice(types, p=probs))

# ---------- 解析工具（CLI用） ----------
def parse_csv_list(s: str) -> List[str]:
    """'a,b,c' -> ['a','b','c']；空 -> []"""
    s = (s or "").strip()
    if not s: return []
    return [x.strip() for x in s.split(",") if x.strip()]

def build_freq_map(seq: List[Dict]) -> dict:
    cnt={}
    for a in seq:
        t=a.get("name")
        if t: cnt[t]=cnt.get(t,0)+1
    return cnt
