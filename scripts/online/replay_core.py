# -*- coding: utf-8 -*-
# scripts/online/replay_core.py
import json, re, pathlib

# 允许的类型名（与线下口径一致）
KNOWN_TYPES = {
    "PrivilegeEscalate", "ExploitRemoteService", "DiscoverNetworkServices", "DiscoverRemoteSystems", "Sleep"
}

_IP_RE    = re.compile(r"\b\d{1,3}(?:\.\d{1,3}){3}\b")
_CIDR_RE  = re.compile(r"\b\d{1,3}(?:\.\d{1,3}){3}/\d{1,2}\b")
_HOST_RE  = re.compile(r"\b([A-Za-z_][\w\-]*)\b")

def _norm_type(name: str) -> str | None:
    if not name:
        return None
    name = str(name)
    if "DiscoverNetworkServices" in name:
        return "DiscoverNetworkServices"
    if "DiscoverRemoteSystems" in name:
        return "DiscoverRemoteSystems"
    if "Exploit" in name:
        return "ExploitRemoteService"
    if "Privilege" in name:
        return "PrivilegeEscalate"
    if "Sleep" in name:
        return "Sleep"
    if name in KNOWN_TYPES:
        return name
    return None

def _parse_args_from_raw(raw: str):
    """从 'Action arg' 这种字符串里提取目标参数（IP/CIDR/Hostname），并给出 kwargs。"""
    raw = raw.strip()
    # IP/CIDR 优先
    if m := _CIDR_RE.search(raw):
        return {"subnet": m.group(0)}
    if m := _IP_RE.search(raw):
        return {"ip_address": m.group(0)}
    # Hostname（用于 PrivilegeEscalate 场景）
    toks = raw.split()
    if len(toks) >= 2:
        # 第二段尝试作为 hostname
        return {"hostname": toks[1]}
    return {}

def _action_dict(name: str, step: int | None = None, raw: str | None = None) -> dict:
    """最小动作描述（用于评分与回放），包含原始 raw 便于回放失败时调试。"""
    return {"name": name, "step": step, "raw": raw}

def _load_replay_style(lines: list[dict]) -> tuple[list[dict], list[dict]]:
    """从逐步回放日志中提取 red/blue 动作（字段兼容多写法）"""
    red, blue = [], []
    for row in lines:
        r = row.get("red_action") or row.get("action_red") or row.get("red_act") or row.get("red")
        b = row.get("blue_action") or row.get("action_blue") or row.get("blue_act") or row.get("blue")
        st = row.get("step")
        if r:
            t = _norm_type(r if isinstance(r, str) else (r.get("name") or r.get("type") or r.get("class")))
            if t:
                raw = r if isinstance(r, str) else str(r)
                red.append(_action_dict(t, st if isinstance(st, int) else None, raw=raw))
        if b:
            t = _norm_type(b if isinstance(b, str) else (b.get("name") or b.get("type") or b.get("class")))
            if t:
                raw = b if isinstance(b, str) else str(b)
                blue.append(_action_dict(t, st if isinstance(st, int) else None, raw=raw))
    return red, blue

def _load_case_style(lines: list[dict]) -> tuple[list[dict], list[dict]]:
    """
    兼容“案例型”日志（含 mse/mce_events）：
    - 把 mce_events 视作红方动作序列（按 step 排序）
    """
    evs = []
    for row in lines:
        mce = row.get("mce_events") or []
        for e in mce:
            t = _norm_type(e.get("type"))
            if not t: continue
            step = e.get("step")
            evs.append(_action_dict(t, step, raw=f"{t} {e.get('target','')}"))
    evs.sort(key=lambda x: (999999 if x.get("step") is None else int(x["step"])))
    return evs, []

def load_episode_actions(path: pathlib.Path) -> tuple[list[dict], list[dict]]:
    """
    返回 (red_actions, blue_actions)，元素是 {"name": 类型名, "step": 可选, "raw": 原始串}
    自动识别：
      1) 回放型：每步包含 red_action/blue_action
      2) 案例型：只包含 mce/mse_events
    """
    if isinstance(path, str):
        path = pathlib.Path(path)
    lines = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    lines.append(json.loads(line))
                except Exception:
                    pass
    if not lines:
        return [], []
    has_replay = any(("red_action" in r or "action_red" in r or "red_act" in r or "red" in r) for r in lines)
    has_case  = any(("mce_events" in r or "mse_events" in r) for r in lines)
    if has_replay:
        return _load_replay_style(lines)
    if has_case:
        return _load_case_style(lines)
    return [], []

# ---------- 关键：把日志动作还原成 CybORG Action 对象 ----------
def action_from_record(act: dict):
    """
    支持 act:
      {"name": "DiscoverNetworkServices", "raw": "...", "step": ...}
    或纯字符串（向后兼容）。
    """
    try:
        from CybORG.Simulator.Actions import (
            DiscoverNetworkServices, DiscoverRemoteSystems,
            ExploitRemoteService, PrivilegeEscalate, Sleep
        )
    except Exception:
        # 如果导入失败，回到 Sleep，交给上层兜底
        return None

    if not act:
        return Sleep()
    if isinstance(act, str):
        raw = act
        t = _norm_type(raw)
        kwargs = _parse_args_from_raw(raw)
    else:
        raw = act.get("raw") or act.get("name", "")
        t = _norm_type(act.get("name") or raw)
        kwargs = _parse_args_from_raw(raw)

    # 所有动作优先加上 session=0
    if "session" not in kwargs:
        kwargs["session"] = 0

    try:
        if t == "DiscoverNetworkServices":
            return DiscoverNetworkServices(**kwargs)
        if t == "DiscoverRemoteSystems":
            return DiscoverRemoteSystems(**kwargs)
        if t == "ExploitRemoteService":
            return ExploitRemoteService(**kwargs)
        if t == "PrivilegeEscalate":
            # 若没有 hostname 参数，再从 raw 的第二段兜底取
            if "hostname" not in kwargs:
                toks = str(raw).split()
                if len(toks) >= 2:
                    kwargs["hostname"] = toks[1]
            return PrivilegeEscalate(**kwargs)
        if t == "Sleep":
            return Sleep(**kwargs)
    except TypeError:
        # 参数不匹配时作为 Sleep 兜底
        return Sleep()

    # 未识别类型 → Sleep
    return Sleep()
