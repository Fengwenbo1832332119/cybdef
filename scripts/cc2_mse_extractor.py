# -*- coding: utf-8 -*-
import json, re, pathlib, ipaddress
from collections import deque, defaultdict

# === 输入/输出 ===
IN_FILES = [
    r"C:\cybdef\logs\cc2_remove_meander.jsonl",
    r"C:\cybdef\logs\cc2_restore_meander.jsonl",
    r"C:\cybdef\logs\cc2_remove_bline.jsonl",
    r"C:\cybdef\logs\cc2_restore_bline.jsonl",
]
OUT_PATH = pathlib.Path(r"C:\cybdef\graph\mse_cases.jsonl")

# === 参数 ===
WINDOW_STEPS = 12

# === 解析与标准化 ===
IP_RE       = re.compile(r"\b\d{1,3}(?:\.\d{1,3}){3}\b")
USR_RE      = re.compile(r"\b(User\d+)\b")
ENT_RE      = re.compile(r"\b(Enterprise\d+)\b")
SUBNET_RE   = re.compile(r"\b\d{1,3}(?:\.\d{1,3}){3}/\d{1,2}\b")
HOSTNAME_RE = re.compile(r"\b(op[_-]?server\d+|db[_-]?server\d+|web[_-]?server\d+)\b", re.I)

def norm_target(s: str):
    if not s: return None, None
    s = s.strip()
    if (m:=IP_RE.search(s)):        return "Host", m.group(0)
    if (m:=ENT_RE.search(s)):       return "Enterprise", m.group(1)
    if (m:=USR_RE.search(s)):       return "User", m.group(1)
    if (m:=SUBNET_RE.search(s)):    return "Subnet", m.group(0)
    if (m:=HOSTNAME_RE.search(s)):  return "HostName", m.group(1).lower()
    return "Object", s

def parse_action(a_str):
    if not a_str: return None, None
    parts = a_str.strip().split(maxsplit=1)
    act = parts[0]; tgt = parts[1] if len(parts)>1 else None
    return act, tgt

def ip_in_subnet(ip: str, cidr: str) -> bool:
    try:
        return ipaddress.ip_address(ip) in ipaddress.ip_network(cidr, strict=False)
    except Exception:
        return False

def _equal_name(a, b):
    return str(a or "").lower().replace("_","-") == str(b or "").lower().replace("_","-")

# ======= 收紧后的“强匹配”口径：仅允许 Host/HostName 强等价 或 Subnet→Host 实包含 =======
def match_target(red_type, red_tnorm, blue_tnorm):
    if not red_tnorm or not blue_tnorm:
        return False
    (rtp, rval), (btp, bval) = red_tnorm, blue_tnorm

    # Host <-> Host 或 HostName <-> HostName：同名视为命中（忽略下划线/大小写）
    if rtp == btp and rtp in ("Host", "HostName") and _equal_name(rval, bval):
        return True
    # Host <-> HostName：同名视为命中
    if {rtp, btp} == {"Host", "HostName"} and _equal_name(rval, bval):
        return True
    # Subnet -> Host：必须真实包含
    if rtp == "Subnet" and btp == "Host" and ip_in_subnet(bval, rval):
        return True

    # ❌ 取消：HostName↔Enterprise、Exploit→User 等“弱对齐”，避免伪相关
    return False

# 动作重要性（用于排序/打分）
PRIORITY = {
    "PrivilegeEscalate": 4,
    "ExploitRemoteService": 3,
    "DiscoverNetworkServices": 2,
    "DiscoverRemoteSystems": 2,
}
def is_red(a_type): return a_type in PRIORITY

# === 多因解释选择器 ===
MAX_TOTAL_EVS = 4
MAX_PER_TYPE  = 2
RECENCY_DECAY = 0.98

def select_explanations(cands, mode="minimal"):
    if not cands: return []
    if mode == "minimal":
        kept = {}
        for c in sorted(cands, key=lambda x: x["step"], reverse=True):
            if c["type"] not in kept:
                kept[c["type"]] = c
        return sorted(kept.values(), key=lambda x: PRIORITY.get(x["type"], 0), reverse=True)
    elif mode == "multi":
        max_step = max(c["step"] for c in cands)
        scored = []
        for c in cands:
            p = PRIORITY.get(c["type"], 0)
            rec = RECENCY_DECAY ** max(0, (max_step - c["step"]))
            scored.append((p + rec, c))
        scored.sort(key=lambda x: x[0], reverse=True)
        per_type = defaultdict(int); out = []
        for _, c in scored:
            if per_type[c["type"]] >= MAX_PER_TYPE:
                continue
            out.append(c); per_type[c["type"]] += 1
            if len(out) >= MAX_TOTAL_EVS:
                break
        return sorted(out, key=lambda x: (PRIORITY.get(x["type"], 0), x["step"]), reverse=True)
    else:
        raise ValueError("mode must be 'minimal' or 'multi'")

def parse_action_safe(rec, role):
    a = rec.get(f"{role}_action") or rec.get(f"{role}_action_raw")
    if not a: return None, None
    parts = a.strip().split(maxsplit=1)
    return parts[0], (parts[1] if len(parts) > 1 else None)

def norm_target_safe(rec, raw_txt):
    tn = rec.get("target_norm")
    if isinstance(tn, (list, tuple)) and len(tn) >= 2:
        return tn[0], (tn[1] or "").strip().lower()
    return norm_target(raw_txt)

def process_file(path, writer):
    with open(path, "r", encoding="utf-8") as f:
        recs = [json.loads(line) for line in f if line.strip()]
    recs = [r for r in recs if not r.get("meta")]
    recs.sort(key=lambda r: (r.get("episode", -1), r.get("step", -1)))

    wins = defaultdict(deque)  # episode -> deque

    for r in recs:
        step = r.get("step")
        if step is None:
            continue
        ep = r.get("episode", -1)

        win = wins[ep]
        while win and step - win[0]["step"] > WINDOW_STEPS:
            win.popleft()

        ra, rt = parse_action_safe(r, "red")
        ba, bt = parse_action_safe(r, "blue")
        r_norm = norm_target_safe(r, rt) if rt else None
        b_norm = norm_target_safe(r, bt) if bt else None

        if ra and is_red(ra):
            win.append({
                "step": step, "type": ra,
                "target_raw": rt, "target_norm": r_norm,
                "record": r
            })

        if ba in ("Remove","Restore","Monitor"):
            cands = []
            if b_norm:
                seen = set()
                for e in reversed(win):
                    if match_target(e["type"], e["target_norm"], b_norm):
                        key = (e["step"], e["type"], e["target_raw"])
                        if key in seen:
                            continue
                        seen.add(key)
                        cands.append({
                            "step": e["step"],
                            "delta": step - e["step"],
                            "type": e["type"],
                            "target": e["target_raw"],
                            "is_related": True,  # ✅ 显式标注“相关”
                            "red_action": e["record"].get("red_action") or e["record"].get("red_action_raw")
                        })

            mse = select_explanations(cands, mode="minimal")
            mce = select_explanations(cands, mode="multi")
            writer.write(json.dumps({
                "file": str(path),
                "episode": ep,
                "blue_step": step,
                "blue_action": r.get("blue_action") or r.get("blue_action_raw"),
                "blue_target_norm": b_norm,
                "window": WINDOW_STEPS,
                "mse_events": mse,
                "mce_events": mce,
                "candidate_size": len(cands),
                "mse_size": len(mse),
                "mce_size": len(mce)
            }, ensure_ascii=False) + "\n")

def main():
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUT_PATH.open("w", encoding="utf-8") as w:
        for p in IN_FILES:
            fp = pathlib.Path(p)
            if fp.exists():
                process_file(fp, w)
    print("✅ MSE/MCE saved →", OUT_PATH)

if __name__ == "__main__":
    main()
