import json, csv, re, pathlib, sys
from collections import deque

# === 配置：输入日志 & 输出目录 ===
IN_FILES = [
    r"C:\cybdef\logs\cc2_remove.jsonl",
    r"C:\cybdef\logs\cc2_restore.jsonl",
    r"C:\cybdef\logs\cc2_remove_meander.jsonl",
    r"C:\cybdef\logs\cc2_restore_meander.jsonl",
    r"C:\cybdef\logs\cc2_remove_bline.jsonl",
    r"C:\cybdef\logs\cc2_restore_bline.jsonl",
]
OUT_DIR = pathlib.Path(r"/graph")
WINDOW_STEPS = 12  # 与 MSE 保持一致

# === 轻量映射：动作 -> ATT&CK（可后续细化/校准）===
ATTACK_MAP = {
    "DiscoverRemoteSystems": "T1018",
    "DiscoverNetworkServices": "T1046",
    "ExploitRemoteService": "T1210",
    "PrivilegeEscalate": "T1068",
    "Remove": "DFEND_Remove",
    "Restore": "DFEND_Restore",
    "Monitor": "DFEND_Monitor",
}

# === 归一化（与 3.5 一致）===
IP_RE       = re.compile(r"\b\d{1,3}(?:\.\d{1,3}){3}\b")
USR_RE      = re.compile(r"\b(User\d+)\b", re.I)
ENT_RE      = re.compile(r"\b(Enterprise\d+)\b", re.I)
SUBNET_RE   = re.compile(r"\b\d{1,3}(?:\.\d{1,3}){3}/\d{1,2}\b")
HOSTNAME_RE = re.compile(r"\bhost[-_]?(\d+)\b", re.I)

def norm_target(target: str):
    if not target:
        return None, None
    m = SUBNET_RE.search(target)
    if m: return "Subnet", m.group(0)
    m = IP_RE.search(target)
    if m: return "Host", m.group(0)
    m = ENT_RE.search(target)
    if m: return "Enterprise", m.group(1)
    m = USR_RE.search(target)
    if m: return "User", m.group(1)
    m = HOSTNAME_RE.search(target)
    if m: return "HostName", f"host-{m.group(1)}"
    return "Object", target.strip()

def parse_action(a_str):
    if not a_str: return None, None
    parts = a_str.strip().split(maxsplit=1)
    act = parts[0]
    tgt = parts[1] if len(parts) > 1 else None
    return act, tgt

def add_node(nodes, key, ntype, label=None):
    if key not in nodes:
        nodes[key] = {"id": key, "type": ntype, "label": label or key}

def add_edge(edges, src, dst, etype):
    edges.append({"src": src, "dst": dst, "type": etype})

# 匹配蓝方响应与红方证据（与 3.5 口径一致：目标对齐 + 时间因果）
def match_target(r_norm, b_norm):
    if not r_norm or not b_norm: return False
    (rtp, rval), (btp, bval) = r_norm, b_norm
    # 完全一致
    if rtp == btp and rval == bval:
        return True
    # Host 与 HostName 可在此扩展映射（暂不跨映射）
    return False

def process_file(path, nodes, edges):
    agent_red = "Agent:Red"
    agent_blue = "Agent:Blue"
    add_node(nodes, agent_red, "Agent", label="Red")
    add_node(nodes, agent_blue, "Agent", label="Blue")

    # 窗口：仅缓存红方事件用于与后续蓝方进行因果连边
    win = deque()

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            step = rec.get("step")
            t_iso = rec.get("t")

            ra, rt = parse_action(rec.get("red_action"))
            ba, bt = parse_action(rec.get("blue_action"))

            # 维护窗口：弹出超过 WINDOW_STEPS 的红方事件
            while win and step - win[0]["step"] > WINDOW_STEPS:
                win.popleft()

            # 红方事件
            if ra:
                rid = f"Event:R:{step}"
                add_node(nodes, rid, "Event", label=ra)
                add_edge(edges, agent_red, rid, "PERFORMS")
                # Technique
                tech_r = ATTACK_MAP.get(ra, "UNK")
                tid_r = f"ATTACK:{tech_r}"
                add_node(nodes, tid_r, "ATTACK", label=tech_r)
                add_edge(edges, rid, tid_r, "USES_TECHNIQUE")
                # Target
                rttype, rtval = norm_target(rt) if rt else (None, None)
                if rttype and rtval:
                    tnode = f"{rttype}:{rtval}"
                    add_node(nodes, tnode, rttype, label=rtval)
                    add_edge(edges, rid, tnode, "TARGETS")
                # 放入窗口
                win.append({
                    "step": step,
                    "event_id": rid,
                    "type": ra,
                    "target_norm": (rttype, rtval) if (rttype and rtval) else None
                })

            # 蓝方事件
            if ba:
                bid = f"Event:B:{step}"
                add_node(nodes, bid, "Event", label=ba)
                add_edge(edges, agent_blue, bid, "PERFORMS")
                tech_b = ATTACK_MAP.get(ba, "UNK")
                tid_b = f"ATTACK:{tech_b}"
                add_node(nodes, tid_b, "ATTACK", label=tech_b)
                add_edge(edges, bid, tid_b, "USES_TECHNIQUE")
                bttype, btval = norm_target(bt) if bt else (None, None)
                if bttype and btval:
                    tnode = f"{bttype}:{btval}"
                    add_node(nodes, tnode, bttype, label=btval)
                    add_edge(edges, bid, tnode, "TARGETS")

                # 仅在蓝方出现时，向过去窗口中的红方找“同目标”的证据并连 TRIGGERS（严格时间因果）
                if bttype and btval:
                    b_norm = (bttype, btval)
                    for e in reversed(win):  # 近→远
                        if e["step"] < step and e["target_norm"] and match_target(e["target_norm"], b_norm):
                            add_edge(edges, e["event_id"], bid, "TRIGGERS")
                            # 如需只连最近一条，可 break；若希望保留全部证据链，不 break
                            break

    return nodes, edges

def main():
    paths = [p for p in IN_FILES if pathlib.Path(p).exists()]
    if not paths:
        print("No input jsonl found. Edit IN_FILES.")
        sys.exit(1)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    nodes, edges = {}, []
    for p in paths:
        print("Parsing:", p)
        process_file(p, nodes, edges)

    # 写出 CSV（保持你的列结构）
    npath = OUT_DIR / "cskg_nodes.csv"
    epath = OUT_DIR / "cskg_edges.csv"
    with npath.open("w", newline="", encoding="utf-8") as nf:
        w = csv.DictWriter(nf, fieldnames=["id","type","label"])
        w.writeheader()
        for n in nodes.values():
            w.writerow({"id": n["id"], "type": n["type"], "label": n.get("label", n["id"])})

    with epath.open("w", newline="", encoding="utf-8") as ef:
        w = csv.DictWriter(ef, fieldnames=["src","dst","type"])
        w.writeheader()
        for e in edges:
            w.writerow({"src": e["src"], "dst": e["dst"], "type": e["type"]})

    print("✅ Saved:", npath, epath)

if __name__ == "__main__":
    main()
