
# -*- coding: utf-8 -*-
from pathlib import Path
import os
DEFAULT_ROOT = Path(os.environ.get("CYBDEF_ROOT", r"C:\cybdef"))
LOGS_DIR   = Path(os.environ.get("CYBDEF_LOGS",   str(DEFAULT_ROOT / "logs")))
GRAPH_DIR  = Path(os.environ.get("CYBDEF_GRAPH",  str(DEFAULT_ROOT / "graph")))
REPORTS_DIR= Path(os.environ.get("CYBDEF_REPORTS",str(DEFAULT_ROOT / "reports")))
def ensure_dirs():
    for p in [LOGS_DIR, GRAPH_DIR, REPORTS_DIR]:
        p.mkdir(parents=True, exist_ok=True)
