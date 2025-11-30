"""Environment wrappers and registry exports."""
from __future__ import annotations

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.envs.primaite_wrapper import PrimaiteWrapper
from scripts.envs.registry import (
    ENV_REGISTRY,
    PROJECT_ROOT,
    PRIMAITE_CFG_DIR,
    make_cyborg,
    make_ics,
    make_lot,
    make_robotics,
)

# 不再用 find_spec("CybORG")，直接尝试 import，
# 具体的 sys.path 注入由 cyborg_wrapper 自己处理
try:
    from scripts.envs.cyborg_wrapper import CybORGWrapper
except Exception as e:  # noqa: BLE001
    print(f"[envs.__init__] ⚠ failed to import CybORGWrapper: {e!r}")
    CybORGWrapper = None  # type: ignore

__all__ = [
    "ENV_REGISTRY",
    "PROJECT_ROOT",
    "PRIMAITE_CFG_DIR",
    "CybORGWrapper",
    "PrimaiteWrapper",
    "make_cyborg",
    "make_ics",
    "make_lot",
    "make_robotics",
]
