"""Unified environment registry for multi-scenario blue training.

This module exposes factory helpers (``make_cyborg``, ``make_ics`` etc.)
so training scripts can lazily create environments from a single place.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Callable, Dict

from scripts.envs.primaite_wrapper import PrimaiteWrapper

# Inject repo roots so wrappers can locate third_party packages
ROOT = Path(__file__).resolve().parents[2]
CYBORG_SRC = ROOT / "third_party" / "CybORG"
PRIMAITE_SRC = ROOT / "third_party" / "PrimAITE" / "src"
CONFIG_BASE = PRIMAITE_SRC / "primaite" / "config" / "_package_data"

for p in (ROOT, CYBORG_SRC, PRIMAITE_SRC):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

# 🔧 关键：不再提前 find_spec("CybORG")，让 cyborg_wrapper 自己处理 Debugged_CybORG 路径
try:
    from scripts.envs.cyborg_wrapper import CybORGWrapper
    CYBORG_IMPORT_ERROR = None
except Exception as e:  # noqa: BLE001
    CybORGWrapper = None  # type: ignore
    CYBORG_IMPORT_ERROR = e
    print(f"[envs.registry] ⚠ failed to import CybORGWrapper: {e!r}")

# PrimAITE Gym env (optional)
try:
    from primaite.session.environment import PrimaiteGymEnv  # type: ignore
except Exception:
    PrimaiteGymEnv = None  # type: ignore

PROJECT_ROOT = ROOT
PRIMAITE_CFG_DIR = CONFIG_BASE
CYBORG_ENV_CFG = Path(
    os.environ.get(
        "CYBORG_ENV_CFG",
        PROJECT_ROOT / "scripts" / "configs" / "env.yaml",
    )
)


def make_cyborg():
    """Create CybORG Blue environment via CybORGWrapper."""
    if CybORGWrapper is None:
        raise ImportError(
            "CybORGWrapper could not be imported. "
            f"Original error: {CYBORG_IMPORT_ERROR!r}. "
            "Check that scripts/envs/cyborg_wrapper.py exists and that "
            "third_party/CybORG_plus_plus/Debugged_CybORG is correctly set up."
        )
    if not CYBORG_ENV_CFG.exists():
        raise FileNotFoundError(
            f"CybORG config not found: {CYBORG_ENV_CFG}. "
            "Set CYBORG_ENV_CFG to point to your env.yaml or scenario config."
        )
    return CybORGWrapper(str(CYBORG_ENV_CFG))


def _check_primaite():
    if PrimaiteGymEnv is None:
        raise ImportError("PrimAITE sources not found. Ensure third_party/PrimAITE is available.")


def make_ics():
    _check_primaite()
    cfg = PRIMAITE_CFG_DIR / "ics.yaml"
    if not cfg.exists():
        raise FileNotFoundError(f"Scenario config missing: {cfg}")
    return PrimaiteWrapper(str(cfg))


def make_lot():
    _check_primaite()
    cfg = PRIMAITE_CFG_DIR / "lot.yaml"
    if not cfg.exists():
        raise FileNotFoundError(f"Scenario config missing: {cfg}")
    return PrimaiteWrapper(str(cfg))


def make_robotics():
    _check_primaite()
    cfg = PRIMAITE_CFG_DIR / "robotics.yaml"
    if not cfg.exists():
        raise FileNotFoundError(f"Scenario config missing: {cfg}")
    return PrimaiteWrapper(str(cfg))


ENV_REGISTRY: Dict[str, Callable[[], object]] = {
    "cyborg": make_cyborg,
    "ics": make_ics,
    "lot": make_lot,
    "robotics": make_robotics,
}

__all__ = [
    "ENV_REGISTRY",
    "make_cyborg",
    "make_ics",
    "make_lot",
    "make_robotics",
    "PRIMAITE_CFG_DIR",
    "PROJECT_ROOT",
    "PrimaiteWrapper",
]
