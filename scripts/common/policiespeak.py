# -*- coding: utf-8 -*-
"""PolicySpeak schema定义、证据校验与约束验证工具。

- 提供 PolicySpeak JSON Schema 以及轻量校验。
- 支持证据存在性检查（并可自动修复/降级）。
- 将动作编译成 OpenC2/playbook，并可用 Z3 约束求解器校验。
- 所有校验日志写入 ``scripts/logs/policiespeak_validation.log``。
"""

from __future__ import annotations

import json
import logging
import importlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

SCRIPT_ROOT = Path(__file__).resolve().parents[1]
LOG_DIR = SCRIPT_ROOT / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOG_DIR / "policiespeak_validation.log"

_logger = logging.getLogger("policiespeak")
if not _logger.handlers:
    _logger.setLevel(logging.INFO)
    _handler = logging.FileHandler(LOG_FILE, encoding="utf-8")
    _formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    _handler.setFormatter(_formatter)
    _logger.addHandler(_handler)


def policiespeak_schema() -> Dict[str, Any]:
    """返回 PolicySpeak JSON Schema（供外部持久化或展示）。"""
    return {
        "$schema": "http://json-schema.org/draft-07/schema#",
        "type": "object",
        "required": ["context", "intent", "actions"],
        "properties": {
            "context": {"type": "string"},
            "intent": {"type": "string"},
            "actor": {"type": "string"},
            "metadata": {"type": "object"},
            "constraints": {
                "type": "object",
                "properties": {
                    "max_latency_ms": {"type": "integer", "minimum": 0},
                    "require_evidence": {"type": "boolean"},
                    "min_confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                },
            },
            "evidence": {
                "type": "array",
                "items": {
                    "type": "object",
                    "required": ["id", "source", "statement"],
                    "properties": {
                        "id": {"type": "string"},
                        "source": {"type": "string"},
                        "statement": {"type": "string"},
                        "citation": {"type": "string"},
                        "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                    },
                },
            },
            "actions": {
                "type": "array",
                "minItems": 1,
                "items": {
                    "type": "object",
                    "required": ["name", "targets"],
                    "properties": {
                        "name": {"type": "string"},
                        "description": {"type": "string"},
                        "targets": {
                            "type": "array",
                            "minItems": 1,
                            "items": {"type": "string"},
                        },
                        "parameters": {"type": "object"},
                        "evidence": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                        "severity": {"type": "integer", "minimum": 0, "maximum": 10},
                    },
                },
            },
        },
        "additionalProperties": False,
    }


@dataclass
class ConstraintSet:
    """用于 Planner/Z3 的约束集合。"""

    max_actions: int = 10
    min_confidence: float = 0.45
    max_latency_ms: Optional[int] = None
    require_evidence: bool = True
    description: str = "默认约束"


@dataclass
class ValidationResult:
    ok: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    payload: Optional[Dict[str, Any]] = None


def _log(message: str) -> None:
    _logger.info(message)


def _validate_schema(payload: Dict[str, Any], schema: Dict[str, Any]) -> List[str]:
    errors: List[str] = []
    for key in schema.get("required", []):
        if key not in payload:
            errors.append(f"Missing required field: {key}")
    properties = schema.get("properties", {})
    for key, value in payload.items():
        if key not in properties:
            errors.append(f"Unexpected field: {key}")
            continue
        prop = properties[key]
        if prop.get("type") == "array" and not isinstance(value, list):
            errors.append(f"Field '{key}' must be array")
        if prop.get("type") == "object" and not isinstance(value, dict):
            errors.append(f"Field '{key}' must be object")
        if prop.get("type") == "string" and not isinstance(value, str):
            errors.append(f"Field '{key}' must be string")
    return errors


def _evidence_index(payload: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    index: Dict[str, Dict[str, Any]] = {}
    for ev in payload.get("evidence", []) or []:
        ev_id = ev.get("id")
        if ev_id:
            index[ev_id] = ev
    return index


def validate_policiespeak(
    payload: Dict[str, Any],
    *,
    auto_fix: bool = True,
    constraints: Optional[ConstraintSet] = None,
) -> ValidationResult:
    """校验 PolicySpeak 文档，必要时自动修复/降级。

    - 若缺少 evidence 引用，将根据 auto_fix 删除引用并记录 warning。
    - 若未满足约束，尽量调整 severity/confidence 以降级，并输出 warning。
    """

    schema = policiespeak_schema()
    errors = _validate_schema(payload, schema)
    warnings: List[str] = []

    evidence_idx = _evidence_index(payload)
    for action in payload.get("actions", []) or []:
        evid_refs = action.get("evidence") or []
        missing = [e for e in evid_refs if e not in evidence_idx]
        if missing:
            msg = f"Action '{action.get('name')}' 引用了缺失证据: {missing}"
            if auto_fix:
                action["evidence"] = [e for e in evid_refs if e in evidence_idx]
                warnings.append(msg + " -> 已降级为移除缺失引用")
            else:
                errors.append(msg)

    constraint_set = constraints or ConstraintSet()
    # 证据存在性/可信度约束
    if constraint_set.require_evidence:
        for idx, action in enumerate(payload.get("actions", [])):
            if not action.get("evidence"):
                msg = f"Action[{idx}] 缺少证据引用"
                if auto_fix:
                    action["evidence"] = []
                    warnings.append(msg + " -> 自动降级但标记为空列表")
                else:
                    errors.append(msg)

    for ev in payload.get("evidence", []) or []:
        conf = ev.get("confidence")
        if conf is not None and conf < constraint_set.min_confidence:
            msg = f"Evidence {ev.get('id')} 置信度 {conf} 低于阈值 {constraint_set.min_confidence}"
            if auto_fix:
                ev["confidence"] = max(conf, constraint_set.min_confidence)
                warnings.append(msg + " -> 自动抬升至阈值")
            else:
                errors.append(msg)

    for action in payload.get("actions", []) or []:
        sev = action.get("severity")
        if sev is None:
            continue
        if not 0 <= int(sev) <= 10:
            msg = f"Action '{action.get('name')}' 的 severity={sev} 超出 [0,10]"
            if auto_fix:
                action["severity"] = min(10, max(0, int(sev)))
                warnings.append(msg + " -> 自动裁剪")
            else:
                errors.append(msg)

    if constraint_set.max_latency_ms is not None:
        latency = payload.get("constraints", {}).get("max_latency_ms")
        if latency and latency > constraint_set.max_latency_ms:
            msg = f"max_latency_ms={latency} 超出约束 {constraint_set.max_latency_ms}"
            if auto_fix:
                payload.setdefault("constraints", {})["max_latency_ms"] = constraint_set.max_latency_ms
                warnings.append(msg + " -> 自动降级至约束值")
            else:
                errors.append(msg)

    ok = not errors
    res = ValidationResult(ok=ok, errors=errors, warnings=warnings, payload=payload if ok else None)
    _log(json.dumps({
        "ok": ok,
        "errors": errors,
        "warnings": warnings,
        "context": payload.get("context"),
        "intent": payload.get("intent"),
    }, ensure_ascii=False))
    return res


def build_policiespeak_template(mse_event: Dict[str, Any], evidence: List[Dict[str, Any]]) -> Dict[str, Any]:
    """将单条 MSE 事件转为 PolicySpeak 模板。"""
    ev_ids = [ev.get("id") for ev in evidence if ev.get("id")]
    return {
        "context": mse_event.get("context", "mse-derived"),
        "intent": mse_event.get("type", "Detect"),
        "actor": mse_event.get("actor", "agent"),
        "constraints": {
            "max_latency_ms": mse_event.get("deadline_ms", 5000),
            "require_evidence": True,
            "min_confidence": 0.5,
        },
        "evidence": evidence,
        "actions": [
            {
                "name": mse_event.get("action", mse_event.get("type", "")),
                "description": mse_event.get("summary", ""),
                "targets": mse_event.get("targets", []),
                "parameters": mse_event.get("parameters", {}),
                "evidence": ev_ids,
                "severity": mse_event.get("severity", 5),
            }
        ],
    }


def compile_to_openc2(payload: Dict[str, Any]) -> Dict[str, Any]:
    """简化的 OpenC2/playbook 结构。"""
    commands = []
    for action in payload.get("actions", []):
        commands.append(
            {
                "action": action.get("name"),
                "target": action.get("targets", []),
                "actuator": payload.get("actor", "agent"),
                "args": {
                    **(action.get("parameters", {})),
                    "severity": action.get("severity", 5),
                },
                "evidence": action.get("evidence", []),
            }
        )
    return {
        "playbook": commands,
        "intent": payload.get("intent"),
        "context": payload.get("context"),
    }


class Z3ConstraintChecker:
    """封装 Z3 约束校验，若环境缺失则自动降级。"""

    def __init__(self) -> None:
        self.z3 = None
        try:
            self.z3 = importlib.import_module("z3")
        except Exception:
            self.z3 = None

    def check(self, plan: Dict[str, Any], constraint_set: ConstraintSet) -> Dict[str, Any]:
        if self.z3 is None:
            msg = "z3 未安装，使用软校验降级"
            _log(msg)
            return {"status": "degraded", "issues": [msg], "satisfied": False}

        z3 = self.z3
        solver = z3.Solver()
        for idx, cmd in enumerate(plan.get("playbook", [])):
            sev = int(cmd.get("args", {}).get("severity", 5))
            sev_var = z3.Int(f"severity_{idx}")
            solver.add(sev_var == sev)
            solver.add(sev_var >= 0, sev_var <= 10)
        if constraint_set.max_latency_ms is not None:
            latency_var = z3.Int("max_latency_ms")
            solver.add(latency_var <= constraint_set.max_latency_ms)
        if constraint_set.max_actions:
            solver.add(z3.Int("actions_count") == len(plan.get("playbook", [])))
            solver.add(z3.Int("actions_count") <= constraint_set.max_actions)

        sat = solver.check() == z3.sat
        issues: List[str] = []
        if not sat:
            issues.append("Z3 constraints unsatisfied")
        _log(json.dumps({"z3": "checked", "sat": sat, "issues": issues}, ensure_ascii=False))
        return {"status": "checked", "issues": issues, "satisfied": sat}