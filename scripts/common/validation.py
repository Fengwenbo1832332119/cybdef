"""
Utilities for MSE approximation and PolicySpeak validation.

The MSE approximation module mimics different strategies (greedy selection,
gradient-aware masking, attention-thresholding) to align action priors with
KnowledgeBridge outputs. PolicySpeakValidator ensures the produced
explanations follow a JSON schema and remain aligned with KnowledgeBridge
rules/evidence identifiers.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

jsonschema_spec = importlib.util.find_spec("jsonschema")
if jsonschema_spec is not None:
    import jsonschema  # type: ignore
else:
    jsonschema = None

# JSON Schemas for downstream consumers.
MSE_SCHEMA: Dict[str, Any] = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "title": "MSEApproximation",
    "type": "object",
    "properties": {
        "target_mask": {"type": "array", "items": {"type": "number"}},
        "prediction": {"type": "array", "items": {"type": "number"}},
        "mode": {"type": "string", "enum": ["greedy", "gradient_mask", "attention_threshold"]},
        "attention": {"type": ["array", "null"], "items": {"type": "number"}},
        "mse": {"type": "number"},
        "evidence_ids": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["target_mask", "prediction", "mode", "mse"],
}

POLICYSPEAK_SCHEMA: Dict[str, Any] = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "title": "PolicySpeakStatement",
    "type": "object",
    "properties": {
        "statement": {"type": "string"},
        "action": {"type": "string"},
        "evidence_id": {"type": "string"},
        "violation": {"type": "boolean"},
        "justification": {"type": "string"},
    },
    "required": ["statement", "action", "evidence_id", "violation"],
}


def _maybe_validate(payload: Dict[str, Any], schema: Dict[str, Any]) -> None:
    """Validate payload with jsonschema if available."""

    if jsonschema is None:
        return
    jsonschema.validate(instance=payload, schema=schema)


@dataclass
class MSEApproximationResult:
    mse: float
    evidence_ids: List[str]
    mode: str


class MSEApproximation:
    """Approximate rule-based masks with neural outputs and track evidence IDs."""

    def __init__(self, default_mode: str = "greedy") -> None:
        self.default_mode = default_mode

    @staticmethod
    def _normalize(arr: np.ndarray) -> np.ndarray:
        if arr.size == 0:
            return arr
        s = float(np.abs(arr).sum())
        return arr if s == 0.0 else arr / s

    def _evidence_ids(self, target_mask: Sequence[float], prediction: Sequence[float]) -> List[str]:
        ids: List[str] = []
        for tgt, pred in zip(target_mask, prediction):
            raw = f"{tgt:.6f}|{pred:.6f}".encode()
            ids.append(hashlib.sha256(raw).hexdigest()[:12])
        return ids

    def approximate(
        self,
        target_mask: Sequence[float],
        prediction: Sequence[float],
        mode: Optional[str] = None,
        attention: Optional[Sequence[float]] = None,
        threshold: float = 0.5,
    ) -> MSEApproximationResult:
        mode = mode or self.default_mode
        target_np = np.asarray(target_mask, dtype=np.float32).reshape(-1)
        pred_np = np.asarray(prediction, dtype=np.float32).reshape(-1)

        if target_np.shape != pred_np.shape:
            raise ValueError(f"Shape mismatch: {target_np.shape} vs {pred_np.shape}")

        target_norm = self._normalize(target_np)
        pred_norm = self._normalize(pred_np)

        if mode == "gradient_mask" and attention is not None:
            attn = np.asarray(attention, dtype=np.float32).reshape(-1)
            attn = self._normalize(attn)
            pred_norm = pred_norm * (1.0 + attn)
        elif mode == "attention_threshold" and attention is not None:
            attn = np.asarray(attention, dtype=np.float32).reshape(-1)
            attn_mask = (attn >= threshold).astype(np.float32)
            pred_norm = pred_norm * attn_mask
        elif mode == "greedy":
            top_idx = int(np.argmax(pred_norm))
            greedy = np.zeros_like(pred_norm)
            greedy[top_idx] = 1.0
            pred_norm = greedy

        mse = float(np.mean((target_norm - pred_norm) ** 2))
        evidence_ids = self._evidence_ids(target_mask=target_np, prediction=pred_np)

        result = {
            "target_mask": target_np.tolist(),
            "prediction": pred_np.tolist(),
            "mode": mode,
            "attention": None if attention is None else list(attention),
            "mse": mse,
            "evidence_ids": evidence_ids,
        }
        _maybe_validate(result, MSE_SCHEMA)

        return MSEApproximationResult(mse=mse, evidence_ids=evidence_ids, mode=mode)


class PolicySpeakValidator:
    """Validate and penalize PolicySpeak statements with KnowledgeBridge outputs."""

    def __init__(self, schema: Dict[str, Any] | None = None) -> None:
        self.schema = schema or POLICYSPEAK_SCHEMA

    @staticmethod
    def _rule_fingerprint(kb_explain: Dict[str, Any]) -> str:
        """Derive a deterministic evidence identifier from KnowledgeBridge output."""

        active_rules = kb_explain.get("active_prior_rules", []) + kb_explain.get(
            "active_mask_rules", []
        )
        rule_raw = json.dumps(active_rules, sort_keys=True, ensure_ascii=False).encode()
        return hashlib.sha256(rule_raw).hexdigest()[:12]

    def align_evidence_ids(
        self, kb_explain: Dict[str, Any], statements: Iterable[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Attach deterministic evidence IDs derived from KnowledgeBridge rules."""

        rule_id = self._rule_fingerprint(kb_explain)

        aligned: List[Dict[str, Any]] = []
        for s in statements:
            enriched = dict(s)
            enriched.setdefault("evidence_id", rule_id)
            aligned.append(enriched)
        return aligned

    def validate(self, statements: Iterable[Dict[str, Any]]) -> None:
        for s in statements:
            _maybe_validate(s, self.schema)

    def violation_penalty(
        self, kb_explain: Dict[str, Any], statements: Iterable[Dict[str, Any]]
    ) -> Tuple[float, List[Dict[str, Any]]]:
        """Return penalty (count) for PolicySpeak violations and the checked payload."""

        aligned = self.align_evidence_ids(kb_explain, statements)
        self.validate(aligned)

        violation = 0
        for s in aligned:
            if s.get("violation"):
                violation += 1
            if not s.get("justification"):
                violation += 0.5  # partial penalty for missing justification
        return float(violation), aligned

    def build_statement(
        self,
        action: str,
        violation: bool,
        justification: str = "",
        evidence_hint: str | None = None,
    ) -> Dict[str, Any]:
        """Create a PolicySpeak statement with optional justification/evidence hint."""

        payload = {
            "statement": f"Action `{action}` executed",
            "action": action,
            "violation": bool(violation),
            "justification": justification,
        }
        if evidence_hint:
            payload["evidence_id"] = evidence_hint
        return payload

    def loggable_payload(
        self, kb_explain: Dict[str, Any], statements: Iterable[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Return a structured payload (with schema) ready for JSON logging."""

        _, aligned = self.violation_penalty(kb_explain, statements)
        return {
            "evidence_fingerprint": self._rule_fingerprint(kb_explain),
            "statements": aligned,
        }