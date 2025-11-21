# -*- coding: utf-8 -*-
"""协同检索-校验-解释-规划管线。

组件：
- Retriever/Verifier：检索证据并仅输出带引用的 JSON 事实。
- Explainer：将 MSE 事件转为 PolicySpeak 模板。
- Planner：编译成 OpenC2/playbook，使用 Z3 校验约束并输出日志。

所有校验日志写入 ``scripts/logs/``，评测结果写入 ``reports/policiespeak_eval.md``。
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.common.policiespeak import (
    ConstraintSet,
    Z3ConstraintChecker,
    build_policiespeak_template,
    compile_to_openc2,
    validate_policiespeak,
)
LOG_DIR = REPO_ROOT / "scripts" / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
REPORT_PATH = REPO_ROOT / "reports" / "policiespeak_eval.md"
REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)


@dataclass
class Evidence:
    id: str
    statement: str
    source: str
    citation: str
    confidence: float = 0.5

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "statement": self.statement,
            "source": self.source,
            "citation": self.citation,
            "confidence": self.confidence,
        }


class Retriever:
    """简单关键词匹配检索器。"""

    def __init__(self, min_confidence: float = 0.3) -> None:
        self.min_confidence = min_confidence

    def retrieve(self, query: str, knowledge_base: Sequence[Dict[str, Any]]) -> List[Evidence]:
        hits: List[Evidence] = []
        for idx, item in enumerate(knowledge_base):
            score = float(item.get("score", 0))
            if query.lower() not in item.get("text", "").lower():
                continue
            if score < self.min_confidence:
                continue
            ev = Evidence(
                id=item.get("id", f"kb-{idx}"),
                statement=item.get("text", ""),
                source=item.get("source", "kb"),
                citation=item.get("citation", item.get("source", "kb")),
                confidence=score,
            )
            hits.append(ev)
        return hits


class Verifier:
    """根据检索结果生成带引用的 JSON 事实。"""

    def __init__(self, threshold: float = 0.5) -> None:
        self.threshold = threshold

    def verify(self, query: str, evidences: Sequence[Evidence]) -> List[Dict[str, Any]]:
        facts: List[Dict[str, Any]] = []
        for ev in evidences:
            if ev.confidence < self.threshold:
                continue
            facts.append(
                {
                    "fact": ev.statement,
                    "evidence_id": ev.id,
                    "citation": ev.citation,
                    "confidence": ev.confidence,
                    "query": query,
                }
            )
        return facts


class MultiLLMOrchestrator:
    """最小多模型协调器，可插拔多 LLM/模板生成策略。"""

    def __init__(
        self,
        backends: Optional[Dict[str, Any]] = None,
    ) -> None:
        # backends: name -> callable(payload) or callable(mse_event, evidence)
        self.backends = backends or {
            "rule_based": self._rule_based_template,
        }

    def _rule_based_template(self, mse_event: Dict[str, Any], evidence: List[Dict[str, Any]]) -> Dict[str, Any]:
        return build_policiespeak_template(mse_event, evidence)

    def generate_templates(
        self,
        mse_event: Dict[str, Any],
        evidence: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        outputs: List[Dict[str, Any]] = []
        for name, fn in self.backends.items():
            try:
                tmpl = fn(mse_event, evidence)
                if tmpl:
                    metadata = tmpl.setdefault("metadata", {}) if isinstance(tmpl, dict) else {}
                    if isinstance(metadata, dict):
                        metadata["llm_backend"] = name
                    outputs.append(tmpl)
            except Exception as exc:  # pragma: no cover - defensive
                outputs.append({
                    "llm_backend": name,
                    "error": str(exc),
                })
        return outputs


class Explainer:
    """将 MSE 事件转 PolicySpeak，支持多模型协调。"""

    def __init__(
        self,
        constraint_set: Optional[ConstraintSet] = None,
        orchestrator: Optional[MultiLLMOrchestrator] = None,
    ) -> None:
        self.constraint_set = constraint_set or ConstraintSet()
        self.orchestrator = orchestrator or MultiLLMOrchestrator()

    def explain(
        self,
        mse_events: Sequence[Dict[str, Any]],
        evidence: Sequence[Evidence],
        *,
        training_refs: Optional[Sequence[str]] = None,
    ) -> List[Dict[str, Any]]:
        ev_dicts = [ev.to_dict() for ev in evidence]
        policies: List[Dict[str, Any]] = []
        for mse in mse_events:
            templates = self.orchestrator.generate_templates(mse, ev_dicts)
            for template in templates:
                if training_refs:
                    template.setdefault("metadata", {})["training_refs"] = list(training_refs)
                validated = validate_policiespeak(template, constraints=self.constraint_set)
                if validated.ok:
                    policies.append(validated.payload or template)
        return policies


class Planner:
    """编译为 OpenC2/playbook 并用 Z3 校验。"""

    def __init__(self, constraint_set: Optional[ConstraintSet] = None) -> None:
        self.constraint_set = constraint_set or ConstraintSet()
        self.z3_checker = Z3ConstraintChecker()

    def plan(self, policies: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
        plans: List[Dict[str, Any]] = []
        for pol in policies:
            playbook = compile_to_openc2(pol)
            z3_result = self.z3_checker.check(playbook, self.constraint_set)
            playbook["z3"] = z3_result
            plans.append(playbook)
        return plans


@dataclass
class GenerationRecord:
    fact: str
    evidence_id: Optional[str]
    confidence: float
    label: Optional[int] = None
    run_id: Optional[str] = None


class EvaluationReporter:
    """评估 Hallucination@0、Calibration-E 与一致性。"""

    def __init__(self) -> None:
        self.records: List[GenerationRecord] = []

    def add(self, fact: Dict[str, Any], label: Optional[int] = None, run_id: Optional[str] = None) -> None:
        self.records.append(
            GenerationRecord(
                fact=fact.get("fact", ""),
                evidence_id=fact.get("evidence_id"),
                confidence=float(fact.get("confidence", 0.0)),
                label=label,
                run_id=run_id,
            )
        )

    def hallucination_at_0(self) -> float:
        if not self.records:
            return 0.0
        missing = [r for r in self.records if not r.evidence_id]
        return len(missing) / len(self.records)

    def calibration_e(self) -> Dict[str, float]:
        labeled = [r for r in self.records if r.label is not None]
        if not labeled:
            return {"brier": math.nan, "nll": math.nan}
        brier = sum((r.confidence - r.label) ** 2 for r in labeled) / len(labeled)
        eps = 1e-8
        nll = -sum(
            r.label * math.log(max(eps, min(1 - eps, r.confidence)))
            + (1 - r.label) * math.log(max(eps, min(1 - eps, 1 - r.confidence)))
            for r in labeled
        ) / len(labeled)
        return {"brier": brier, "nll": nll}

    def consistency(self) -> float:
        if not self.records:
            return 1.0
        by_ev: Dict[str, List[str]] = defaultdict(list)
        for r in self.records:
            if r.evidence_id:
                by_ev[r.evidence_id].append(r.fact.strip().lower())
        if not by_ev:
            return 0.0
        ratios: List[float] = []
        for facts in by_ev.values():
            counts = defaultdict(int)
            for f in facts:
                counts[f] += 1
            max_freq = max(counts.values())
            ratios.append(max_freq / len(facts))
        return sum(ratios) / len(ratios)

    def write_report(self, path: Path) -> None:
        cal = self.calibration_e()
        report = ["# PolicySpeak 生成评测", "", f"- Hallucination@0: {self.hallucination_at_0():.4f}", f"- Calibration-E (Brier): {cal.get('brier', float('nan')):.4f}", f"- Calibration-E (NLL): {cal.get('nll', float('nan')):.4f}", f"- 同证据一致性: {self.consistency():.4f}", ""]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(report), encoding="utf-8")


class CollaborativePipeline:
    """打通 Retriever -> Verifier -> Explainer -> Planner 的协同链路。"""

    def __init__(self, llm_backends: Optional[Dict[str, Any]] = None) -> None:
        self.retriever = Retriever()
        self.verifier = Verifier()
        self.explainer = Explainer(orchestrator=MultiLLMOrchestrator(backends=llm_backends))
        self.planner = Planner()
        self.reporter = EvaluationReporter()

    def run(
        self,
        query: str,
        mse_events: Sequence[Dict[str, Any]],
        knowledge_base: Sequence[Dict[str, Any]],
        labels: Optional[Sequence[int]] = None,
        run_id: Optional[str] = None,
        training_refs: Optional[Sequence[str]] = None,
    ) -> Dict[str, Any]:
        evidences = self.retriever.retrieve(query, knowledge_base)
        facts = self.verifier.verify(query, evidences)
        for idx, fact in enumerate(facts):
            label = labels[idx] if labels and idx < len(labels) else None
            self.reporter.add(fact, label=label, run_id=run_id)
        policies = self.explainer.explain(mse_events, evidences, training_refs=training_refs)
        plans = self.planner.plan(policies)
        self.reporter.write_report(REPORT_PATH)
        log_path = LOG_DIR / "collaborative_pipeline.jsonl"
        with log_path.open("a", encoding="utf-8") as w:
            w.write(json.dumps({
                "query": query,
                "facts": facts,
                "policies": policies,
                "plans": plans,
                "run_id": run_id,
                "training_refs": training_refs,
            }, ensure_ascii=False) + "\n")
        return {"facts": facts, "policies": policies, "plans": plans, "report": str(REPORT_PATH)}


def demo() -> None:
    kb = [
        {"id": "ev1", "text": "Service X leaked credentials", "source": "log", "citation": "log://123", "score": 0.9},
        {"id": "ev2", "text": "SSH open on 10.0.0.5", "source": "scan", "citation": "scan://22", "score": 0.7},
    ]
    mse = [{"type": "Contain", "action": "deny", "targets": ["10.0.0.5"], "severity": 6, "summary": "隔离受感染主机"}]

    def gemini_style(mse_event: Dict[str, Any], evidence: List[Dict[str, Any]]) -> Dict[str, Any]:
        # 离线示例：模拟 Gemini 风格输出但不调用外部 API。
        tmpl = build_policiespeak_template(mse_event, evidence)
        tmpl.setdefault("metadata", {})["llm_hint"] = "gemini-offline"
        return tmpl

    llm_backends = {
        "rule_based": lambda mse_evt, ev: build_policiespeak_template(mse_evt, ev),
        "gemini_stub": gemini_style,
    }

    pipeline = CollaborativePipeline(llm_backends=llm_backends)
    result = pipeline.run(
        "credentials",
        mse,
        kb,
        labels=[1],
        training_refs=["finetune_run_v1", "kb_snapshot_2024-06"],
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    demo()