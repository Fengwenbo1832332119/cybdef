# -*- coding: utf-8 -*-
"""协同检索-校验-解释-规划管线。

组件：
- Retriever/Verifier：检索证据并仅输出带引用的 JSON 事实。
- Explainer：将 MSE 事件转为 PolicySpeak 模板。
- Planner：编译成 OpenC2/playbook，使用 Z3 校验约束并输出日志。

所有校验日志写入 ``scripts/logs/``，评测结果写入 ``reports/policiespeak_eval.md``。
"""

from __future__ import annotations

from importlib import util as importlib_util
import json
import math
import os
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Sequence
from urllib import error as urllib_error, request as urllib_request

_openai_spec = importlib_util.find_spec("openai")
if _openai_spec is not None:
    from openai import OpenAI
else:
    OpenAI = None

# 可直接在此粘贴 chatanywhere/GPT_API_free 的 API Key；默认留空以避免硬编码密钥。
INLINE_CHATANYWHERE_API_KEY = os.getenv("CHATANYWHERE_API_KEY_INLINE", "").strip()
INLINE_CHATANYWHERE_BASE_URL = os.getenv("CHATANYWHERE_BASE_URL_INLINE", "https://api.chatanywhere.tech/v1")

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


class ChatAnywhereLLM:
    """OpenAI 官方 SDK 风格客户端，可调用 chatanywhere/GPT_API_free 暴露的免费额度。"""

    def __init__(self, api_key: str, base_url: str) -> None:
        # ChatAnywhere 兼容 OpenAI Chat Completions 协议；若未安装 openai SDK 则回退到 HTTP 请求。
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.client = OpenAI(api_key=api_key, base_url=self.base_url) if OpenAI else None

    @classmethod
    def from_env(cls) -> Optional["ChatAnywhereLLM"]:
        api_key = os.getenv("CHATANYWHERE_API_KEY") or INLINE_CHATANYWHERE_API_KEY
        if not api_key:
            return None
        base_url = os.getenv("CHATANYWHERE_BASE_URL", INLINE_CHATANYWHERE_BASE_URL)
        return cls(api_key=api_key, base_url=base_url)

    def chat_completion(
        self,
        model: str,
        messages: List[Dict[str, str]],
        temperature: float = 0.0,
    ) -> str:
        if self.client:
            try:
                resp = self.client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                )
            except Exception as exc:  # pragma: no cover - 网络/配额故障兜底
                raise RuntimeError(f"LLM 请求失败: {exc}") from exc
            if not resp.choices:
                raise RuntimeError("LLM 返回空结果")
            return resp.choices[0].message.content or ""

        url = f"{self.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
        }
        req = urllib_request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        try:
            with urllib_request.urlopen(req, timeout=30) as resp:
                raw = resp.read().decode("utf-8")
                data = json.loads(raw)
        except (urllib_error.HTTPError, urllib_error.URLError, TimeoutError, ValueError) as exc:
            raise RuntimeError(f"LLM 请求失败: {exc}") from exc
        choices = data.get("choices", [])
        if not choices:
            raise RuntimeError("LLM 返回空结果")
        return choices[0].get("message", {}).get("content", "")


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


def _parse_policiespeak_from_text(text: str, role_name: str) -> Dict[str, Any]:
    try:
        payload = json.loads(text)
        if isinstance(payload, dict):
            payload.setdefault("metadata", {})["llm_raw"] = "parsed_json"
            return payload
    except Exception:
        pass
    return {
        "context": "mse-derived",
        "intent": "ParseFailed",
        "actions": [],
        "metadata": {"llm_backend": role_name, "llm_raw": text, "parse_error": True},
    }


def make_chatanywhere_backend(
    client: Optional[ChatAnywhereLLM],
    model: str,
    role_name: str,
) -> Any:
    """生成一个调用 chatanywhere 免费额度的 backend；无密钥时回退到 rule-based。"""

    def _backend(mse_event: Dict[str, Any], evidence: List[Dict[str, Any]]) -> Dict[str, Any]:
        if client is None:
            tmpl = build_policiespeak_template(mse_event, evidence)
            tmpl.setdefault("metadata", {})["llm_backend"] = f"{role_name}-offline"
            return tmpl
        messages = [
            {
                "role": "system",
                "content": (
                    f"你是{role_name}，需要输出符合 PolicySpeak schema 的 JSON。"
                    "要求：保持所有 evidence 引用，输出 JSON 而非自然语言。"
                ),
            },
            {
                "role": "user",
                "content": json.dumps({"mse_event": mse_event, "evidence": evidence}, ensure_ascii=False),
            },
        ]
        text = client.chat_completion(model=model, messages=messages, temperature=0.0)
        payload = _parse_policiespeak_from_text(text, role_name)
        payload.setdefault("metadata", {})["llm_backend"] = role_name
        payload["metadata"]["model"] = model
        return payload

    return _backend


def gpt51_backend(mse_event: Dict[str, Any], evidence: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """使用官方 OpenAI SDK 直接调用 GPT-5.1 输出 PolicySpeak JSON。

    - 需要在环境变量中提供 ``OPENAI_API_KEY``。
    - 若未安装 openai 库或缺少密钥，则返回 ``None`` 以便 orchestrator 跳过该 backend。
    """

    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key or OpenAI is None:
        return None

    client = OpenAI(api_key=api_key)
    resp = client.chat.completions.create(
        model="gpt-5.1",
        messages=[
            {"role": "system", "content": "输出 PolicySpeak JSON"},
            {
                "role": "user",
                "content": json.dumps({"事件": mse_event, "证据": evidence}, ensure_ascii=False),
            },
        ],
        temperature=0.0,
    )
    text = resp.choices[0].message.content if resp and resp.choices else ""
    payload = _parse_policiespeak_from_text(text, "gpt51")
    payload.setdefault("metadata", {})["model"] = "gpt-5.1"
    return payload


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
        input_source: Optional[str] = None,
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
                "input_source": input_source,
            }, ensure_ascii=False) + "\n")
        return {
            "facts": facts,
            "policies": policies,
            "plans": plans,
            "human_readable": render_human_summary(plans, source=input_source, training_refs=training_refs),
            "narrative": render_narrative_from_artifacts(log_path, REPORT_PATH),
            "report": str(REPORT_PATH),
        }


def render_human_summary(
    plans: Sequence[Dict[str, Any]],
    *,
    source: Optional[str] = None,
    training_refs: Optional[Sequence[str]] = None,
) -> str:
    lines = ["# 策略官摘要", ""]
    if not plans:
        lines.append("暂无可执行计划。")
        return "\n".join(lines)

    if source:
        lines.append(f"输入来源：{source}")
    if training_refs:
        lines.append("训练参考：" + ", ".join(training_refs))
    if source or training_refs:
        lines.append("")

    # 汇总行动去重，方便人读。
    seen_actions = set()
    summarized_actions: List[str] = []
    for plan in plans:
        for cmd in plan.get("playbook", []):
            action = cmd.get("action", "")
            targets_raw = cmd.get("target")
            targets = ", ".join(targets_raw) if isinstance(targets_raw, list) else str(targets_raw)
            sev = cmd.get("args", {}).get("severity")
            ev = cmd.get("evidence", [])
            key = (action, targets, sev)
            if key in seen_actions:
                continue
            seen_actions.add(key)
            ev_str = ",".join(ev) if isinstance(ev, list) else str(ev)
            summarized_actions.append(f"{action} -> {targets}（severity={sev}，evidence={ev_str}）")

    lines.append("行动要点：")
    for idx, desc in enumerate(summarized_actions, 1):
        lines.append(f"- [{idx}] {desc}")

    constraint_notes: List[str] = []
    for plan in plans:
        z3 = plan.get("z3", {}) or {}
        status = z3.get("status")
        if status:
            issues = "; ".join(z3.get("issues", [])) if isinstance(z3.get("issues"), list) else z3.get("issues", "")
            constraint_notes.append(f"{status}: {issues}")

    if constraint_notes:
        lines.append("")
        lines.append("约束校验：")
        for note in sorted(set(constraint_notes)):
            lines.append(f"- {note}")

    return "\n".join(lines)


def _load_latest_jsonl(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except Exception:
        return None
    for line in reversed(lines):
        line = line.strip()
        if not line:
            continue
        try:
            return json.loads(line)
        except Exception:
            continue
    return None


def render_narrative_from_artifacts(log_path: Path, report_path: Path) -> str:
    """根据最新 JSONL 与评测报告生成自然语言摘要。"""

    entry = _load_latest_jsonl(log_path)
    lines = ["# 协同结果自然语言摘要", ""]
    if entry:
        lines.append(f"查询：{entry.get('query', '')}")
        src = entry.get("input_source")
        if src:
            lines.append(f"输入来源：{src}")
        lines.append(f"事实：{len(entry.get('facts', []))} 条 | 模板：{len(entry.get('policies', []))} 个 | 计划：{len(entry.get('plans', []))} 个")
        training_refs = entry.get("training_refs")
        if training_refs:
            lines.append("训练参考：" + ", ".join(training_refs))

        # 角色到模板的简短归纳
        role_lines: List[str] = []
        for pol in entry.get("policies", []):
            meta = pol.get("metadata", {}) if isinstance(pol, dict) else {}
            backend = meta.get("llm_backend", "")
            intent = pol.get("intent", "")
            role_lines.append(f"{backend} -> {intent}")
        if role_lines:
            lines.append("模板归因：" + "; ".join(role_lines))

        # 提取行动一句话串
        actions: List[str] = []
        for plan in entry.get("plans", []):
            for cmd in plan.get("playbook", []):
                action = cmd.get("action", "")
                targets_raw = cmd.get("target")
                targets = ", ".join(targets_raw) if isinstance(targets_raw, list) else str(targets_raw)
                sev = cmd.get("args", {}).get("severity")
                actions.append(f"{action}@{targets} (sev={sev})")
        if actions:
            lines.append("行动总览：" + "; ".join(sorted(set(actions))))

        # 约束整体提示
        z3_notes = []
        for plan in entry.get("plans", []):
            z3 = plan.get("z3", {}) or {}
            status = z3.get("status")
            if status:
                issues_val = z3.get("issues", [])
                issues = "; ".join(issues_val) if isinstance(issues_val, list) else issues_val
                z3_notes.append(f"{status}: {issues}")
        if z3_notes:
            lines.append("约束校验：" + "; ".join(sorted(set(z3_notes))))
    else:
        lines.append("未找到日志条目。")

    if report_path.exists():
        lines.append("")
        lines.append("## 评测指标")
        try:
            metrics = _parse_metrics(report_path.read_text(encoding="utf-8"))
            if metrics:
                for k, v in metrics.items():
                    lines.append(f"- {k}: {v}")
            else:
                lines.append(report_path.read_text(encoding="utf-8").strip())
        except Exception:
            lines.append("（读取评测报告失败）")
    return "\n".join(lines)


def _parse_metrics(text: str) -> Dict[str, str]:
    metrics: Dict[str, str] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line.startswith("- "):
            continue
        body = line[2:]
        if ":" in body:
            key, val = body.split(":", 1)
            metrics[key.strip()] = val.strip()
    return metrics


def load_live_inputs_from_redteam(
    default_query: str,
    default_kb: Sequence[Dict[str, Any]],
    default_mse: Sequence[Dict[str, Any]],
    log_path: Path = LOG_DIR / "redteam_actions.jsonl",
) -> Dict[str, Any]:
    """从红方落盘的 JSONL 中提取最新 kb/mse/query，否则回退到默认示例。

    约定（容错）：
    - JSONL 每行应包含 ``knowledge_base`` 或 ``kb``（列表），``mse_events`` 或 ``events`` 或 ``actions``（列表），可选 ``query``。
    - 若字段缺失或格式不符，自动回退到默认值，确保 demo 离线可跑。
    """

    entry = _load_latest_jsonl(log_path)
    if not entry:
        return {
            "query": default_query,
            "kb": list(default_kb),
            "mse": list(default_mse),
            "source": "default",
        }

    kb = entry.get("knowledge_base") or entry.get("kb") or entry.get("evidence") or entry.get("facts")
    mse = entry.get("mse_events") or entry.get("events") or entry.get("actions") or entry.get("mse")
    query = entry.get("query") or entry.get("mse_query") or default_query

    def _ensure_list(obj: Any) -> List[Dict[str, Any]]:
        if isinstance(obj, list):
            return [x for x in obj if isinstance(x, dict)]
        if isinstance(obj, dict):
            return [obj]
        return []

    kb_list = _ensure_list(kb)
    mse_list = _ensure_list(mse)
    if not kb_list:
        kb_list = list(default_kb)
    if not mse_list:
        mse_list = list(default_mse)

    return {
        "query": query,
        "kb": kb_list,
        "mse": mse_list,
        "source": log_path.name,
    }


def demo() -> None:
    default_kb = [
        {"id": "ev1", "text": "Service X leaked credentials", "source": "log", "citation": "log://123", "score": 0.9},
        {"id": "ev2", "text": "SSH open on 10.0.0.5", "source": "scan", "citation": "scan://22", "score": 0.7},
    ]
    default_mse = [
        {"type": "Contain", "action": "deny", "targets": ["10.0.0.5"], "severity": 6, "summary": "隔离受感染主机"},
    ]

    live_inputs = load_live_inputs_from_redteam(
        default_query="credentials",
        default_kb=default_kb,
        default_mse=default_mse,
    )
    kb = live_inputs["kb"]
    mse = live_inputs["mse"]
    query = live_inputs["query"]

    chatanywhere_client = ChatAnywhereLLM.from_env()
    llm_backends = {
        "gpt51": gpt51_backend,
        "evidence_officer_deepseek": make_chatanywhere_backend(chatanywhere_client, "deepseek-chat", "证据官"),
        "explainer_gemini": make_chatanywhere_backend(chatanywhere_client, "gemini-pro", "解释官"),
        "planner_gpt": make_chatanywhere_backend(chatanywhere_client, "gpt-4o-mini", "策略官"),
        "rule_based": lambda mse_evt, ev: build_policiespeak_template(mse_evt, ev),
    }

    pipeline = CollaborativePipeline(llm_backends=llm_backends)
    result = pipeline.run(
        query,
        mse,
        kb,
        labels=[1],
        training_refs=["finetune_run_v1", "kb_snapshot_2024-06"],
        input_source=live_inputs.get("source"),
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    demo()