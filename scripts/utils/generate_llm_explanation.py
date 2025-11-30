# -*- coding: utf-8 -*-
"""
使用 Gemini API 生成 episode 的多视角自然语言总结 + 反事实解释 + MSE 因果解释。

功能：
1) generate_episode_summary_llm(episode, mode)：
   - mode = "tech" / "exec" / "soc"
   - 输出：reports/episode{episode}_explanation_{mode}.md

2) generate_episode_all_views(episode)：
   - 一次性生成 tech / exec / soc 三个视角

3) build_mse_for_episode(episode)：
   - 从 policiespeak_last.json 中抽取每个 Policy 的候选证据
   - 近似构造 MSE（最小证据集）
   - 输出：reports/episode{episode}_mse.json

4) generate_mse_explanation_llm(episode)：
   - 基于 episode{episode}_mse.json + timeline + metrics
   - 输出：reports/episode{episode}_mse_explanation_llm.md

5) generate_counterfactual_llm(episode, step)：
   - 基于 kb_episode{episode}.json 中指定 step 的 facts/rules/blue_action/recommendation
   - 输出：reports/episode{episode}_step{step}_counterfactual.md
"""

import os
import json
from pathlib import Path
import sys
from typing import Any, Dict, List

import google.generativeai as genai

print(f"[INFO] 当前 google-generativeai 版本: {genai.__version__}")

# repo 根路径
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


# ========= 基础工具 =========

def _load_text(path: Path) -> str:
    return path.read_text(encoding="utf-8") if path.exists() else ""


def _load_json(path: Path) -> Any:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _require_gemini() -> None:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("缺少 GEMINI_API_KEY 环境变量！请先在当前终端中设置，例如：\n"
                           '  $env:GEMINI_API_KEY = "your_api_key_here"')
    genai.configure(api_key=api_key)


def _make_prompt_for_mode(
    episode: int,
    timeline_md: str,
    policies: Any,
    metrics_md: str,
    mode: str,
) -> str:
    """根据不同视角（tech/exec/soc）生成差异化 prompt。"""

    base_intro = f"""
你是一名网络安全蓝队/防守方分析专家，正在解读 Episode {episode} 的对抗对局。
下面是本局的结构化信息：
- 时间线（Timeline）：按 step 记录了环境事实、规则触发、推荐动作与蓝方实际动作。
- PolicySpeak JSON：对蓝方策略/动作进行了结构化抽象。
- 评测指标：对解释质量进行了量化评估（如 Hallucination@0、Calibration-E、Consistency 等）。
"""

    if mode == "tech":
        role_spec = """
你的受众是**技术专家 / 研究人员**，希望看到尽可能多的细节，重点包括：
- 攻击链各阶段的演进过程（Recon → Initial Access → Lateral Movement → Impact）。
- 蓝方在关键分叉点的决策逻辑（为何 Investigate / Decoy / Isolation）。
- 规则（CSKG 规则）如何影响策略，例如哪些 FACT 触发了哪些规则。
- 结合 PolicySpeak 与评测指标，对“解释是否忠于证据”给出技术点评。
输出要有：小标题、列表、适合写进技术报告或论文附录。
"""
    elif mode == "exec":
        role_spec = """
你的受众是**管理层 / 安全负责人（CISO/总监）**，他们关心的是：
- 对本局整体风险态势的“鸟瞰视角”。
- 攻击者是否有机会接近关键资产（如 Op_Server0、Enterprise 系统）。
- 蓝方采取的关键决策对“业务风险”有什么影响（降低了多少风险、避免了什么后果）。
- 需要补哪些能力（规则、监控、自动化响应）。
要求：
- 避免过多技术细节和命令名，聚焦“影响”“决策理由”“改进建议”。
- 小段落 + 条理清晰，适合发给领导看。
"""
    else:  # soc
        mode = "soc"
        role_spec = """
你的受众是**SOC 分析师 / 一线安全运营人员**，他们关心：
- 每个关键告警背后对应的证据是什么。
- 分析流程：从告警 → 研判 → 调查 → 隔离/诱捕 的决策链条。
- 哪些 Step 是“必须介入”的高优先级场景。
- 哪些规则/Playbook 对实际处置有帮助，还缺什么触发条件。
要求：
- 输出中要用到类似“当 Step X 出现 Y 证据时 → 我们执行 Z 操作”的格式。
- 更像“实战复盘文档”，供新来的 SOC 同事学习。
"""

    prompt = f"""
{base_intro}

# 时间线（Timeline）
{timeline_md}

# PolicySpeak JSON（结构化策略）
{json.dumps(policies, ensure_ascii=False, indent=2)}

# 评测指标（解释质量评估）
{metrics_md}

{role_spec}

通用要求：
1. 严格基于给定信息，不要虚构不存在的主机、攻击步骤或事件。
2. 输出使用 **中文**，使用 **Markdown 格式**。
3. 结构上建议包含：整体态势概述 → 威胁与防御演进 → 蓝方关键决策解析 → 规则/证据的作用 → 总结与改进建议。
"""

    return prompt


# ========= 1) 多视角 Episode Summary =========

def generate_episode_summary_llm(
    episode: int = 20,
    mode: str = "tech",
    model: str = "models/gemini-flash-latest",
) -> str:
    """
    使用 Gemini 生成 Episode Summary（单一视角）。

    mode:
      - "tech": 技术版
      - "exec": 管理版
      - "soc" : SOC 分析师版
    """

    _require_gemini()

    timeline_path = REPO_ROOT / "reports" / f"episode{episode}_timeline.md"
    policiespeak_path = REPO_ROOT / "reports" / "policiespeak_last.json"
    metrics_path = REPO_ROOT / "reports" / "policiespeak_eval.md"

    timeline_md = _load_text(timeline_path)
    policies = _load_json(policiespeak_path)
    metrics_md = _load_text(metrics_path)

    prompt = _make_prompt_for_mode(
        episode=episode,
        timeline_md=timeline_md,
        policies=policies,
        metrics_md=metrics_md,
        mode=mode,
    )

    model_g = genai.GenerativeModel(model)
    resp = model_g.generate_content(prompt)
    text = resp.text

    suffix = {
        "tech": "tech",
        "exec": "exec",
        "soc": "soc",
    }.get(mode, "tech")

    out_path = REPO_ROOT / "reports" / f"episode{episode}_explanation_{suffix}.md"
    out_path.write_text(text, encoding="utf-8")
    print(f"[INFO] {mode} 视角解释已写入: {out_path}")
    return str(out_path)


def generate_episode_all_views(
    episode: int = 20,
    model: str = "models/gemini-flash-latest",
) -> List[str]:
    """
    一次性生成 tech / exec / soc 三种视角的解释。
    """
    paths: List[str] = []
    for mode in ("tech", "exec", "soc"):
        p = generate_episode_summary_llm(episode=episode, mode=mode, model=model)
        paths.append(p)
    return paths


# ========= 2) MSE（最小证据集）构造 + 因果解释 =========

def build_mse_for_episode(episode: int = 20) -> str:
    """
    从 policiespeak_last.json 中抽取 MSE 近似结果。

    约定：
    - policiespeak_last.json 结构类似：
      {
        "query": "...",
        "facts": [...],
        "policies": [...],
        "plans": [...]
      }
    - evidence 在 policy["evidence"] 中，action["evidence"] 是引用的 id 列表。

    简单启发式：
    - 候选证据 = 被 action 引用到的 evidence id
    - env_facts / cskg_rules 源的证据优先作为“核心因果证据”
    - confidence >= 0.7 更优先
    """

    policiespeak_path = REPO_ROOT / "reports" / "policiespeak_last.json"
    payload = _load_json(policiespeak_path)

    if isinstance(payload, dict) and "policies" in payload:
        policies = payload.get("policies", [])
        evidence_all = payload.get("evidence", []) or payload.get("evidences", [])
    else:
        # 兼容直接就是 policies 列表的情况
        policies = payload if isinstance(payload, list) else []
        evidence_all = []

    # 建立 evidence 索引
    ev_index: Dict[str, Dict[str, Any]] = {}
    if isinstance(evidence_all, list):
        for ev in evidence_all:
            ev_id = ev.get("id")
            if ev_id:
                ev_index[ev_id] = ev

    mse_list: List[Dict[str, Any]] = []

    for idx, pol in enumerate(policies):
        actions = pol.get("actions", []) or []
        actor = pol.get("actor", "agent")
        intent = pol.get("intent", f"policy_{idx}")

        # 收集所有被 action 引用的 evidence id
        ev_ids: List[str] = []
        for act in actions:
            for ev_id in act.get("evidence", []) or []:
                if ev_id not in ev_ids:
                    ev_ids.append(ev_id)

        # 取出具体 evidence 对象
        ev_objs: List[Dict[str, Any]] = []
        for eid in ev_ids:
            if eid in ev_index:
                ev_objs.append(ev_index[eid])

        # 启发式“最小证据集”：优先 env_facts / cskg_rules 且 conf>=0.7
        core: List[Dict[str, Any]] = []
        backup: List[Dict[str, Any]] = []

        for ev in ev_objs:
            src = ev.get("source", "")
            conf = float(ev.get("confidence", 0.0) or 0.0)
            if src in ("env_facts", "cskg_rules") and conf >= 0.7:
                core.append(ev)
            else:
                backup.append(ev)

        mse_evs = core if core else ev_objs  # 至少保证不为空
        mse_ids = [e.get("id") for e in mse_evs if e.get("id")]

        mse_list.append(
            {
                "policy_index": idx,
                "policy_intent": intent,
                "actor": actor,
                "actions": [a.get("name") for a in actions],
                "mse_evidence_ids": mse_ids,
                "mse_evidence": mse_evs,
                "all_evidence_ids": ev_ids,
            }
        )

    out_path = REPO_ROOT / "reports" / f"episode{episode}_mse.json"
    out_path.write_text(json.dumps(mse_list, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[INFO] MSE 近似结果已写入: {out_path}")
    return str(out_path)


def generate_mse_explanation_llm(
    episode: int = 20,
    model: str = "models/gemini-flash-latest",
) -> str:
    """
    基于 episode{episode}_mse.json + timeline + metrics，生成 MSE 因果解释。
    """

    _require_gemini()

    timeline_path = REPO_ROOT / "reports" / f"episode{episode}_timeline.md"
    metrics_path = REPO_ROOT / "reports" / "policiespeak_eval.md"
    mse_path = REPO_ROOT / "reports" / f"episode{episode}_mse.json"

    timeline_md = _load_text(timeline_path)
    metrics_md = _load_text(metrics_path)
    mse_json = _load_json(mse_path)

    prompt = f"""
你是一名可解释强化学习与网络安全的专家。

下面是 Episode {episode} 的信息：
- 时间线（Timeline）：按 step 的事实与动作
{timeline_md}

- 解释质量评估指标：
{metrics_md}

- 针对每条策略/动作抽取的“最小证据集”（MSE）结构化结果：
{json.dumps(mse_json, ensure_ascii=False, indent=2)}

请你完成以下任务：
1. 用通俗但专业的中文说明“最小证据集”的概念，以及它在本 Episode 中有什么作用。
2. 选取 1~3 条代表性的策略（例如包含 Investigate / Decoy / Containment 的动作）：
   - 对每条策略，指出其对应的 MSE 证据有哪些；
   - 解释：如果缺少其中某条证据，这个决策是否仍然合理？为什么？
3. 结合评测指标（Hallucination@0、Calibration-E、Consistency），评价本 Episode 的解释“是否忠于证据”“是否稳定”。
4. 使用 Markdown 格式输出，并适合直接放入论文或技术报告的“解释性分析”章节。
"""

    model_g = genai.GenerativeModel(model)
    resp = model_g.generate_content(prompt)
    text = resp.text

    out_path = REPO_ROOT / "reports" / f"episode{episode}_mse_explanation_llm.md"
    out_path.write_text(text, encoding="utf-8")
    print(f"[INFO] MSE 因果解释已写入: {out_path}")
    return str(out_path)


# ========= 3) 反事实解释（Counterfactual） =========

def generate_counterfactual_llm(
    episode: int,
    step: int,
    model: str = "models/gemini-flash-latest",
) -> str:
    """
    基于 kb_episode{episode}.json 中指定 step 的内容，生成反事实解释。

    约定：
    - kb_episode{episode}.json 的元素中包含字段：
      - episode, step, kind, text, source 等
    - kind:
      - "fact"             环境事实
      - "rule"             规则触发
      - "recommendation"   prior 推荐动作
      - "blue_action"      蓝方实际动作
    """

    _require_gemini()

    kb_path = REPO_ROOT / "reports" / f"kb_episode{episode}.json"
    kb = _load_json(kb_path)
    if not isinstance(kb, list):
        raise RuntimeError(f"kb_episode{episode}.json 结构异常，期望是 list。")

    # 筛选指定 step 的事件
    events = [
        e for e in kb
        if int(e.get("episode", -1)) == episode and int(e.get("step", -1)) == step
    ]

    facts = [e for e in events if e.get("kind") == "fact"]
    rules = [e for e in events if e.get("kind") == "rule"]
    recs = [e for e in events if e.get("kind") == "recommendation"]
    acts = [e for e in events if e.get("kind") == "blue_action"]

    # 粗糙打包给 LLM
    payload = {
        "facts": facts,
        "rules": rules,
        "recommendations": recs,
        "blue_actions": acts,
    }

    prompt = f"""
你是一名网络安全蓝队对抗复盘专家，现在需要对 Episode {episode} 的 Step {step} 做“反事实解释”。

下面是该 Step 的结构化信息（JSON）：
{json.dumps(payload, ensure_ascii=False, indent=2)}

请你完成以下任务：
1. 先用简洁的自然语言，总结这个 Step 里发生了什么：
   - 环境事实（facts）
   - 哪些规则被触发（rules）
   - 系统推荐了哪些动作（recommendations）
   - 蓝方最终采取了什么动作（blue_actions）。

2. 做“反事实”分析：
   a) 假设蓝方 **没有执行任何动作**（即保持不作为），请基于事实与规则，推断可能的风险演化：
      - 攻击者更容易达成哪些目标？
      - 对关键资产（如 Enterprise / Op_Server0）风险如何变化？（定性描述即可）
   b) 如果蓝方采用“推荐动作”中的另一个备选（若存在），可能带来怎样的不同效果？
      - 请比较当前实际动作与备选动作的利弊。

3. 强调“为什么当前决策是合理/必要的”，并尽量把因果逻辑说清楚：
   - 哪些事实 → 触发了哪些规则 → 导致推荐 / 实际动作。

4. 输出使用中文、Markdown 格式，适合作为技术报告中的“反事实分析”小节。
"""

    model_g = genai.GenerativeModel(model)
    resp = model_g.generate_content(prompt)
    text = resp.text

    out_path = REPO_ROOT / "reports" / f"episode{episode}_step{step}_counterfactual.md"
    out_path.write_text(text, encoding="utf-8")
    print(f"[INFO] 反事实解释已写入: {out_path}")
    return str(out_path)
