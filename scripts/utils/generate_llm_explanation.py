# -*- coding: utf-8 -*-
"""
使用 Gemini API 生成 episode 的人类可读自然语言总结。
"""

import os
import json
from pathlib import Path
import sys
from typing import Any

import google.generativeai as genai

print(f"当前 google-generativeai 版本: {genai.__version__}")

# 临时检查：看看你的 API KEY 能访问哪些模型
# 注意：需要在 configure 之后运行
# genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
# for m in genai.list_models():
#     if 'generateContent' in m.supported_generation_methods:
#         print(m.name)

# repo 根路径
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _load_text(path: Path) -> str:
    return path.read_text(encoding="utf-8") if path.exists() else ""


def _load_json(path: Path) -> Any:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def generate_episode_summary_llm(episode: int = 20, model: str = "models/gemini-flash-latest") -> str:
    """
    使用 Google Gemini API 生成 Episode Summary 的自然语言解释。
    """

    # 载入文件
    kb_path = REPO_ROOT / "reports" / f"kb_episode{episode}.json"
    timeline_path = REPO_ROOT / "reports" / f"episode{episode}_timeline.md"
    policiespeak_path = REPO_ROOT / "reports" / "policiespeak_last.json"
    metrics_path = REPO_ROOT / "reports" / "policiespeak_eval.md"

    timeline_md = _load_text(timeline_path)
    policies = _load_json(policiespeak_path)
    metrics_md = _load_text(metrics_path)

    # 使用 Gemini API Key
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("缺少 GEMINI_API_KEY 环境变量！请设置你的 Gemini API Key。")

    genai.configure(api_key=api_key)

    prompt = f"""
你是一名网络安全蓝队/防守方分析专家。

基于以下结构化信息，为 Episode {episode} 生成一份自然语言总结，适用于技术汇报。

# 时间线（Timeline）
{timeline_md}

# PolicySpeak JSON
{json.dumps(policies, ensure_ascii=False, indent=2)}

# 评测指标（解释质量）
{metrics_md}

要求：
1. 输出 **中文**，使用 **Markdown 格式**。
2. 结构建议：总体态势 → 威胁演进 → 蓝方关键操作 → 规则触发 → 最终评估。
3. 不得虚构任何未出现的事件、主机、攻击步骤。
4. 语言专业、可读性强。
"""

    model_g = genai.GenerativeModel(model)

    response = model_g.generate_content(prompt)

    text = response.text

    # 写入文件
    out_path = REPO_ROOT / "reports" / f"episode{episode}_explanation_llm.md"
    out_path.write_text(text, encoding="utf-8")

    return str(out_path)
