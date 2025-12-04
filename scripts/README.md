## Multi-LLM 协同管线概览
- 入口：`scripts/agent/collaborative_pipeline.py` 的 `CollaborativePipeline` 打通了 Retriever → Verifier → Explainer → Planner；运行 `python scripts/agent/collaborative_pipeline.py` 可查看示例。
- 多模型协调：`MultiLLMOrchestrator`（同文件）维护一个 `backends` 映射，在 `generate_templates` 中轮询每个后端并把 backend 名写入 `metadata.llm_backend`，体现「多 LLM 分角色出稿」。
- 角色示例：默认 `rule_based` 后端直接调用模板函数；示例中的 `gemini_stub` 则模拟 Gemini 风格输出并标记 `metadata.llm_hint="gemini-offline"`，便于和真实 API 集成时替换。
- 元数据携带：`Explainer.explain` 会在模板上保留 `metadata.training_refs`，连同 backend 标识一起写入日志与报告，方便追踪训练引用与模型来源。

## 我要改哪里才能接入/替换不同的 LLM？
- **核心位置**：`scripts/agent/collaborative_pipeline.py`。
  - `MultiLLMOrchestrator` 类：在其 `__init__` 里提供 `backends` 字典，形如 `{"backend_name": callable(mse_event, evidence)}`。这里是「多个 LLM 分角色」的挂载点。
  - `demo()` 函数的 `llm_backends` 示例：`gemini_stub` 只是离线占位符，你可以直接把它替换成真实调用器，或新增键值对并传给 `CollaborativePipeline(llm_backends=...)`。
- **推荐替换方式**：
  1. 在 `scripts/agent/collaborative_pipeline.py` 顶部（或你自己新建的模块中）实现一个真实的 backend 函数，例如 `gpt51_backend`。
  2. 在 `llm_backends = {...}` 中把 `"gemini_stub"` 替换成 `"gpt51"` 并指向你的函数，或在调用 `CollaborativePipeline` 时传入自定义 `llm_backends`，无需改动其他逻辑。
  3. 确保 backend 返回的对象符合 `build_policiespeak_template` 的结构（字典，包含 `context/intent/actions/evidence` 等字段）；`MultiLLMOrchestrator` 会自动附加 `metadata.llm_backend` 供日志追踪。

### 示例：把占位符换成真实 GPT-5.1 调用
```python
# 文件：scripts/agent/collaborative_pipeline.py（示例可在 demo() 附近替换）
import os
from some_llm_sdk import LLMClient  # 伪代码，请替换为真实 SDK

def gpt51_backend(mse_event, evidence):
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return None  # 没有密钥时跳过，保持离线可运行
    client = LLMClient(api_key=api_key, model="gpt-5.1")
    prompt = f"Draft a PolicySpeak JSON for: {mse_event}\nEvidence: {evidence}"
    response = client.generate(prompt)
    return response  # 应返回 PolicySpeak 结构化字典

llm_backends = {
    "rule_based": lambda mse_evt, ev: build_policiespeak_template(mse_evt, ev),
    "gpt51": gpt51_backend,  # 用真实调用器替换 gemini_stub
}



### 示例：把占位符换成真实 GPT-5.1 调用
```python
# 文件：scripts/agent/collaborative_pipeline.py（示例可在 demo() 附近替换）
import os
from some_llm_sdk import LLMClient  # 伪代码，请替换为真实 SDK

def gpt51_backend(mse_event, evidence):
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return None  # 没有密钥时跳过，保持离线可运行
    client = LLMClient(api_key=api_key, model="gpt-5.1")
    prompt = f"Draft a PolicySpeak JSON for: {mse_event}\nEvidence: {evidence}"
    response = client.generate(prompt)
    return response  # 应返回 PolicySpeak 结构化字典

llm_backends = {
    "rule_based": lambda mse_evt, ev: build_policiespeak_template(mse_evt, ev),
    "gpt51": gpt51_backend,  # 用真实调用器替换 gemini_stub
}

pipeline = CollaborativePipeline(llm_backends=llm_backends)


## 为什么默认不调用外部 LLM API？可以用 GPT-5.1/Gemini 吗？
- 当前示例运行在离线环境，默认使用 `rule_based` 与 `gemini_stub` 以零成本完成端到端演示，避免因缺少网络或密钥导致报错。
- 你可以随时在 `llm_backends` 中注册真实 GPT-5.1 / Gemini / 其他供应商的 SDK 调用；费用会按供应商计费（tokens/字符），需要准备密钥、额度与网络连通性。缺少上述条件时保持默认离线后端即可。