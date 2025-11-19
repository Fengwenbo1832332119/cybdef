# tests/smoke_test_cskg.py
# -*- coding: utf-8 -*-
"""
对 CSKG / KnowledgeBridge 做一次简易 smoke test：

- 使用 CybORGWrapper + KnowledgeBridge
- 每一步：
  1) 用 env._extract_facts() 从 raw_obs 提取 facts
  2) 用 CSKG 计算 action mask + prior logits
  3) 用一个“全 0 未训练策略” + 规则，采样动作
  4) 记录到 policy_*.jsonl，方便你回放 / 排错

重点观察：
- 在 suspicious_activity=False & high_risk_state=False 时，
  Decoy 是否被 mask 掉，Monitor 是否被加先验。
"""

import os
import json
import time
from pathlib import Path
from typing import Any, Dict

import numpy as np

from scripts.envs.cyborg_wrapper import CybORGWrapper
from scripts.cskg.reasoner import KnowledgeBridge


def softmax_masked(logits: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """对带掩码的 logits 做 softmax（mask=0 的动作概率强制为 0）"""
    x = logits - np.max(logits)
    probs = np.exp(x)
    probs = probs * mask
    s = probs.sum()
    if s <= 0:
        # 如果全被乘没了，就均匀分到合法动作上（mask>0）
        msum = mask.sum()
        if msum <= 0:
            return np.ones_like(mask, dtype=np.float32) / len(mask)
        return (mask / msum).astype(np.float32)
    return (probs / s).astype(np.float32)


def to_jsonable(obj: Any):
    """递归把 numpy 类型转成原生 Python，确保可以 JSON 序列化"""
    if isinstance(obj, dict):
        return {k: to_jsonable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [to_jsonable(v) for v in obj]
    elif isinstance(obj, tuple):
        return [to_jsonable(v) for v in obj]  # tuple -> list 也能存
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    else:
        return obj


def main():
    # === 路径与输出 ===
    # 假设本文件在 scripts/tests/ 下，则 parents[1] 是 scripts 目录
    # === 路径与输出 ===
    PROJECT_ROOT = Path(__file__).resolve().parents[2]  # 提升一层，到 C:\cybdef
    CONFIG_DIR = PROJECT_ROOT / "scripts" / "configs"

    ENV_YAML = CONFIG_DIR / "env.yaml"
    CSKG_YAML = CONFIG_DIR / "cskg.yaml"
    SEED_JSON = CONFIG_DIR / "seed_graph.json"

    RUN_DIR = PROJECT_ROOT / "scripts" / "runs" / "smoke"
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    log_path = RUN_DIR / f"policy_{int(time.time())}.jsonl"

    print(f"ENV_YAML = {ENV_YAML}")
    print(f"CSKG_YAML = {CSKG_YAML}")
    print(f"SEED_JSON = {SEED_JSON}")
    print(f"日志输出：{log_path}")

    # === 初始化环境 & KB ===
    env = CybORGWrapper(str(ENV_YAML))
    kb = KnowledgeBridge(
        seed_graph_path=str(SEED_JSON),
        cskg_rules_path=str(CSKG_YAML),
        recent_steps=5,
    )

    # 看看动作空间信息
    action_names = env.action_space.names
    A = len(action_names)
    print(f"🎯 动作空间大小: {A}")
    print("🎯 前 20 个动作名示例:")
    for i, n in enumerate(action_names[:20]):
        print(f"  [{i}] {n}")

    # 开始一个“回合”
    obs_raw = env.reset()
    last_reward_env = 0.0

    f = open(log_path, "w", encoding="utf-8")

    steps = 10  # smoke test 先跑 10 步看一眼
    for t in range(steps):
        # 1) 从 raw obs 提取事实（与训练脚本保持一致）
        facts: Dict[str, Any] = env._extract_facts(obs_raw, reward=last_reward_env)

        # 2) 生成初始 logits（未训练策略：全 0）
        logits = np.zeros(A, dtype=np.float32)

        # 3) 从 KB 拿掩码与先验 logits（新版接口）
        rule_mask, active_mask_rules = kb.query_action_mask(facts, action_names)
        prior, active_prior_rules = kb.prior_logits(facts, action_names)

        # 环境合法掩码
        legal_mask = env._current_legal_mask().astype(np.float32)

        # === 守护性断言 ===
        assert logits.shape[-1] == env.action_dim, "logits 维度与动作空间不一致"
        assert rule_mask.shape[0] == env.action_dim, "mask 维度与动作空间不一致"

        # rule_mask 极端情况为 0：放开一个 no-op（这里假设 Sleep 索引是 0）
        if rule_mask.sum() <= 0:
            rule_mask[0] = 1.0

        # 融合掩码（环境 × 规则）
        combined_mask = (legal_mask * rule_mask).astype(np.float32)
        if combined_mask.sum() <= 0:
            combined_mask[0] = 1.0

        # 4) 融合先验 + 掩码（用 log(mask) 做“半硬”约束）
        mask_alpha = 2.0
        logits = logits + prior
        logits = logits + np.log(np.clip(combined_mask, 1e-6, 1.0)) * mask_alpha

        # 5) 采样动作
        probs = softmax_masked(logits, (combined_mask > 0).astype(np.float32))
        a_idx = int(np.random.choice(A, p=probs))
        a_name = action_names[a_idx]

        # 6) 环境步进
        next_obs_raw, r_env, done, info = env.step(a_idx)

        # 7) 奖励塑形（直接用新版 step_update）
        r_total = kb.step_update(facts, a_name, float(r_env))

        # 更新最近 reward，供下一步 _extract_facts 使用
        last_reward_env = float(r_env)

        # 8) 可解释日志
        explain = {}
        try:
            explain = kb.explain_decision(facts, action_names)
        except Exception:
            pass

        # top_prior：把 prior 的最大 3 个动作用于观测
        top_idx = np.argsort(prior)[-3:][::-1]
        top_prior = [
            (action_names[i], float(prior[i])) for i in top_idx
        ]

        rec = {
            "step": t + 1,
            "action_idx": a_idx,
            "action_name": a_name,
            "reward_env": float(r_env),
            "reward_total": float(r_total),
            "legal_mask_sum": float(legal_mask.sum()),
            "rule_mask_sum": float(rule_mask.sum()),
            "combined_mask_sum": float(combined_mask.sum()),
            "top_prior": top_prior,
            "fact": facts,
            "explain": explain,
        }
        f.write(json.dumps(to_jsonable(rec), ensure_ascii=False) + "\n")
        f.flush()

        obs_raw = next_obs_raw
        if done:
            break

    f.close()
    env.close()
    print(f"✅ Smoke test 完成：日志已写入 {log_path}")


if __name__ == "__main__":
    main()
