# scripts/cskg/multi_kb.py
# -*- coding: utf-8 -*-
"""
MultiEnvKB: 管理多个场景的 KnowledgeBridge

- 每个场景一个 KnowledgeBridge + 一份 action_names
- 训练 / eval 时只需要传入 env_name，就能：
    - 取对应场景的 prior_logits（并自动 pad 到全局 action_dim）
    - 做奖励塑形（step_update）
    - 做 episode 重置（reset_episode）

当前支持：
    - from_cyborg_only(...)      # 仅 CybORG（兼容你之前版本）
    - from_env_specs(...)        # 通用多场景（cyborg + ics / lot / robotics）
"""

from __future__ import annotations

import os
import pathlib
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.cskg.reasoner import KnowledgeBridge


@dataclass
class EnvKB:
    """单个场景的一套 KB + 动作名"""
    name: str
    kb: KnowledgeBridge
    action_names: List[str]


class MultiEnvKB:
    """
    管理多场景的 KnowledgeBridge：

    - env_kbs: { "cyborg": EnvKB(.), "ics": EnvKB(.), ... }
    - 只要某个场景没有配置 KB，就自动退化为“纯 PPO”（返回零先验、不做塑形）
    """

    def __init__(self, env_kbs: Dict[str, EnvKB]):
        self.env_kbs: Dict[str, EnvKB] = env_kbs
        # CSKG_DEBUG_ENVS=cyborg,ics 用于在指定场景打印 prior/reward 调试日志
        dbg_envs = os.getenv("CSKG_DEBUG_ENVS", "")
        self.debug_envs = {e.strip() for e in dbg_envs.split(",") if e.strip()}

    # ===== 便捷构造 1：只挂 CybORG，一步到位（保留兼容） =====
    @classmethod
    def from_cyborg_only(
            cls,
            seed_graph_path: str | pathlib.Path,
            cskg_path: str | pathlib.Path,
            cyborg_action_names: List[str],
            recent_steps: int = 10,
    ) -> "MultiEnvKB":
        kb = KnowledgeBridge(
            seed_graph_path=str(seed_graph_path),
            cskg_rules_path=str(cskg_path),
            recent_steps=recent_steps,
        )
        env_kb = EnvKB(
            name="cyborg",
            kb=kb,
            action_names=list(cyborg_action_names),
        )
        return cls({"cyborg": env_kb})

    # ===== 便捷构造 2：通用多场景（cyborg + ics 等） =====
    @classmethod
    def from_env_specs(
            cls,
            env_specs: Dict[str, Dict[str, Any]],
            recent_steps: int = 10,
    ) -> "MultiEnvKB":
        """
        env_specs 结构示例：

        env_specs = {
            "cyborg": {
                "seed_graph": "scripts/configs/seed_graph.json",
                "cskg": "scripts/configs/cskg_cyborg_weak.yaml",
                "action_names": [...],     # 长度 = cyborg 动作空间大小
            },
            "ics": {
                "seed_graph": "scripts/configs/seed_graph_ics.json",
                "cskg": "scripts/configs/cskg_ics_weak.yaml",
                "action_names": [...],     # ICS 的语义动作名列表
            },
            ...
        }
        """
        env_kbs: Dict[str, EnvKB] = {}

        for env_name, spec in env_specs.items():
            try:
                seed_graph_path = spec["seed_graph"]
                cskg_path = spec["cskg"]
                action_names = list(spec["action_names"])

                kb = KnowledgeBridge(
                    seed_graph_path=str(seed_graph_path),
                    cskg_rules_path=str(cskg_path),
                    recent_steps=recent_steps,
                )
                env_kbs[env_name] = EnvKB(
                    name=env_name,
                    kb=kb,
                    action_names=action_names,
                )
                print(
                    f"[MultiEnvKB] 已挂载 KB: env={env_name}, "
                    f"actions={len(action_names)}, seed_graph={seed_graph_path}, cskg={cskg_path}"
                )
            except Exception as e:
                print(f"[MultiEnvKB] ⚠ 初始化 KB 失败, env={env_name}, error={e}")

        return cls(env_kbs)

    # ===== 基本查询 =====
    def has_kb(self, env_name: str) -> bool:
        return env_name in self.env_kbs

    def get_kb(self, env_name: str) -> Optional[KnowledgeBridge]:
        ek = self.env_kbs.get(env_name)
        return ek.kb if ek is not None else None

    def get_action_names(self, env_name: str) -> Optional[List[str]]:
        ek = self.env_kbs.get(env_name)
        return ek.action_names if ek is not None else None

    # ===== 生命周期管理（可选） =====
    def reset_episode(self, env_name: str) -> None:
        """
        某个场景的 episode 重置时，顺带把 KB 的 episode 状态也清零。
        """
        kb = self.get_kb(env_name)
        if kb is not None and hasattr(kb, "reset_episode"):
            kb.reset_episode()

    def update_from_facts(self, env_name: str, facts: Dict[str, Any]) -> None:
        """
        如果 KB 支持基于 facts 做持续更新，可以统一在这里调用。
        """
        kb = self.get_kb(env_name)
        if kb is not None and hasattr(kb, "update_from_facts"):
            kb.update_from_facts(facts)

    # ===== 先验 logits（用于 soft prior） =====
    def prior_logits(
            self,
            env_name: str,
            facts: Dict[str, Any],
            global_act_dim: int,
    ) -> np.ndarray:
        """
        返回一个长度 = global_act_dim 的 prior 向量：
        - 如果该场景没有 KB，返回全 0
        - 如果该场景只定义了前 K 个动作的 prior，会自动 pad 到 global_act_dim

        你可以在 PPO 里这样用：
            prior = multi_kb.prior_logits(cur_env, facts, act_dim)
            logits = logits + rule_coef * torch.from_numpy(prior).to(device)
        """
        ek = self.env_kbs.get(env_name)
        if ek is None:
            return np.zeros(global_act_dim, dtype=np.float32)

        kb = ek.kb
        act_names = ek.action_names

        active_rules = []
        try:
            prior_ret = kb.prior_logits(facts, act_names)
            # KnowledgeBridge.prior_logits 可能返回 (prior_vec, active_rules)
            if isinstance(prior_ret, (tuple, list)):
                if len(prior_ret) >= 1:
                    prior = prior_ret[0]
                if len(prior_ret) >= 2:
                    active_rules = prior_ret[1] or []
            else:
                prior = prior_ret
            prior = np.asarray(prior, dtype=np.float32)
        except Exception as e:
            print(f"[MultiEnvKB] ⚠ prior_logits 出错, env={env_name}, error={e}")
            return np.zeros(global_act_dim, dtype=np.float32)

        # pad / 截断 到 global_act_dim
        if prior.shape[0] < global_act_dim:
            pad = np.zeros(global_act_dim, dtype=np.float32)
            pad[: prior.shape[0]] = prior
            prior = pad
        elif prior.shape[0] > global_act_dim:
            prior = prior[:global_act_dim]

        if env_name in self.debug_envs:
            if active_rules or np.any(prior != 0):
                names = [r.get("name", "<rule>") for r in active_rules]
                topk = np.argsort(-prior)[:5]
                top_actions = [(act_names[i] if i < len(act_names) else str(i), float(prior[i])) for i in topk if
                               i < len(prior)]
                print(
                    f"[MultiEnvKB][DEBUG prior] env={env_name} rules={names} top={top_actions} facts_keys={list(facts.keys())}"
                )

        return prior

    # ===== 奖励塑形（step_update 封装） =====
    def shape_reward(
            self,
            env_name: str,
            facts: Dict[str, Any],
            action_idx: int,
            env_reward: float,
    ) -> float:
        """
        调用 KB.step_update 做奖励塑形：

        - 如果该场景没有 KB，直接返回 env_reward
        - 如果 action_idx 超出该场景动作名长度，用 "Unknown" 兜底
        """
        ek = self.env_kbs.get(env_name)
        if ek is None:
            return float(env_reward)

        kb = ek.kb
        act_names = ek.action_names

        try:
            if 0 <= action_idx < len(act_names):
                act_name = act_names[action_idx]
            else:
                act_name = "Unknown"
            shaped = kb.step_update(facts, act_name, float(env_reward))
            if env_name in self.debug_envs:
                if shaped != float(env_reward):
                    print(
                        f"[MultiEnvKB][DEBUG reward] env={env_name} action={act_name} env_r={env_reward:.4f} shaped={shaped:.4f}"
                    )
            return float(shaped)
        except Exception as e:
            print(f"[MultiEnvKB] ⚠ shape_reward 出错, env={env_name}, error={e}")
            return float(env_reward)
