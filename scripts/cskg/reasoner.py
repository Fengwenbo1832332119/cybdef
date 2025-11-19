# scripts/cskg/reasoner.py
# -*- coding: utf-8 -*-
"""
KnowledgeBridge / CSKG 规则引擎（与 cskg.yaml 对齐的版本）

支持的 condition 语法：
- FACT('suspicious_activity')
- NOT FACT('suspicious_activity')
- FACT('recent_reward') < -0.1
- 逻辑组合：AND / OR / NOT

返回内容：
- query_action_mask -> (np.ndarray(mask), active_mask_rules)
- prior_logits      -> (np.ndarray(prior), active_prior_rules)
- step_update       -> shaped_reward
- explain_decision  -> {active_mask_rules, active_prior_rules, recommended_actions}
"""

import json
import re
from typing import Any, Dict, List, Tuple

import numpy as np
import yaml
import pathlib


class KnowledgeBridge:
    def __init__(
        self,
        seed_graph_path: str,
        cskg_rules_path: str,
        recent_steps: int = 10,
    ) -> None:
        self.seed_graph_path = seed_graph_path
        self.cskg_rules_path = cskg_rules_path
        self.recent_steps = recent_steps

        # 规则配置
        self.meta = {}
        self.facts_schema: List[str] = []
        self.actions_cfg: Dict[str, Any] = {}
        self.mask_rules: List[Dict[str, Any]] = []
        self.prior_rules: List[Dict[str, Any]] = []
        self.reward_rules: List[Dict[str, Any]] = []

        # 最近一个 step 的激活规则（方便 explain_decision 用）
        self._last_active_mask_rules: List[Dict[str, Any]] = []
        self._last_active_prior_rules: List[Dict[str, Any]] = []
        self._last_prior_logits: np.ndarray | None = None

        # 简单读取配置
        self._load_rules()

        # 这里暂时不强依赖 seed_graph，先读出来占个位
        self.seed_graph = None
        try:
            p = pathlib.Path(seed_graph_path)
            if p.exists():
                with p.open("r", encoding="utf-8") as f:
                    self.seed_graph = json.load(f)
        except Exception:
            self.seed_graph = None

    # ------------------------------------------------------------------
    # 加载 YAML 规则
    # ------------------------------------------------------------------
    def _load_rules(self) -> None:
        with open(self.cskg_rules_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

        self.meta = cfg.get("meta", {})
        self.facts_schema = cfg.get("facts", [])
        self.actions_cfg = cfg.get("actions", {})

        self.mask_rules = cfg.get("mask_rules", [])
        self.prior_rules = cfg.get("prior_rules", [])
        self.reward_rules = cfg.get("reward_rules", [])

    # ------------------------------------------------------------------
    # 条件解析器：把 DSL 转成 Python 表达式并 eval
    # ------------------------------------------------------------------
    _fact_pattern = re.compile(r"FACT\('([^']+)'\)")

    def _eval_condition(self, cond: str, facts: Dict[str, Any]) -> bool:
        """
        支持：
        - FACT('x')
        - NOT FACT('x')
        - FACT('recent_reward') < -0.1
        - AND / OR / NOT
        """
        if not cond:
            return True

        expr = cond

        # 替换 FACT('xxx') -> 对应的值（可以是 bool / float）
        def _replace_fact(m: re.Match) -> str:
            name = m.group(1)
            val = facts.get(name, False)
            # 直接 repr，让数值比较也能工作
            return repr(val)

        expr = self._fact_pattern.sub(_replace_fact, expr)

        # 逻辑操作符
        expr = expr.replace("AND", "and")
        expr = expr.replace("OR", "or")
        expr = expr.replace("NOT", "not")

        try:
            # 安全 eval：禁用 builtins
            result = eval(expr, {"__builtins__": {}}, {})
            return bool(result)
        except Exception:
            # 出错就当 False，避免把所有规则都触发
            return False

    # ------------------------------------------------------------------
    # 工具：根据动作名称列表，找出匹配某组 actions 的 index
    # ------------------------------------------------------------------
    def _match_action_indices(
        self, action_names: List[str], target_actions: List[str]
    ) -> List[int]:
        """
        target_actions 里的名字是 "RestoreService" / "DecoyApache" 这类；
        action_names 是 env.action_space.names。

        这里采用“前缀匹配 + 全等匹配”的方式，兼容多个 host 上的同类动作：
        例如 target="DecoyApache"，action_names 里有一堆 "DecoyApache", "DecoyApache_0", ...
        """
        target_set = set(target_actions)
        idxs: List[int] = []

        for i, name in enumerate(action_names):
            if name in target_set:
                idxs.append(i)
            else:
                # 兼容类似 "DecoyApache_E1" 这种
                for t in target_set:
                    if name.startswith(t + "_") or name.startswith(t + " "):
                        idxs.append(i)
                        break

        return idxs

    # ------------------------------------------------------------------
    # 1) 动作掩码
    # ------------------------------------------------------------------
    def query_action_mask(
        self, facts: Dict[str, Any], action_names: List[str]
    ) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        """
        返回：
        - mask: np.ndarray, shape [N], 1=允许, 0=禁止
        - active_rules: 实际命中的掩码规则列表
        """
        n = len(action_names)
        mask = np.ones(n, dtype=np.float32)
        active_rules: List[Dict[str, Any]] = []

        for rule in self.mask_rules:
            cond = rule.get("condition", "")
            if not self._eval_condition(cond, facts):
                continue

            effect = rule.get("effect", {})
            rtype = effect.get("type", "hard_mask")
            actions = effect.get("actions", [])
            if not actions:
                continue

            idxs = self._match_action_indices(action_names, actions)
            if not idxs:
                continue

            if rtype == "hard_mask":
                for i in idxs:
                    mask[i] = 0.0
            elif rtype == "soft_mask":
                alpha = float(effect.get("alpha", 0.5))
                for i in idxs:
                    mask[i] *= alpha

            active_rules.append(rule)

        # 记下来，方便 explain_decision 用
        self._last_active_mask_rules = active_rules
        return mask, active_rules

    # ------------------------------------------------------------------
    # 2) 动作先验 logits
    # ------------------------------------------------------------------
    def prior_logits(
        self, facts: Dict[str, Any], action_names: List[str]
    ) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        """
        返回：
        - prior: np.ndarray, shape [N], 对应每个动作的 logits 偏置
        - active_rules: 实际命中的先验规则列表
        """
        n = len(action_names)
        prior = np.zeros(n, dtype=np.float32)
        active_rules: List[Dict[str, Any]] = []

        for rule in self.prior_rules:
            cond = rule.get("condition", "")
            if not self._eval_condition(cond, facts):
                continue

            effect = rule.get("effect", {})
            actions = effect.get("actions", [])
            value = float(effect.get("value", 0.0))

            if not actions or value == 0.0:
                continue

            idxs = self._match_action_indices(action_names, actions)
            if not idxs:
                continue

            for i in idxs:
                prior[i] += value

            active_rules.append(rule)

        self._last_active_prior_rules = active_rules
        self._last_prior_logits = prior.copy()
        return prior, active_rules

    # ------------------------------------------------------------------
    # 3) 奖励塑形
    # ------------------------------------------------------------------
    def step_update(
        self, facts: Dict[str, Any], action_name: str, env_reward: float
    ) -> float:
        """
        输入：
        - facts: 当前 step 的事实
        - action_name: 当前执行的动作名（字符串）
        - env_reward: 环境原始 reward

        输出：
        - shaped_reward: 加上规则塑形后的 reward
        """
        r = float(env_reward)

        for rule in self.reward_rules:
            cond = rule.get("condition", "")
            if cond and not self._eval_condition(cond, facts):
                continue

            effect = rule.get("effect", {})
            # 默认 apply_to_all_actions
            actions = effect.get("actions", None)
            val = float(effect.get("value", 0.0))

            if val == 0.0:
                continue

            if actions:
                # 只在当前动作命中时生效
                idxs = self._match_action_indices([action_name], actions)
                if not idxs:
                    continue

            # type 可以扩展，现在都当 simple add 用
            r += val

        return r

    # ------------------------------------------------------------------
    # 4) 可解释输出
    # ------------------------------------------------------------------
    def explain_decision(
        self,
        facts: Dict[str, Any],
        action_names: List[str],
        top_k: int = 3,
    ) -> Dict[str, Any]:
        """
        根据最近一次 query_action_mask / prior_logits 的记录，给出规则层面的解释。
        """
        # 推荐动作：按 last_prior_logits 排序（如果没有就不给）
        recommended: List[Dict[str, Any]] = []
        if self._last_prior_logits is not None and len(action_names) == len(
            self._last_prior_logits
        ):
            scores = self._last_prior_logits
            idxs = np.argsort(scores)[-top_k:][::-1]
            for i in idxs:
                if float(scores[i]) == 0.0:
                    continue
                recommended.append(
                    {
                        "action": action_names[i],
                        "priority": float(scores[i]),
                    }
                )

        return {
            "active_mask_rules": self._last_active_mask_rules,
            "active_prior_rules": self._last_active_prior_rules,
            "recommended_actions": recommended,
        }
