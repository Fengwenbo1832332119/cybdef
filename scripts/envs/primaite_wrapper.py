# scripts/envs/primaite_wrapper.py
# -*- coding: utf-8 -*-
"""Lightweight but PPO-ready adapter for PrimAITE gym environments.

封装目标：
- 对齐 CybORGWrapper 的核心接口，方便 MultiEnvWrapper / PPO 统一处理：
    - 属性：obs_dim / action_dim / action_space.names
    - 方法：
        - reset() -> {"obs_vec", "facts", "raw"}
        - step(action_idx) -> (obs_dict, reward, done, info)
        - _current_legal_mask() -> np.ndarray[float32]  （合法动作=1，其它=0）
- 为后续 CSKG 提供基础 facts：
    - recent_reward / bad_recent_reward / very_bad_recent_reward
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Tuple, Optional, List

import numpy as np
import pathlib
import yaml

from primaite.session.environment import PrimaiteGymEnv


@dataclass
class ActionSpace:
    names: List[str]

    @property
    def n(self) -> int:
        return len(self.names)


class PrimaiteWrapper:
    """
    Thin but PPO-ready wrapper around PrimaiteGymEnv.

    目标：
    - 对齐 CybORGWrapper 的核心接口：
        - obs_dim / action_dim / action_space.names
        - reset() -> {"obs_vec", "facts", "raw"}
        - step(action_idx) -> (obs_dict, reward, done, info)
        - _current_legal_mask() -> np.ndarray[float32]
    - 目前 facts 只提供 reward 相关信息，方便 weak CSKG。
    """

    def __init__(self, config_path: str):
        # 底层原生环境
        self.config_path = str(config_path)
        self.env = PrimaiteGymEnv(self.config_path)

        # 直接从环境拿观测 / 动作空间
        # 这里先从 observation_space 取 shape，后续 _flatten_obs 里会真正展平
        shape = getattr(self.env.observation_space, "shape", None)
        if shape is None:
            raise RuntimeError("PrimaiteGymEnv.observation_space 没有 shape 属性，无法推断 obs_dim")
        self.obs_dim: int = int(np.prod(shape))

        self.action_dim: int = int(self.env.action_space.n)

        # 为了 CSKG，需要一份基于“动作语义”的名称列表
        self.action_space = ActionSpace(names=self._load_action_names())

        # 缓存最近一次合法动作 mask（如果环境支持）
        self._last_mask: Optional[np.ndarray] = None

        print(
            f"[PrimaiteWrapper] init: cfg={self.config_path}, "
            f"obs_dim={self.obs_dim}, action_dim={self.action_dim}"
        )

    # ====== 内部工具 ======

    def _load_action_names(self) -> List[str]:
        """
        从 config YAML 解析动作语义，如果失败则退化为 ['action_0', ...]。

        目前用于 ICS / LOT / Robotics：
        - 在 YAML 的 agents[*] 中找到 team: BLUE 的 agent
        - 读取 agent.action_space.action_map[k].action 作为语义名称
        """
        names: List[str] = []
        default_names = [f"action_{i}" for i in range(self.action_dim)]

        try:
            path = pathlib.Path(self.config_path)
            if not path.exists():
                return default_names

            with path.open("r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f)

            agents = cfg.get("agents", [])
            blue_agent: Optional[Dict[str, Any]] = None
            for a in agents:
                if isinstance(a, dict) and a.get("team") == "BLUE":
                    blue_agent = a
                    break

            if blue_agent is None:
                return default_names

            action_map = (
                blue_agent.get("action_space", {}).get("action_map", {}) or {}
            )
            # action_map 形如 {0: {'action': 'do-nothing', 'options': {...}}, ...}
            for idx in range(self.action_dim):
                entry = action_map.get(idx)
                if isinstance(entry, dict) and "action" in entry:
                    names.append(str(entry["action"]))
                else:
                    names.append(f"action_{idx}")

            return names
        except Exception:
            return default_names

    def _flatten_obs(self, obs_raw: Any) -> np.ndarray:
        """
        将 PrimaiteGymEnv 返回的 obs 展平成 1D np.array(float32)。

        兼容几种情况：
        - Gymnasium: (obs, info) -> 先拆 tuple
        - dict: 优先用 obs_raw["observation"] / ["obs"]
        - 其它：直接 np.asarray 再 reshape
        """
        # 1) 兼容 Gymnasium 风格 (obs, info)
        if isinstance(obs_raw, tuple) and len(obs_raw) == 2:
            obs_raw, _info = obs_raw

        # 2) 处理 dict 结构
        if isinstance(obs_raw, dict):
            if "observation" in obs_raw:
                arr = np.asarray(obs_raw["observation"], dtype=np.float32)
            elif "obs" in obs_raw:
                arr = np.asarray(obs_raw["obs"], dtype=np.float32)
            else:
                # 兜底：把 dict 的 values 拼成一个大数组（不推荐长期用，只做保险）
                try:
                    arr = np.concatenate(
                        [np.asarray(v, dtype=np.float32).reshape(-1) for v in obs_raw.values()]
                    )
                except Exception:
                    arr = np.asarray(list(obs_raw.values()), dtype=np.float32)
        else:
            arr = np.asarray(obs_raw, dtype=np.float32)

        return arr.reshape(-1).astype(np.float32)

    def _current_legal_mask(self) -> np.ndarray:
        """
        返回当前时间步的合法动作 mask（1=合法，0=非法）。

        - 如果底层 env 有 action_masks() 接口，则直接使用
        - 否则默认所有动作都合法（全 1）
        """
        # 优先用底层的 action_masks（PrimAITE 有）
        if hasattr(self.env, "action_masks"):
            try:
                m = self.env.action_masks()
                if m is not None:
                    m = np.asarray(m, dtype=np.float32).reshape(-1)
                    if m.size == self.action_dim:
                        self._last_mask = m
                        return m
            except Exception:
                pass

        # 兜底：全 1
        if self._last_mask is None or self._last_mask.size != self.action_dim:
            self._last_mask = np.ones(self.action_dim, dtype=np.float32)
        return self._last_mask

    def _extract_facts(self, obs_raw: Any, reward: float = 0.0) -> Dict[str, Any]:
        """
        极简版 ICS/LOT/Robotics facts：

        - recent_reward: 最近一步环境 reward
        - bad_recent_reward: recent_reward 明显为负（阈值可以后调整）
        - very_bad_recent_reward: recent_reward 非常差
        """
        r = float(reward)
        facts: Dict[str, Any] = {}
        facts["recent_reward"] = r
        facts["bad_recent_reward"] = r < -0.1
        facts["very_bad_recent_reward"] = r < -0.3

        return facts

    # ====== 对外 Gym 接口 ======

    def reset(self) -> Dict[str, Any]:
        """
        兼容 Gymnasium reset：可能返回 obs 或 (obs, info)。
        """
        obs_raw = self.env.reset()
        # 这里不拆 tuple，交给 _flatten_obs/_extract_facts 统一处理
        obs_vec = self._flatten_obs(obs_raw)
        facts = self._extract_facts(obs_raw, reward=0.0)

        obs = {
            "obs_vec": obs_vec,
            "facts": facts,
            "raw": obs_raw,
        }

        # 每个新 episode 都刷新合法掩码
        self._last_mask = None
        return obs

    def step(self, action_idx: int) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        """
        与 Gym 一致的 step 接口，返回统一 obs_dict。

        兼容两种返回格式：
        - (obs, reward, done, info)
        - (obs, reward, terminated, truncated, info)
        """
        res = self.env.step(int(action_idx))

        if isinstance(res, tuple) and len(res) == 4:
            next_obs_raw, reward, done, info = res
        elif isinstance(res, tuple) and len(res) == 5:
            next_obs_raw, reward, terminated, truncated, info = res
            done = bool(terminated or truncated)
        else:
            raise RuntimeError(f"未知 PrimaiteGymEnv.step 返回格式: type={type(res)}, len={len(res)}")

        obs_vec = self._flatten_obs(next_obs_raw)
        facts = self._extract_facts(next_obs_raw, reward=float(reward))

        obs = {
            "obs_vec": obs_vec,
            "facts": facts,
            "raw": next_obs_raw,
        }

        info = dict(info or {})
        info["legal_mask"] = self._current_legal_mask()

        return obs, float(reward), bool(done), info

    def close(self) -> None:
        if hasattr(self.env, "close"):
            self.env.close()
