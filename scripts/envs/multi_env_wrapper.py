# scripts/envs/multi_env_wrapper.py
# -*- coding: utf-8 -*-
"""
MultiEnvWrapper: 统一 cyborg + ics + lot + robotics 多场景的环境包装器

功能：
- 按权重在多个场景中采样 episode（多任务训练）
- 自动探测各场景的 obs_dim / action_dim
- 统一成：
    obs_dim = 所有场景中的最大维度（做 0-padding）
    action_dim = 所有场景中的最大动作数（多余的动作永远 mask=0）
- 对外暴露的接口与单场景 env 尽量一致：
    reset() -> {"obs_vec", "facts", "raw", "env_name"}
    step(a) -> (obs_dict, reward, done, info)
"""

from __future__ import annotations

import os
import random
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from scripts.envs import ENV_REGISTRY


@dataclass
class _EnvSpec:
    name: str
    make_fn: Callable[[], Any]
    obs_dim: int
    act_dim: int
    reward_scale: Optional[float]
    allow_reward_norm: bool = True


class MultiEnvWrapper:
    """
    统一多场景环境为一个 Gym 风格 env，用于单一 PPO 策略多任务训练。
    """

    def __init__(
        self,
        env_names: List[str] | None = None,
        weights: List[float] | None = None,
        mode: str = "train",
        use_semantic_intents: bool = True,
    ):
        """
        :param env_names: 使用的场景列表，默认 ["cyborg", "ics", "lot", "robotics"]
        :param weights:   每个场景被采样的权重，长度与 env_names 对应
        :param mode:      目前只是占位，与 CybORGWrapper 的 mode 语义保持一致
        :env MULTIENV_REWARD_NORM: 控制奖励归一化（none / per_env_max），默认 none。
        注：CybORG 的分值体系本身是负向的，不做归一化处理。
        """
        if env_names is None:
            env_names = ["cyborg", "ics", "lot", "robotics"]

        self.mode = mode
        self.env_specs: Dict[str, _EnvSpec] = {}
        self.env_names: List[str] = []
        self.weights: List[float] = []
        self.use_semantic_intents = use_semantic_intents
        self.reward_norm_mode: str = "none"

        # 支持可选的奖励归一化模式（目前：none / per_env_max）
        # per_env_max：用子环境暴露的 reward_scale 归一化，便于跨场景把「正分 / 负分」拉到同一量级
        self.reward_norm_mode = os.environ.get(
            "MULTIENV_REWARD_NORM", "none"
        ).lower()

        # ---- 探测 & 注册每个场景 ----
        for name in env_names:
            if name not in ENV_REGISTRY:
                raise KeyError(f"未知场景 '{name}'，请检查 ENV_REGISTRY")

            make_fn = ENV_REGISTRY[name]
            obs_dim, act_dim, reward_scale = self._probe_env(make_fn, name)
            allow_norm = name.lower() != "cyborg"
            spec = _EnvSpec(
                name=name,
                make_fn=make_fn,
                obs_dim=obs_dim,
                act_dim=act_dim,
                reward_scale=reward_scale,
                allow_reward_norm=allow_norm,
            )
            self.env_specs[name] = spec
            self.env_names.append(name)

        if weights is None:
            # 默认等权重
            self.weights = [1.0 / len(self.env_names)] * len(self.env_names)
        else:
            assert len(weights) == len(self.env_names), "weights 长度必须与 env_names 一致"
            self.weights = list(weights)

        # 全局统一维度（obs / action）
        self.obs_dim: int = max(s.obs_dim for s in self.env_specs.values())
        self.intent_dim: int = 4 if self.use_semantic_intents else None  # type: ignore
        if self.use_semantic_intents:
            self.action_dim = self.intent_dim  # 统一为 4 个语义意图
        else:
            self.action_dim = max(s.act_dim for s in self.env_specs.values())

        print("🔗 MultiEnvWrapper 初始化完成：")
        for n in self.env_names:
            s = self.env_specs[n]
            print(f"   - {n}: obs_dim={s.obs_dim}, act_dim={s.act_dim}")
        print(f"   => 统一 obs_dim={self.obs_dim}, action_dim={self.action_dim}")

        # 当前激活的子环境
        self._cur_env = None
        self._cur_spec: _EnvSpec | None = None
        self._cur_step = 0
        self._last_mask: np.ndarray | None = None
        self._active_intent_map: List[int] | None = None
        self._active_action_names: List[str] | None = None

    # ---------- 内部工具 ----------

    def _probe_env(
            self, make_fn: Callable[[], Any], env_name: str
    ) -> Tuple[int, int, Optional[float]]:
        """
        启动一个 env 实例，reset 一次，自动探测 obs_dim / act_dim，然后关闭。
        """
        env = make_fn()
        obs_raw = env.reset()
        # 兼容 Gymnasium 风格：(obs, info)
        if isinstance(obs_raw, tuple) and len(obs_raw) == 2:
            obs_raw, _info = obs_raw

        # CybORGWrapper / PrimaiteWrapper 风格：dict 包含 obs_vec
        if isinstance(obs_raw, dict) and "obs_vec" in obs_raw:
            vec = np.asarray(obs_raw["obs_vec"], dtype=np.float32).reshape(-1)
            obs_dim = int(vec.shape[0])
        else:
            vec = np.asarray(obs_raw, dtype=np.float32).reshape(-1)
            obs_dim = int(vec.shape[0])

        # 动作维度：优先用 env.action_dim，其次用 action_space.n
        act_dim = getattr(env, "action_dim", None)
        if act_dim is None:
            space = getattr(env, "action_space", None)
            if space is not None and hasattr(space, "n"):
                act_dim = int(space.n)
            elif isinstance(space, (list, tuple)):
                act_dim = len(space)
            else:
                raise RuntimeError("无法从环境中推断 action_dim")

        act_dim = int(act_dim)

        # 关闭探测用 env
        if hasattr(env, "close"):
            try:
                env.close()
            except Exception:
                pass

        rs_val = getattr(env, "reward_scale", None)
        reward_scale: Optional[float]
        if rs_val is None:
            reward_scale = None
        else:
            try:
                reward_scale = float(rs_val)
            except (TypeError, ValueError):
                reward_scale = None

        return obs_dim, act_dim, reward_scale

    def _pad_obs(self, obs_vec: np.ndarray) -> np.ndarray:
        """
        将各场景的 obs_vec 统一 padding/crop 到 self.obs_dim。
        """
        obs_vec = np.asarray(obs_vec, dtype=np.float32).reshape(-1)
        if obs_vec.size == self.obs_dim:
            return obs_vec
        out = np.zeros(self.obs_dim, dtype=np.float32)
        n = min(self.obs_dim, obs_vec.size)
        out[:n] = obs_vec[:n]
        return out

    # scripts/envs/multi_env_wrapper.py

    # scripts/envs/multi_env_wrapper.py

    def current_action_mask(self) -> np.ndarray:
        """
        返回当前子环境对应的合法动作掩码（长度 = 全局 action_dim）。
        优先使用最近一步缓存的 self._last_mask。
        """
        spec = getattr(self, "_cur_spec", None)

        # 如果已经有缓存的合法掩码，并且有当前场景，就直接用
        if self._last_mask is not None and spec is not None:
            return self._last_mask.copy()

        # 否则就按当前子环境重新推一遍 local mask -> global mask
        mask = np.ones(self.action_dim, dtype=np.float32)  # 默认全 1，避免一开始出错
        if spec is None:
            return mask

        try:
            local_mask = self._get_local_mask()
            global_mask = self._build_global_mask(local_mask, spec.act_dim)
            self._last_mask = global_mask
            return global_mask.copy()
        except Exception:
            # 出问题就退化为「前 act_dim = 1, 后面 = 0」
            fallback = np.zeros(self.action_dim, dtype=np.float32)
            fallback[: spec.act_dim] = 1.0
            self._last_mask = fallback
            return fallback.copy()

    def _build_global_mask(self, local_mask: np.ndarray, local_act_dim: int) -> np.ndarray:
        """
        将子环境的合法掩码扩展为全局掩码：
        - 前 local_act_dim 个 = local_mask
        - 后面补 0
        """
        local_mask = np.asarray(local_mask, dtype=np.float32).reshape(-1)
        g = np.zeros(self.action_dim, dtype=np.float32)
        n = min(self.action_dim, local_act_dim, local_mask.size)
        g[:n] = local_mask[:n]
        # 确保至少有一个动作是合法的
        if g.sum() <= 0:
            g[0] = 1.0
        return g

    def _intent_mask_from_local(
            self, local_mask: np.ndarray, intent_map: List[int], default: float = 0.0
    ) -> np.ndarray:
        """
        将子环境合法掩码映射到语义意图掩码：
        - intent i 对应 local action intent_map[i]
        - 如果 local_mask 不可用则默认全 1
        """

        if local_mask is None:
            return np.ones(len(intent_map), dtype=np.float32) * default

        out = np.zeros(len(intent_map), dtype=np.float32)
        for i, local_idx in enumerate(intent_map):
            if 0 <= local_idx < local_mask.shape[0]:
                out[i] = float(local_mask[local_idx])
            else:
                out[i] = default

        # 至少保留一个动作合法
        if out.sum() <= 0:
            out[0] = 1.0
        return out

    def _build_intent_map(self, env_name: str, env_obj: Any) -> List[int] | None:
        """
        针对不同场景，硬编码 4 个语义动作 -> 具体动作 idx：
        0: Monitor   1: Block   2: Restore   3: Sleep
        """

        if not self.use_semantic_intents:
            return None

        names = self._extract_action_names(env_obj)
        if names:
            def _find(substrs, default_idx: int = 0) -> int:
                for idx, nm in enumerate(names):
                    if any(s in nm for s in substrs):
                        return idx
                return default_idx

            monitor = _find(["investigate", "analyse", "analyze", "monitor", "scan"], 0)
            block = _find(["remove", "block", "isolate", "acl", "deny"], monitor)
            restore = _find(["restore", "startup", "start"], monitor)
            sleep = _find(["sleep", "noop", "none", "idle"], monitor)
            # 同时缓存动作名，方便 step 时打印语义映射
            self._active_action_names = names
            return [monitor, block, restore, sleep]

        # 最后兜底：部分场景保留旧的硬编码索引
        primaite_maps: Dict[str, List[int]] = {
            "ics": [6, 2, 5, 0],
            "lot": [6, 2, 5, 0],
            "robotics": [6, 2, 5, 0],
        }
        return primaite_maps.get(env_name)

    def _extract_action_names(self, env_obj: Any) -> List[str]:
        """尽量从 wrapper / action_space 中提取动作名，用于语义映射。"""

        # 先看 wrapper 是否直接提供了 action_names
        for attr in ("action_names", "get_action_names"):
            if hasattr(env_obj, attr):
                try:
                    names = getattr(env_obj, attr)
                    names = names() if callable(names) else names
                    if isinstance(names, (list, tuple)):
                        return [str(n).lower() for n in names]
                except Exception:
                    pass

        # 其次尝试从 action_space.names 提取
        try:
            names = getattr(getattr(env_obj, "action_space", None), "names", [])
            if names:
                return [str(n).lower() for n in names]
        except Exception:
            pass

        return []

    def _ensure_env(self):
        if self._cur_env is None or self._cur_spec is None:
            raise RuntimeError("当前没有激活的子环境，请先调用 reset()")

    # ---------- 公共属性 ----------

    @property
    def current_env_name(self) -> str | None:
        return self._cur_spec.name if self._cur_spec is not None else None

    # 提供和单场景一样的接口名，方便后面训练脚本用
    @property
    def observation_space(self):
        return {"shape": (self.obs_dim,), "dtype": "float32"}

    # ---------- 接口：reset / step / close ----------

    def reset(self) -> Dict[str, Any]:
        """
        开启一个新的 episode：
        - 按权重随机挑一个场景
        - 构造新的子环境实例
        - 返回统一后的 obs_dict
        """
        # 如果之前有 env，先关掉
        if self._cur_env is not None and hasattr(self._cur_env, "close"):
            try:
                self._cur_env.close()
            except Exception:
                pass

        self._cur_step = 0
        self._last_mask = None

        # 采样场景
        env_name = random.choices(self.env_names, weights=self.weights, k=1)[0]
        spec = self.env_specs[env_name]
        env = spec.make_fn()

        self._active_action_names = self._extract_action_names(env)
        self._active_intent_map = self._build_intent_map(env_name, env)

        self._cur_env = env
        self._cur_spec = spec

        # 调用子环境 reset
        obs_raw = env.reset()
        if isinstance(obs_raw, tuple) and len(obs_raw) == 2:
            obs_raw, _info = obs_raw

        if isinstance(obs_raw, dict) and "obs_vec" in obs_raw:
            vec = np.asarray(obs_raw["obs_vec"], dtype=np.float32).reshape(-1)
            facts = dict(obs_raw.get("facts", {}))
            raw = obs_raw.get("raw", obs_raw)
        else:
            vec = np.asarray(obs_raw, dtype=np.float32).reshape(-1)
            facts = {"recent_reward": 0.0, "has_obs": True}
            raw = obs_raw

        obs_vec = self._pad_obs(vec)

        # 初始化合法掩码
        local_act_dim = spec.act_dim
        local_mask = self._get_local_mask()
        if self._active_intent_map is not None:
            global_mask = self._intent_mask_from_local(local_mask, self._active_intent_map)
        else:
            global_mask = self._build_global_mask(local_mask, local_act_dim)
        self._last_mask = global_mask

        # facts 加上 env_name 信息
        facts.setdefault("env_name", env_name)

        obs = {
            "obs_vec": obs_vec,
            "facts": facts,
            "raw": raw,
            "env_name": env_name,
        }
        return obs

    def _get_local_mask(self) -> np.ndarray:
        """
        尝试从子环境中拿当前 action mask（PrimAITE / CybORG 不同分支）。
        """
        self._ensure_env()
        env = self._cur_env

        # PrimAITE: 有 action_masks() 方法
        if hasattr(env, "action_masks"):
            try:
                m = env.action_masks()
                return np.asarray(m, dtype=np.float32).reshape(-1)
            except Exception:
                pass

        # CybORGWrapper: 有 _current_legal_mask()
        if hasattr(env, "_current_legal_mask"):
            try:
                m = env._current_legal_mask()
                return np.asarray(m, dtype=np.float32).reshape(-1)
            except Exception:
                pass

        # 兜底：全 1
        spec = self._cur_spec
        return np.ones(spec.act_dim, dtype=np.float32)

    def _current_legal_mask(self) -> np.ndarray:
        """
        对外暴露：统一的全局合法掩码。
        PPO 训练脚本可以直接调用 env._current_legal_mask()
        """
        if self._last_mask is None:
            # lazily 再算一次
            local_mask = self._get_local_mask()
            if self._active_intent_map is not None:
                self._last_mask = self._intent_mask_from_local(local_mask, self._active_intent_map)
            else:
                self._last_mask = self._build_global_mask(local_mask, self._cur_spec.act_dim)
        return self._last_mask

    def step(self, action_idx: int):
        """
        执行一步：
        - 将全局动作 idx 映射到子环境 local idx（超出 local_act_dim 的一律当成 0）
        - 调用子环境 step
        - 统一 obs_vec padding 和合法掩码 padding
        """
        self._ensure_env()
        self._cur_step += 1

        env = self._cur_env
        spec = self._cur_spec

        # 确保是 int
        a_global = int(action_idx)
        # 映射到局部动作空间
        intent_map = self._active_intent_map
        mapped_action_name: Optional[str] = None
        intent_name: Optional[str] = None

        if intent_map is not None:
            if a_global < 0 or a_global >= len(intent_map):
                a_local = intent_map[0]
            else:
                a_local = intent_map[a_global]
            if 0 <= a_global < len(self.intent_labels):
                intent_name = self.intent_labels[a_global]
            if self._active_action_names and 0 <= a_local < len(self._active_action_names):
                mapped_action_name = self._active_action_names[a_local]
        else:
            if a_global < 0 or a_global >= spec.act_dim:
                a_local = 0
            else:
                a_local = a_global
            if self._active_action_names and 0 <= a_local < len(self._active_action_names):
                mapped_action_name = self._active_action_names[a_local]

        # 调用子环境 step
        res = env.step(a_local)

        # 兼容两种返回风格：
        # 1) (obs, reward, done, info)
        # 2) (obs, reward, terminated, truncated, info)
        if len(res) == 4:
            obs_raw, reward, done, info = res
        elif len(res) == 5:
            obs_raw, reward, terminated, truncated, info = res
            done = bool(terminated or truncated)
        else:
            raise RuntimeError(f"未知 step 返回格式: {type(res)} len={len(res)}")

        # 统一 obs
        if isinstance(obs_raw, tuple) and len(obs_raw) == 2:
            obs_raw, _info2 = obs_raw

        if isinstance(obs_raw, dict) and "obs_vec" in obs_raw:
            vec = np.asarray(obs_raw["obs_vec"], dtype=np.float32).reshape(-1)
            facts = dict(obs_raw.get("facts", {}))
            raw = obs_raw.get("raw", obs_raw)
        else:
            vec = np.asarray(obs_raw, dtype=np.float32).reshape(-1)
            facts = {"recent_reward": float(reward), "has_obs": True}
            raw = obs_raw

        obs_vec = self._pad_obs(vec)
        facts.setdefault("env_name", spec.name)

        # 更新合法掩码
        local_mask = info.get("legal_mask", self._get_local_mask())
        if intent_map is not None:
            global_mask = self._intent_mask_from_local(local_mask, intent_map)
        else:
            global_mask = self._build_global_mask(local_mask, spec.act_dim)
        self._last_mask = global_mask

        info_out = dict(info)
        info_out["env_name"] = spec.name
        info_out["legal_mask"] = global_mask
        info_out["local_act_dim"] = spec.act_dim
        info_out["local_action_idx"] = int(a_local)
        info_out["global_action_idx"] = int(a_global)
        if intent_name is not None:
            info_out["intent_name"] = intent_name
        if mapped_action_name is not None:
            info_out["mapped_action_name"] = mapped_action_name

        reward = float(reward)
        info_out["raw_reward"] = reward

        if (
                self.reward_norm_mode == "per_env_max"
                and spec.allow_reward_norm
                and spec.reward_scale
        ):
            reward = reward / max(spec.reward_scale, 1e-6)
            info_out["reward_scale"] = spec.reward_scale
            info_out["normalized_reward"] = reward

    def close(self):
        if self._cur_env is not None and hasattr(self._cur_env, "close"):
            try:
                self._cur_env.close()
            except Exception:
                pass
        self._cur_env = None
        self._cur_spec = None
        self._last_mask = None
        self._cur_step = 0
        self._active_intent_map = None
