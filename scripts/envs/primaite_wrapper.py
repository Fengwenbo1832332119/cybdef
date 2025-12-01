# scripts/envs/primaite_wrapper.py
# -*- coding: utf-8 -*-
"""
PrimaiteWrapper: 适配 PrimAITE -> 我们的多环境训练框架

约定：
- 对外暴露 Gym 风格接口：
    reset() -> {"obs_vec", "facts", "raw"}
    step(a) -> (obs_dict, reward, done, info)
- obs_vec 是 float32 的一维向量
- facts 是给 CSKG 用的轻量级状态特征
- raw 保留原始环境 obs（dict 或 ndarray）
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import yaml

from primaite.session.environment import PrimaiteGymEnv


class PrimaiteWrapper:
    """
    薄封装：负责
    - 调用 PrimaiteGymEnv
    - 将 obs flatten 成 obs_vec
    - 从 obs_raw 中提取 ICS 相关的 facts（给 CSKG 用）
    - 暴露 action_masks 以支持合法动作掩码
    """

    # 不同场景下的关键节点（给 future: critical_node_down 用）
    CRITICAL_BY_SCENARIO = {
        "ics": {"ot_gateway", "ot_controller", "plc_node"},
        "lot": {"ot_controller", "plc_node", "ot_gateway"},
        "robotics": {"robot_controller", "robot_safety_plc", "actuator_bus"},
    }

    def __init__(self, config_path: str):
        self.config_path = Path(config_path)
        self.scenario = self.config_path.stem.lower()

        # ---- 创建原始 Primaite 环境 ----
        self.env = PrimaiteGymEnv(str(self.config_path))

        # 原始（未 flatten）观察空间，用于后续解包 obs_raw
        self._raw_observation_space = None
        try:
            self._raw_observation_space = self.env.agent.observation_manager.space
        except Exception:
            self._raw_observation_space = None
        # 一定要在任何 reset / _extract_facts 之前挂上 spaces
        self.observation_space = getattr(self.env, "observation_space", None)
        self.action_space = getattr(self.env, "action_space", None)

        # ---- Host / Critical host 信息（供后续更复杂 facts 使用）----
        self.host_names: List[str] = []
        self.critical_hosts = set(self.CRITICAL_BY_SCENARIO.get(self.scenario, set()))
        self._load_hosts_from_config()
        # 观测里 host 以 HOST0/HOST1... 命名，按照配置顺序做一个别名映射
        self.host_alias: Dict[str, str] = {}
        for idx, name in enumerate(self.host_names):
            alias = f"host{idx}"
            self.host_alias[alias] = name
            self.host_alias[alias.upper()] = name

        # ---- 动作名（给 CSKG / Debug 用）----
        self.action_dim: int = 0
        self.action_names: List[str] = []

        # 先尝试从 config 直接读取蓝方 action_map（即便 gym 的 action_space 没暴露名字，也能复原顺序）
        cfg_action_names = self._load_action_names_from_config()
        if cfg_action_names:
            self.action_names = cfg_action_names
            self.action_dim = len(cfg_action_names)

        if self.action_space is not None and hasattr(self.action_space, "n"):
            self.action_dim = int(self.action_space.n)

        # 有些版本的 Primaite 会在 action_space 里暴露 action_map
        try:
            amap = getattr(self.action_space, "action_map", None)
            if isinstance(amap, dict) and amap:
                # 按 index 排序，提取 "action" 字段
                self.action_names = [str(amap[i]["action"]) for i in sorted(amap.keys())]
                self.action_dim = len(self.action_names)
        except Exception:
            pass

        print(
            f"[PrimaiteWrapper] init: cfg={self.config_path}, "
            f"scenario={self.scenario}, "
            f"obs_dim="
            f"{getattr(getattr(self.observation_space, 'shape', None), '__getitem__', lambda x: 'unknown')(0)}, "
            f"action_dim={self.action_dim}"
        )
        if self.action_names:
            print(f"[PrimaiteWrapper] action_names={self.action_names}")

    # ------------------------------------------------------------------
    #  配置文件中把蓝方能观测到的 host 名字捞出来（后续做更细粒度 facts 可用）
    # ------------------------------------------------------------------
    def _load_hosts_from_config(self) -> None:
        try:
            cfg = yaml.safe_load(self.config_path.read_text(encoding="utf-8"))
        except Exception:
            return

        for agent in cfg.get("agents", []):
            if str(agent.get("team", "")).upper() != "BLUE":
                continue
            obs_cfg = agent.get("observation_space", {}).get("options", {})
            for comp in obs_cfg.get("components", []):
                if comp.get("type") != "nodes":
                    continue
                hosts: Iterable[Dict[str, str]] = comp.get("options", {}).get("hosts", [])
                for h in hosts:
                    name = h.get("hostname")
                    if name:
                        self.host_names.append(name)

    def _load_action_names_from_config(self) -> List[str]:
        """从场景 yaml 解析蓝方 action_map，保持 index 顺序。"""
        try:
            cfg = yaml.safe_load(self.config_path.read_text(encoding="utf-8"))
        except Exception:
            return []

        for agent in cfg.get("agents", []):
            if str(agent.get("team", "")).upper() != "BLUE":
                continue
            action_map = agent.get("action_space", {}).get("action_map", {})
            if not isinstance(action_map, dict):
                continue
            # 按照 index 0..N-1 排序取出 action 名称
            names: List[str] = []
            for idx in sorted(action_map.keys(), key=lambda x: int(x)):
                try:
                    names.append(str(action_map[idx]["action"]))
                except Exception:
                    names.append(str(action_map[idx]))
            if names:
                return names
        return []

    # ------------------------------------------------------------------
    #  obs flatten / facts 提取
    # ------------------------------------------------------------------
    def _flatten_obs(self, obs_raw: Any) -> np.ndarray:
        """把 PrimAITE 的 obs（dict / ndarray）统一成 1D float32 向量。"""
        # 如果环境已经返回 flatten ndarray，就直接用
        if isinstance(obs_raw, np.ndarray):
            return obs_raw.astype(np.float32).reshape(-1)

        # 如果是结构化 dict，且有 observation_space，就用 gymnasium.flatten
        if isinstance(obs_raw, dict) and self.observation_space is not None:
            try:
                from gymnasium.spaces import flatten

                arr = flatten(self.observation_space, obs_raw)
                return np.asarray(arr, dtype=np.float32).reshape(-1)
            except Exception:
                pass

        # 兜底：直接 np.asarray
        arr = np.asarray(obs_raw, dtype=np.float32)
        return arr.reshape(-1)

    def _extract_facts(self, obs_raw: Any, reward: float = 0.0) -> Dict[str, Any]:
        """
        轻量版 ICS facts：
        - recent_reward: 最近一步环境奖励
        - positive_recent_reward: 奖励是否偏正，用于“系统稳定时”的规则
        - suspicious_activity / nmne_* / traffic_spike / node_down / critical_node_down
          尽量从 obs 里推一点（软规则用，不会强约束 PPO）
        """

        def _to_level(val: Any) -> int:
            try:
                return int(val)
            except Exception:
                pass

            if isinstance(val, bool):
                return 1 if val else 0

            if isinstance(val, str):
                v = val.lower()
                if v in {"high", "critical"}:
                    return 3
                if v in {"medium", "med"}:
                    return 2
                if v in {"low", "warn"}:
                    return 1
                if v in {"none", "ok", "normal"}:
                    return 0
            return 0

        facts: Dict[str, Any] = {
            "recent_reward": float(reward),
            "positive_recent_reward": float(reward) > 0.05,
            "negative_recent_reward": float(reward) < -0.05,
            "suspicious_activity": False,
            "nmne_detected": False,
            "nmne_high": False,
            "nmne_medium": False,
            "traffic_spike": False,
            "traffic_tcp_high": False,
            "traffic_udp_high": False,
            "traffic_icmp_high": False,
            "node_down": False,
            "critical_node_down": False,
            # 这些字段目前在 ICS obs 中没有直接对应，先占位为 False，便于规则引用
            "failed_connections": False,
            "failed_requests": False,
            "dos_detected": False,
            "ransomware_detected": False,
            "manipulation_detected": False,
            "env_name": self.scenario,
        }

        # 尝试从 obs 里解析结构
        structured: Optional[Any] = None
        if isinstance(obs_raw, dict):
            structured = obs_raw
        elif isinstance(obs_raw, np.ndarray):
            # 尽量用「未 flatten 前的空间」去解包，才能解析 NMNE / 节点状态等层级字段
            target_space = self._raw_observation_space or self.observation_space
            if target_space is not None:
                try:
                    from gymnasium.spaces import unflatten

                    structured = unflatten(target_space, obs_raw)
                except Exception:
                    structured = None

        nmne_levels: List[int] = []
        traffic_levels: List[int] = []
        traffic_by_proto: Dict[str, List[int]] = {"tcp": [], "udp": [], "icmp": []}
        node_statuses: Dict[str, int] = {}

        def _walk(obj: Any, path: List[str]) -> None:
            if isinstance(obj, dict):
                for k, v in obj.items():
                    key = str(k).lower()
                    next_path = path + [str(k)]

                    # nmne 结构：尝试把 value 统一成等级
                    if key == "nmne" and isinstance(v, dict):
                        for val in v.values():
                            nmne_levels.append(_to_level(val))

                    # traffic 结构：多层 dict，尽量把叶子收集进 traffic_levels
                    if key == "traffic" and isinstance(v, dict):
                        def _collect_traffic(x: Any, proto: Optional[str] = None):
                            if isinstance(x, dict):
                                for kk, vv in x.items():
                                    next_proto = proto
                                    kk_low = str(kk).lower()
                                    if kk_low in traffic_by_proto:
                                        next_proto = kk_low
                                    _collect_traffic(vv, next_proto)
                            else:
                                lvl = _to_level(x)
                                traffic_levels.append(lvl)
                                if proto in traffic_by_proto:
                                    traffic_by_proto[proto].append(lvl)

                        _collect_traffic(v)

                    # 节点状态（operating_status / nic_status 之类）
                    if key in {"operating_status", "nic_status"}:
                        try:
                            status_val = int(v)
                            # 粗暴一点：如果 path 里包含某个 hostname，就记到该 host 上
                            host_hit = next(
                                (
                                    alias
                                    for seg in path + next_path
                                    for alias in (
                                    self.host_alias.get(seg.lower())
                                    or self.host_alias.get(seg),
                                )
                                    if alias
                                ),
                                None,
                            )
                            if host_hit:
                                node_statuses[host_hit] = status_val
                            else:
                                node_statuses.setdefault("*", status_val)
                        except Exception:
                            pass

                    _walk(v, next_path)

            elif isinstance(obj, (list, tuple)):
                for idx, item in enumerate(obj):
                    _walk(item, path + [str(idx)])

        if structured is not None:
            _walk(structured, [])

        # ---- nmne / traffic 逻辑 ----
        facts["nmne_detected"] = any(level > 0 for level in nmne_levels)
        facts["nmne_medium"] = any(level >= 1 for level in nmne_levels)
        facts["nmne_high"] = any(level >= 2 for level in nmne_levels)
        facts["traffic_spike"] = any(level >= 2 for level in traffic_levels)
        facts["traffic_tcp_high"] = any(level >= 2 for level in traffic_by_proto["tcp"])
        facts["traffic_udp_high"] = any(level >= 2 for level in traffic_by_proto["udp"])
        facts["traffic_icmp_high"] = any(level >= 2 for level in traffic_by_proto["icmp"])

        # ---- 节点存活状态 ----
        def _is_down(val: int) -> bool:
            # 这里假设 1 = up，其余都当 down
            return val != 1

        any_down = any(_is_down(v) for v in node_statuses.values())
        facts["node_down"] = any_down

        critical_down = False
        for host, status in node_statuses.items():
            if host in self.critical_hosts and _is_down(status):
                critical_down = True
                break
        facts["critical_node_down"] = critical_down

        # 如果出现 nmne / traffic spike / 关键节点 down，就视为 "suspicious_activity"
        facts["suspicious_activity"] = (
                facts["nmne_detected"]
                or facts["traffic_spike"]
                or critical_down
                or facts["negative_recent_reward"]
        )

        return facts

    # ------------------------------------------------------------------
    #  对外 Gym 风格接口：reset / step / action_masks / close
    # ------------------------------------------------------------------
    def reset(self, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        # PrimAITE Gym 环境是 gymnasium 风格：obs, info
        obs_raw, info = self.env.reset(*args, **kwargs)
        obs_vec = self._flatten_obs(obs_raw)
        facts = self._extract_facts(obs_raw, reward=0.0)

        return {
            "obs_vec": obs_vec,
            "facts": facts,
            "raw": obs_raw,
            "env_name": self.scenario,  # 给 MultiEnvWrapper / MultiEnvKB 用
        }

    def step(self, action: int):
        """
        gymnasium 风格：返回 (obs_dict, reward, done, info)
        供 MultiEnvWrapper 统一处理。
        """
        obs_raw, reward, terminated, truncated, info = self.env.step(action)
        done = bool(terminated or truncated)

        obs_vec = self._flatten_obs(obs_raw)
        facts = self._extract_facts(obs_raw, reward=float(reward))

        obs = {
            "obs_vec": obs_vec,
            "facts": facts,
            "raw": obs_raw,
            "env_name": self.scenario,
        }
        return obs, float(reward), done, info

    def action_masks(self) -> np.ndarray:
        """
        暴露给 MultiEnvWrapper._get_local_mask 使用。
        """
        if hasattr(self.env, "action_masks"):
            try:
                m = self.env.action_masks()
                return np.asarray(m, dtype=np.float32).reshape(-1)
            except Exception:
                pass

        # 没拿到的话，就全 1（全部合法）
        if self.action_dim <= 0:
            return np.array([], dtype=np.float32)
        return np.ones(self.action_dim, dtype=np.float32)

    def close(self) -> None:
        if hasattr(self.env, "close"):
            self.env.close()
