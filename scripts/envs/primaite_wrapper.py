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

        # 一定要在任何 reset / _extract_facts 之前挂上 spaces
        self.observation_space = getattr(self.env, "observation_space", None)
        self.action_space = getattr(self.env, "action_space", None)

        # ---- Host / Critical host 信息（供后续更复杂 facts 使用）----
        self.host_names: List[str] = []
        self.critical_hosts = set(self.CRITICAL_BY_SCENARIO.get(self.scenario, set()))
        self._load_hosts_from_config()

        # ---- 动作名（给 CSKG / Debug 用）----
        self.action_dim: int = 0
        self.action_names: List[str] = []

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
        ICS 高级事实解析：
        - nmne_detected / nmne_high
        - traffic_spike
        - node_down / critical_node_down
        - suspicious_activity
        """

        facts = {
            "recent_reward": float(reward),
            "positive_recent_reward": float(reward) > 0.05,

            "suspicious_activity": False,
            "nmne_detected": False,
            "nmne_high": False,
            "traffic_spike": False,
            "node_down": False,
            "critical_node_down": False,
        }

        # 获取 structured obs（PrimAITE 默认是 dict）
        structured = None
        if isinstance(obs_raw, dict):
            structured = obs_raw
        else:
            return facts  # 保险措施

        nmne_values = []
        traffic_values = []
        node_status = {}  # host → status (1=UP, 0=DOWN)

        critical = self.critical_hosts  # PLC / HMI / controller

        def walk(obj, path):
            # 遍历 structured obs
            if isinstance(obj, dict):
                for k, v in obj.items():
                    key = str(k).lower()

                    # --- NMNE ---
                    if key == "nmne" and isinstance(v, dict):
                        for vv in v.values():
                            try:
                                nmne_values.append(int(vv))
                            except:
                                pass

                    # --- traffic load ---
                    elif key == "traffic" and isinstance(v, dict):
                        # 分层结构：protocol → ports → values
                        for proto_vals in v.values():
                            if isinstance(proto_vals, dict):
                                for port_vals in proto_vals.values():
                                    if isinstance(port_vals, dict):
                                        for val in port_vals.values():
                                            try:
                                                traffic_values.append(int(val))
                                            except:
                                                pass
                                    else:
                                        try:
                                            traffic_values.append(int(port_vals))
                                        except:
                                            pass

                    # --- node / NIC 状态 ---
                    elif key in ("operating_status", "nic_status"):
                        try:
                            val = int(v)
                            # 查找 host 名字
                            host = None
                            for h in self.host_names:
                                if h in path:
                                    host = h
                                    break
                            if host:
                                node_status[host] = val
                        except:
                            pass

                    walk(v, path + [key])

            elif isinstance(obj, list):
                for idx, item in enumerate(obj):
                    walk(item, path + [str(idx)])

        walk(structured, [])

        # ----------- 事实判断：NMNE -----------
        if nmne_values:
            facts["nmne_detected"] = any(v > 0 for v in nmne_values)
            facts["nmne_high"] = any(v >= 2 for v in nmne_values)

        # ----------- 事实判断：traffic spike -----------
        if traffic_values:
            facts["traffic_spike"] = any(v >= 2 for v in traffic_values)

        # ----------- 事实判断：节点宕机 -----------
        if node_status:
            down_any = any(v != 1 for v in node_status.values())
            facts["node_down"] = down_any

            # critical node
            crit_down = any(h in critical and node_status[h] != 1 for h in node_status)
            facts["critical_node_down"] = crit_down

        # ----------- suspicious_activity -----------
        facts["suspicious_activity"] = (
                facts["nmne_detected"]
                or facts["traffic_spike"]
                or facts["critical_node_down"]
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
