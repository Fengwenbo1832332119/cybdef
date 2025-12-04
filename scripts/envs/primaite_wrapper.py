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

import os
from pathlib import Path
import tempfile
import copy
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import yaml

from primaite import PRIMAITE_CONFIG
from primaite.session.environment import PrimaiteGymEnv
from primaite.utils.cli.primaite_config_utils import update_primaite_application_config


def _to_bool(val: str) -> Optional[bool]:
    """Best-effort string-to-bool converter. Returns None if unknown."""

    v = str(val).strip().lower()
    if v in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if v in {"0", "false", "f", "no", "n", "off"}:
        return False
    return None


def apply_primaite_dev_overrides_from_env() -> None:
    """
    Allow training scripts to toggle PrimAITE developer logging via env vars.

    Supported overrides (name -> PRIMAITE_CONFIG.developer_mode key):
    - PRIMAITE_DEV_MODE      -> enabled
    - PRIMAITE_SYS_LOG_LEVEL -> sys_log_level
    - PRIMAITE_AGENT_LOG_LEVEL -> agent_log_level
    - PRIMAITE_OUTPUT_SYS_LOGS -> output_sys_logs
    - PRIMAITE_OUTPUT_AGENT_LOGS -> output_agent_logs
    - PRIMAITE_OUTPUT_PCAP_LOGS -> output_pcap_logs
    - PRIMAITE_OUTPUT_TERMINAL -> output_to_terminal
    - PRIMAITE_OUTPUT_DIR -> output_dir
    """

    dev_cfg = PRIMAITE_CONFIG.get("developer_mode", {})
    updates: Dict[str, Any] = {}

    toggles = {
        "PRIMAITE_DEV_MODE": "enabled",
        "PRIMAITE_OUTPUT_SYS_LOGS": "output_sys_logs",
        "PRIMAITE_OUTPUT_AGENT_LOGS": "output_agent_logs",
        "PRIMAITE_OUTPUT_PCAP_LOGS": "output_pcap_logs",
        "PRIMAITE_OUTPUT_TERMINAL": "output_to_terminal",
    }

    for env_key, cfg_key in toggles.items():
        if env_key in os.environ:
            parsed = _to_bool(os.environ[env_key])
            if parsed is not None:
                updates[cfg_key] = parsed

    if "PRIMAITE_SYS_LOG_LEVEL" in os.environ:
        updates["sys_log_level"] = os.environ["PRIMAITE_SYS_LOG_LEVEL"].upper()

    if "PRIMAITE_AGENT_LOG_LEVEL" in os.environ:
        updates["agent_log_level"] = os.environ["PRIMAITE_AGENT_LOG_LEVEL"].upper()

    if "PRIMAITE_OUTPUT_DIR" in os.environ:
        updates["output_dir"] = os.environ["PRIMAITE_OUTPUT_DIR"]

    if not updates:
        return

    dev_cfg.update(updates)
    PRIMAITE_CONFIG["developer_mode"] = dev_cfg
    try:
        update_primaite_application_config(config=PRIMAITE_CONFIG)
    except Exception:
        # Do not fail environment construction due to optional logging tweaks
        pass


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
        "ics": {"ot_controller"},
        "lot": {"iot_hub"},
        "robotics": {"robot_controller"},
    }
    def __init__(self, config_path: str, dev_mode_from_env: bool = True):
        self.config_path = Path(config_path)
        self._temp_config_path: Optional[Path] = None
        self.scenario = self.config_path.stem.lower()

        # 4 个统一的语义意图标签，便于单场景 smoke 打印 "Monitor/Block/Restore/Sleep"
        self.intent_labels: List[str] = ["Monitor", "Block", "Restore", "Sleep"]

        # 提前加载配置，后续复用（hosts / action_names / 最大奖励等）
        self._config_cache: Optional[Dict[str, Any]] = None
        try:
            self._config_cache = yaml.safe_load(self.config_path.read_text(encoding="utf-8"))
        except Exception:
            self._config_cache = None

        # 根据环境变量决定是否在 smoke-test 等场景下强行提前红队起手，以便尽快观测扣分
        self._maybe_force_fast_red_start()

        # ---- 创建原始 Primaite 环境 ----
        if dev_mode_from_env:
            apply_primaite_dev_overrides_from_env()

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
        # host alias（lowercase -> canonical），便于在 _extract_facts 里通过路径匹配
        self.host_alias: Dict[str, str] = {}
        for name in self.host_names:
            self.host_alias[name] = name
            self.host_alias[name.lower()] = name

        # ---- 动作名（给 CSKG / Debug 用）----
        self.action_dim: int = 0
        self.action_names: List[str] = []

        # 奖励归一化：计算理论最大奖励（正权重之和），用于跨环境统一尺度
        self.max_reward: float = self._compute_max_reward_from_config()
        self.reward_scale: float = self.max_reward if self.max_reward > 0 else 1.0

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

        if self.max_reward > 0:
            print(
                f"[PrimaiteWrapper] max_reward_from_config={self.max_reward:.4f}"
                f" -> reward_scale={self.reward_scale:.4f}"
            )

    # ------------------------------------------------------------------
    #  配置文件中把蓝方能观测到的 host 名字捞出来（后续做更细粒度 facts 可用）
    # ------------------------------------------------------------------
    def _load_hosts_from_config(self) -> None:
        cfg = self._config_cache
        if not isinstance(cfg, dict):
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
        cfg = self._config_cache
        if not isinstance(cfg, dict):
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

    def _maybe_force_fast_red_start(self) -> None:
        """
        在 smoke test 场景下快速触发红队，以便尽早观察负向奖励。

        触发条件：环境变量 ``PRIMAITE_FORCE_FAST_RED_START`` 为真值时，对配置做临时修改：
        - 所有 Red agent 的 ``start_step`` 置 1、``start_variance`` 置 0、``random_offset_max`` 置 0，确保首步即起手。
        - 将 ``frequency`` 收敛为 1（如果原值缺失或大于 1），加快攻击节奏，方便短步数 smoke-test 观察扣分。
        修改写入临时文件，供本次 wrapper 使用，不影响原 YAML。
        """

        flag = os.environ.get("PRIMAITE_FORCE_FAST_RED_START")
        if flag is None or _to_bool(flag) is False:
            return

        if not isinstance(self._config_cache, dict):
            return

        cfg = copy.deepcopy(self._config_cache)
        changed = False
        for agent in cfg.get("agents", []):
            try:
                team = str(agent.get("team", "")).lower()
            except Exception:
                continue
            if team != "red":
                continue

            if agent.get("start_step", None) != 1:
                agent["start_step"] = 1
                changed = True
            if agent.get("start_variance", None) != 0:
                agent["start_variance"] = 0
                changed = True
            if agent.get("random_offset_max", None) != 0:
                agent["random_offset_max"] = 0
                changed = True
            if int(agent.get("frequency", 0) or 0) != 1:
                agent["frequency"] = 1
                changed = True

        if not changed:
            return

        try:
            tmp = tempfile.NamedTemporaryFile(
                mode="w", suffix=f"_{self.scenario}_fastred.yaml", delete=False
            )
            yaml.safe_dump(cfg, tmp)
            tmp.flush()
            tmp.close()
            self._temp_config_path = Path(tmp.name)
            self.config_path = self._temp_config_path
            self._config_cache = cfg
        except Exception:
            # 如果写入失败，保持原配置即可
            self._temp_config_path = None
            self.config_path = self.config_path

    def _compute_max_reward_from_config(self) -> float:
        """
        读取配置中所有 reward_components 的正向权重之和，作为理论最大奖励。

        说明：PrimAITE 的 reward 组件通常是「正向奖励 + 负向惩罚」，
        我们只累加正权重部分，以便用于归一化（避免负权重抵消总分）。
        """

        cfg = self._config_cache
        if not isinstance(cfg, dict):
            try:
                cfg = yaml.safe_load(self.config_path.read_text(encoding="utf-8"))
            except Exception:
                return 1.0

        reward_components: List[Dict[str, Any]] = []

        def _walk(obj: Any) -> None:
            if isinstance(obj, dict):
                for k, v in obj.items():
                    if k == "reward_components" and isinstance(v, list):
                        reward_components.extend(v)
                    _walk(v)
            elif isinstance(obj, list):
                for item in obj:
                    _walk(item)

        _walk(cfg)

        total = 0.0
        for comp in reward_components:
            try:
                w = float(comp.get("weight", 0.0))
                if w > 0:
                    total += w
            except Exception:
                continue

        return total if total > 0 else 1.0

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
        轻量版 ICS facts（加上最近几步的“记忆”）：
        - recent_reward / positive_recent_reward / negative_recent_reward
        - nmne_* / traffic_*：用总量 + 更宽松阈值
        - nmne_recent_* / traffic_recent_spike：短期记忆 flag，让信号更“黏”
        """

        # ---- 短期记忆容器（挂在 wrapper 实例上）----
        if not hasattr(self, "_ics_state"):
            self._ics_state = {
                "nmne_recent": 0,
                "traffic_recent": 0,
            }

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
            "attack_detected": False,
            "integrity_lost": False,
            "nmne_detected": False,
            "nmne_high": False,
            "nmne_medium": False,
            "traffic_spike": False,
            "traffic_tcp_high": False,
            "traffic_udp_high": False,
            "traffic_icmp_high": False,
            "node_down": False,
            "critical_node_down": False,
            # 计数型：允许规则用 “>0” 或 “>=阈值”
            "failed_connections": 0,
            "failed_requests": 0,
            "dos_detected": False,
            "ransomware_detected": False,
            "manipulation_detected": False,
            # 新增的“近期异常” flag（下面会填值）
            "nmne_recent_mild": False,
            "nmne_recent_high": False,
            "traffic_recent_spike": False,
            "env_name": self.scenario,
        }

        # 尝试从 obs 里解析结构
        structured: Optional[Any] = None
        if isinstance(obs_raw, dict):
            structured = obs_raw
        elif isinstance(obs_raw, np.ndarray):
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

                    # nmne 结构：value 一般是 {'inbound': x, 'outbound': y}
                    if key == "nmne" and isinstance(v, dict):
                        for val in v.values():
                            nmne_levels.append(_to_level(val))

                    # traffic 结构：多层 dict，叶子是各协议/端口的计数
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
                        host_hit = next(
                            (
                                self.host_alias.get(seg.lower())
                                for seg in path + next_path
                                if self.host_alias.get(seg.lower())
                            ),
                            None,
                        )
                        if host_hit:
                            node_statuses[host_hit] = _to_level(v)
                        else:
                            node_statuses.setdefault("*", _to_level(v))

                    if key in {"connection_errors", "failed_connections"}:
                        facts["failed_connections"] += _to_level(v)
                    if key == "failed_requests":
                        facts["failed_requests"] += _to_level(v)

                    # 文件健康度
                    if key == "file_health":
                        lvl = _to_level(v)
                        if lvl != 1:
                            facts["integrity_lost"] = True

                    _walk(v, next_path)

            elif isinstance(obj, (list, tuple)):
                for idx, item in enumerate(obj):
                    _walk(item, path + [str(idx)])

        if structured is not None:
            _walk(structured, [])

        # ---- nmne / traffic 逻辑：用总量 + 更宽松的阈值 ----
        nmne_total = sum(max(0, lvl) for lvl in nmne_levels)
        traffic_total = sum(max(0, lvl) for lvl in traffic_levels)
        traffic_sum_by_proto = {
            proto: sum(max(0, lvl) for lvl in lvls)
            for proto, lvls in traffic_by_proto.items()
        }

        # 只要总量>0 就认为有检测到；total>=3 认为“偏高”
        facts["nmne_detected"] = nmne_total > 0
        facts["nmne_medium"] = nmne_total >= 1
        facts["nmne_high"] = nmne_total >= 3
        facts["attack_detected"] = nmne_total > 5

        # 总流量>=3 认为 spike，某协议>=2 认为 high
        facts["traffic_spike"] = traffic_total >= 3
        facts["traffic_tcp_high"] = traffic_sum_by_proto["tcp"] >= 2
        facts["traffic_udp_high"] = traffic_sum_by_proto["udp"] >= 2
        facts["traffic_icmp_high"] = traffic_sum_by_proto["icmp"] >= 2

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

        # ---- 文本信号：DoS / ransomware / 写入痕迹 ----
        raw_str = str(obs_raw).lower()
        if "dos" in raw_str or "flood" in raw_str:
            facts["dos_detected"] = True
        if "encrypt" in raw_str or "ransom" in raw_str:
            facts["ransomware_detected"] = True
        if "write" in raw_str and ("database" in raw_str or "modbus" in raw_str):
            facts["manipulation_detected"] = True

        # ---- 短期记忆：最近几步是否持续异常 ----
        # nmne：只要 detect 就累加，没 detect 就衰减
        if facts["nmne_detected"]:
            self._ics_state["nmne_recent"] = min(self._ics_state["nmne_recent"] + 1, 10)
        else:
            self._ics_state["nmne_recent"] = max(self._ics_state["nmne_recent"] - 1, 0)

        # traffic：只要有任意 traffic>0（不管是不是 spike）就累加
        traffic_event = traffic_total > 0
        if traffic_event:
            self._ics_state["traffic_recent"] = min(self._ics_state["traffic_recent"] + 1, 10)
        else:
            self._ics_state["traffic_recent"] = max(self._ics_state["traffic_recent"] - 1, 0)

        facts["nmne_recent_mild"] = self._ics_state["nmne_recent"] >= 1
        facts["nmne_recent_high"] = self._ics_state["nmne_recent"] >= 3
        facts["traffic_recent_spike"] = self._ics_state["traffic_recent"] >= 2

        # 如果出现 nmne / traffic spike / 关键节点 down / 负奖励 / 攻击证据，就视为 "suspicious_activity"
        facts["suspicious_activity"] = (
                facts["nmne_detected"]
                or facts["nmne_high"]
                or facts["traffic_spike"]
                or facts["traffic_recent_spike"]
                or critical_down
                or facts["negative_recent_reward"]
                or facts["failed_connections"] > 0
                or facts["failed_requests"] > 0
                or facts["dos_detected"]
                or facts["ransomware_detected"]
                or facts["manipulation_detected"]
                or facts["integrity_lost"]
                or facts["attack_detected"]
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
        try:
            res = self.env.step(action)
        except Exception as exc:  # noqa: BLE001
            # 兜底：一旦底层环境报错，也要返回一个合法的 done step，避免 smoke_test 崩溃
            res = None
            base_info: Dict[str, Any] = {
                "env_name": self.scenario,
                "mapped_action_name": self.get_action_name(action),
                "intent_name": self._infer_intent_name(action),
                "error": f"env.step raised: {exc!r}",
            }
        else:
            base_info = {
                "env_name": self.scenario,
                "mapped_action_name": self.get_action_name(action),
                "intent_name": self._infer_intent_name(action),
            }

        # 某些安装缺失/依赖异常时，上游可能返回 None 或 4-tuple，这里兜底为一次 done step
        if res is None:
            obs_raw = None
            reward = 0.0
            done = True
            info = dict(base_info)
        elif isinstance(res, tuple):
            if len(res) == 5:
                obs_raw, reward, terminated, truncated, info = res
                done = bool(terminated or truncated)
            elif len(res) == 4:
                obs_raw, reward, done, info = res
            else:
                obs_raw, reward, done, info = None, 0.0, True, {"error": f"unexpected step tuple len={len(res)}"}
        else:
            obs_raw, reward, done, info = None, 0.0, True, {"error": f"unexpected step type={type(res)}"}

        # 确保 info 至少是一个 dict
        if not isinstance(info, dict):
            info = {}
        info.update({k: v for k, v in base_info.items() if k not in info})

        # ★ 只增加 blue_action，完全不碰 red_actions / green_actions
        if "blue_action" not in info:
            try:
                info["blue_action"] = self.get_action_name(action)
            except Exception:
                info["blue_action"] = f"action-{int(action)}"

        # 便于 smoke test / 训练日志输出统一包含场景与动作名
        info.setdefault("env_name", self.scenario)
        info.setdefault("mapped_action_name", info.get("blue_action"))
        if "intent_name" not in info:
            info["intent_name"] = self._infer_intent_name(action)

        # 下面是你原来的部分
        if obs_raw is None:
            obs_vec = np.zeros((self.action_dim,), dtype=np.float32)
            facts: Dict[str, Any] = {}
        else:
            obs_vec = self._flatten_obs(obs_raw)
            facts = self._extract_facts(obs_raw, reward=float(reward))

        obs = {
            "obs_vec": obs_vec,
            "facts": facts,
            "raw": obs_raw,
            "env_name": self.scenario,
        }
        return obs, float(reward), done, info

    def _infer_intent_name(self, idx: int) -> Optional[str]:
        """基于动作名的简单关键字匹配，推测 4 个语义意图之一。"""

        if not self.action_names:
            return None

        try:
            name = str(self.action_names[int(idx)]).lower()
        except Exception:
            return None

        def has_any(substrs: Iterable[str]) -> bool:
            return any(s in name for s in substrs)

        if has_any(["block", "deny", "acl", "isolate", "quarantine", "drop"]):
            return self.intent_labels[1]  # Block

        if has_any(["restore", "startup", "start", "repair", "fix", "recover", "enable", "bringup"]):
            return self.intent_labels[2]

    def get_action_name(self, idx: int) -> str:
        """
        给上层训练脚本 / logger 用：
        - 如果 config 里解析到了 action_names，就用人类可读的名字；
        - 否则退回 "action-<idx>"，至少不会报错。
        """
        try:
            i = int(idx)
        except Exception:
            return str(idx)

        if 0 <= i < len(self.action_names):
            return self.action_names[i]
        return f"action-{i}"

    # def action_masks(self) -> np.ndarray:
    #     """
    #     暴露给 MultiEnvWrapper._get_local_mask 使用。
    #     """
    #     if hasattr(self.env, "action_masks"):
    #         try:
    #             m = self.env.action_masks()
    #             return np.asarray(m, dtype=np.float32).reshape(-1)
    #         except Exception:
    #             pass
    #
    #     # 没拿到的话，就全 1（全部合法）
    #     if self.action_dim <= 0:
    #         return np.array([], dtype=np.float32)
    #     return np.ones(self.action_dim, dtype=np.float32)

    def close(self) -> None:
        if hasattr(self.env, "close"):
            self.env.close()
        if self._temp_config_path and self._temp_config_path.exists():
            try:
                self._temp_config_path.unlink()
            except Exception:
                pass