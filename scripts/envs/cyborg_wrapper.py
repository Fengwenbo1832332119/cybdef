import os
import random
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Callable
import numpy as np
import yaml
import json
import pathlib


from CybORG import CybORG
from CybORG.Simulator.Scenarios import FileReaderScenarioGenerator
from CybORG.Agents import B_lineAgent, RedMeanderAgent, BlueReactRestoreAgent, BlueReactRemoveAgent, MonitorAgent
from CybORG.Agents.Wrappers import EnumActionWrapper, BlueTableWrapper

# ====== 代理注册表 ======
RED_AGENT_REGISTRY: Dict[str, Callable[[], Any]] = {
    "B_lineAgent": lambda: B_lineAgent(),
    "MeanderAgent": lambda: RedMeanderAgent(),
}

BLUE_AGENT_REGISTRY: Dict[str, Callable[[], Any]] = {
    "BlueReactRestoreAgent": lambda: BlueReactRestoreAgent(),
    "BlueReactRemoveAgent": lambda: BlueReactRemoveAgent(),
    "MonitorAgent": lambda: MonitorAgent()
}

# ====== 运行统计归一化（Welford） ======
class RunningMeanStd:
    def __init__(self, shape, clip: float = 10.0):
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = 1e-4
        self.clip = clip

    def update(self, x: np.ndarray):
        x = x.astype(np.float64)
        bmean = x.mean(axis=0)
        bvar  = x.var(axis=0)
        bcnt  = x.shape[0] if x.ndim > 1 else 1.0

        delta = bmean - self.mean
        tot   = self.count + bcnt

        new_mean = self.mean + delta * bcnt / tot
        m_a = self.var * self.count
        m_b = bvar * bcnt
        M2  = m_a + m_b + (delta**2) * self.count * bcnt / tot
        new_var = M2 / tot

        self.mean, self.var, self.count = new_mean, new_var, tot

    def normalize(self, x: np.ndarray) -> np.ndarray:
        std = np.sqrt(self.var) + 1e-8
        y = (x - self.mean) / std
        return np.clip(y, -self.clip, self.clip).astype(np.float32)

@dataclass
class ActionSpace:
    names: List[str]
    @property
    def n(self) -> int: return len(self.names)

class CybORGWrapper:
    """
    完整适配配置的 CybORG 包装器（Scenario2）：
    - 支持 red_pool（per_episode/round_robin/weighted）
    - BlueTableWrapper 输出向量 + EnumActionWrapper 离散动作
    - 合法动作掩码自动获取
    - 观察归一化（Welford）
    - 复现性（seed）
    """
    def __init__(self, config_path: str):
        self.config = self._load_config(config_path)

        env_cfg = self.config['environment']
        self.mode = self.config.get('mode', 'train')
        self.max_steps = int(env_cfg.get('max_steps', 100))
        self.reset_tries = int(env_cfg.get('reset_tries', 10))
        self.reward_mode = env_cfg.get('reward_mode', 'dense')
        self.deterministic = bool(env_cfg.get('deterministic', True))
        self._episode_steps = 0

        # 🔹 新增：从 seed_graph.json 读取 host 的角色 / 关键度
        self._init_seed_graph()

        # 红方代理池
        agents_cfg = env_cfg.get('agents', {})
        self.red_pool_cfg = agents_cfg.get('red_pool', [])
        self.red_sampling_mode = agents_cfg.get('red_sampling', 'per_episode')
        self._rr_idx = 0
        self._current_red_agent = None

        # 解析路径
        self._resolve_and_validate_paths()

        # 固定随机性
        self._set_seed(int(env_cfg.get('seed', 42)))

        # 创建环境
        self.env = self._create_environment()

        # 动作空间
        self._setup_action_space()

        # 观察空间
        obs_cfg = env_cfg['observation_space']
        self.obs_dim = int(obs_cfg.get('dimensions', 256))
        self.normalize_obs = bool(obs_cfg.get('normalize', True))
        self.running_stats = bool(obs_cfg.get('running_stats', True))
        self._rms = RunningMeanStd(shape=(self.obs_dim,), clip=10.0) if (self.normalize_obs and self.running_stats) else None

        # 缓存最近一次 mask
        self._last_mask_cache: Optional[np.ndarray] = None
        self._last_result: Optional[Any] = None

        print("✅ 环境包装器初始化完成")
        print(f"   模式={self.mode}  最大步数={self.max_steps}  奖励模式={self.reward_mode}")
        print(f"   红方模式={self.red_sampling_mode}  池={ [x.get('class') for x in self.red_pool_cfg] }")
        print(f"   动作空间={self.action_dim}  观察维度={self.obs_dim}  归一化={self.normalize_obs}/{self.running_stats}")

    # ---------- 配置 & 路径 ----------
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"配置文件不存在: {config_path}")
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)

    def _init_seed_graph(self):
        """
        从 scripts/configs/seed_graph.json 里读出 host 元数据：
        - role: user_host / enterprise_server / operational_server / operational_host / defender
        - criticality: 0~4（我们在 seed_graph.json 里已经配好了）
        """
        # 计算项目根目录：.../scripts/envs/cyborg_wrapper.py -> .../
        cur_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(os.path.dirname(cur_dir))
        sg_path = os.path.join(project_root, "scripts", "configs", "seed_graph.json")

        self._host_roles = {}
        self._host_groups = {
            "user": set(),
            "enterprise": set(),
            "op_hosts": set(),
            "op_server": set(),
            "defender": set(),
        }

        if not os.path.exists(sg_path):
            print(f"⚠ 未找到 seed_graph.json: {sg_path}，将使用默认 host 分组")
            # 默认兜底分组（如果 seed_graph 不存在，就用硬编码）
            for h in [
                "User0", "User1", "User2", "User3", "User4"
            ]:
                self._host_groups["user"].add(h)
            for h in [
                "Enterprise0", "Enterprise1", "Enterprise2"
            ]:
                self._host_groups["enterprise"].add(h)
            for h in [
                "Op_Host0", "Op_Host1", "Op_Host2"
            ]:
                self._host_groups["op_hosts"].add(h)
            self._host_groups["op_server"].add("Op_Server0")
            self._host_groups["defender"].add("Defender")
            return

        try:
            with open(sg_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            print(f"⚠ 读取 seed_graph.json 失败: {e}")
            return

        hosts = data.get("hosts", [])
        for h in hosts:
            hid = h.get("id")
            if not hid:
                continue
            self._host_roles[hid] = h
            role = h.get("role", "")
            if role == "user_host":
                self._host_groups["user"].add(hid)
            elif role == "enterprise_server":
                self._host_groups["enterprise"].add(hid)
            elif role == "operational_host":
                self._host_groups["op_hosts"].add(hid)
            elif role == "operational_server":
                self._host_groups["op_server"].add(hid)
            elif role == "defender":
                self._host_groups["defender"].add(hid)

        print("✅ seed_graph 主机分组加载完成:")
        for k, v in self._host_groups.items():
            print(f"   {k}: {sorted(v)}")


    def _resolve_and_validate_paths(self):
        env_cfg = self.config['environment']
        scenario_file = env_cfg['scenario_file']

        if not os.path.isabs(scenario_file):
            cur_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(os.path.dirname(cur_dir))
            if scenario_file.startswith("../"):
                scenario_file = os.path.join(project_root, scenario_file[3:])
            else:
                scenario_file = os.path.join(project_root, scenario_file)

        if not os.path.exists(scenario_file):
            # 兜底：尝试 1b 与安装路径
            candidates = [
                scenario_file,
                scenario_file.replace("Scenario2.yaml", "Scenario1b.yaml"),
                os.path.join(os.path.dirname(scenario_file), "Scenario1b.yaml"),
            ]
            try:
                import CybORG as _C
                cyb_path = os.path.dirname(os.path.abspath(_C.__file__))
                candidates.append(os.path.join(cyb_path, "Simulator", "Scenarios", "scenario_files", "Scenario1b.yaml"))
            except Exception:
                pass
            for p in candidates:
                if os.path.exists(p):
                    scenario_file = p
                    break
            else:
                raise FileNotFoundError(f"场景文件不存在，尝试过：{candidates}")

        env_cfg['scenario_file'] = scenario_file

    def _set_seed(self, seed: int):
        random.seed(seed); np.random.seed(seed)
        try:
            import torch
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
                if self.deterministic:
                    torch.backends.cudnn.deterministic = True
                    torch.backends.cudnn.benchmark = False
        except Exception:
            pass

    # ---------- 代理构造 ----------
    def _pick_red_agent(self):
        if not self.red_pool_cfg:
            # 单一 red
            single = self.config['environment']['agents'].get('red', 'B_lineAgent')
            return self._build_red_agent(single)
        if self.red_sampling_mode == 'round_robin':
            cls = self.red_pool_cfg[self._rr_idx]['class']
            self._rr_idx = (self._rr_idx + 1) % len(self.red_pool_cfg)
            return self._build_red_agent(cls)
        # per_episode 按权重
        weights = [float(x.get('weight', 1.0)) for x in self.red_pool_cfg]
        classes = [x['class'] for x in self.red_pool_cfg]
        chosen = random.choices(classes, weights=weights, k=1)[0]
        return self._build_red_agent(chosen)

    def _build_red_agent(self, class_name: str):
        if class_name not in RED_AGENT_REGISTRY:
            raise ValueError(f"未知的红方代理: {class_name}")
        agent = RED_AGENT_REGISTRY[class_name]()
        self._current_red_agent = class_name
        return agent

    def _build_blue_agent(self):
        blue_cls = self.config['environment']['agents']['blue']
        if blue_cls not in BLUE_AGENT_REGISTRY:
            raise ValueError(f"未知的蓝方代理: {blue_cls}")
        return BLUE_AGENT_REGISTRY[blue_cls]()

    # ---------- 环境构建 ----------
    def _create_environment(self):
        scen = self.config['environment']['scenario_file']
        seed = int(self.config['environment'].get('seed', 42))

        red = self._pick_red_agent()
        blue = self._build_blue_agent()

        scenario_gen = FileReaderScenarioGenerator(scen)
        base_env = CybORG(
            scenario_generator=scenario_gen,
            environment='sim',
            agents={'Red': red, 'Blue': blue},
            seed=seed
        )
        # BlueTableWrapper 先把观测表格化/向量化，再 EnumActionWrapper 离散化动作
        table_env = BlueTableWrapper(base_env, output_mode='vector')
        enum_env = EnumActionWrapper(table_env)
        return enum_env

    # ---------- 动作空间 / 合法掩码 ----------
    def _setup_action_space(self):
        """设置动作空间 - 从 EnumActionWrapper.possible_actions 中提取真实动作对象，
        并生成带 host 语义的动作名，例如：
            MonitorTraffic
            InvestigateHost_Enterprise1
            RemoveMalware_Op_Server0
            RestoreService_Enterprise2
            DecoyApache_Enterprise1
        """

        # 统一的 host 名称顺序（和 _extract_facts 里的保持一致）
        host_order = [
            "Defender",
            "Enterprise0",
            "Enterprise1",
            "Enterprise2",
            "Op_Host0",
            "Op_Host1",
            "Op_Host2",
            "Op_Server0",
            "User0",
            "User1",
            "User2",
            "User3",
            "User4",
        ]

        def infer_host_from_action(act, raw_name: str) -> str:
            """尽量从动作对象中推断目标 host 名称，失败则返回 'UnknownHost'。"""
            # 1) 先从常见字段里找
            for attr in ("hostname", "host", "target", "session"):
                try:
                    v = getattr(act, attr, None)
                except Exception:
                    v = None
                if isinstance(v, str) and v in host_order:
                    return v

            # 2) 从 __repr__ 或字符串里搜索 host 名
            text = raw_name
            try:
                text = repr(act)
            except Exception:
                pass
            for h in host_order:
                if h in text:
                    return h

            return "UnknownHost"

        try:
            # 1) 拿动作数
            space = self.env.get_action_space('Blue')
            if isinstance(space, int):
                action_count = int(space)
            elif hasattr(space, "n"):
                action_count = int(space.n)
            else:
                action_count = len(space)

            raw_names = []
            semantic_names = []

            # 2) 从 EnumActionWrapper 暴露的 possible_actions 里拿到底层动作对象
            action_objs = None
            if hasattr(self.env, "possible_actions"):
                pa = self.env.possible_actions
                if isinstance(pa, dict) and 'Blue' in pa:
                    action_objs = pa['Blue']
                elif isinstance(pa, (list, tuple)):
                    action_objs = pa

            if action_objs is None or len(action_objs) != action_count:
                # 兜底：拿不到动作对象，就退回到占位名
                self.action_space = ActionSpace(names=[f"action_{i}" for i in range(action_count)])
                print("⚠ 无法从 possible_actions 中获取动作对象，使用占位动作名。")
                return

            for act in action_objs:
                # ---- 原始名字 / 类型判断 ----
                try:
                    cls_name = act.__class__.__name__
                except Exception:
                    cls_name = type(act).__name__
                raw_name = cls_name
                raw_names.append(raw_name)

                low = raw_name.lower()

                # ---- 识别动作大类 ----
                base = None

                if "sleep" in low:
                    base = "Sleep"
                elif "monitor" in low:
                    base = "MonitorTraffic"
                elif "analyse" in low or "analyze" in low or "investigate" in low:
                    base = "InvestigateHost"
                elif "remove" in low:
                    base = "RemoveMalware"
                elif "restore" in low:
                    base = "RestoreService"
                elif "deco" in low:
                    # Decoy 再细分服务类型
                    if "apache" in low:
                        base = "DecoyApache"
                    elif "femitter" in low:
                        base = "DecoyFemitter"
                    elif "haraka" in low or "smtp" in low:
                        base = "DecoyHarakaSMPT"
                    elif "smss" in low:
                        base = "DecoySmss"
                    elif "sshd" in low:
                        base = "DecoySSHD"
                    elif "svchost" in low:
                        base = "DecoySvchost"
                    elif "tomcat" in low:
                        base = "DecoyTomcat"
                    elif "vsftpd" in low or "ftp" in low:
                        base = "DecoyVsftpd"
                    else:
                        base = "DecoyGeneric"
                else:
                    # 完全未知类型，保留原始类名，方便后面 debug
                    base = cls_name

                # ---- 是否需要 host 后缀 ----
                # MonitorTraffic / Sleep 通常不带 host，其他我们尽量 host-aware 化
                if base in ("MonitorTraffic", "Sleep"):
                    semantic_name = base
                else:
                    host = infer_host_from_action(act, raw_name)
                    # 对于 Recover / Investigate / Remove / Decoy，统一加 host 后缀
                    semantic_name = f"{base}_{host}"

                semantic_names.append(semantic_name)

            self.action_space = ActionSpace(names=semantic_names)

            print(f"🎯 真实动作空间大小: {action_count}")
            print("🎯 前 30 个动作名示例:")
            for i, n in enumerate(self.action_space.names[:30]):
                print(f"  [{i}] {n}  (raw={raw_names[i]})")

        except Exception as e:
            print(f"❌ 设置动作空间失败: {e}")
            self.action_space = ActionSpace(names=[f"action_{i}" for i in range(10)])

    def _extract_legal_mask_from_result(self, result) -> Optional[np.ndarray]:
        # 优先：result.action_space（某些版本返回 mask 列表）
        m = None
        try:
            if hasattr(result, 'action_space'):
                rs = result.action_space
                # 可能是 list/np.ndarray/自定义空间
                if isinstance(rs, (list, tuple, np.ndarray)) and len(rs) == self.action_space.n:
                    m = np.asarray(rs, dtype=np.float32).reshape(-1)
                elif hasattr(rs, 'mask'):
                    mask = getattr(rs, 'mask')
                    if mask is not None and len(mask) == self.action_space.n:
                        m = np.asarray(mask, dtype=np.float32).reshape(-1)
        except Exception:
            m = None
        if m is not None:
            return m

        # 其次：环境 action_space 提供 get_action_mask
        try:
            asp = self.env.get_action_space('Blue')
            if hasattr(asp, 'get_action_mask'):
                m2 = np.asarray(asp.get_action_mask(), dtype=np.float32).reshape(-1)
                if m2.size == self.action_space.n:
                    return m2
        except Exception:
            pass
        return None

    def _current_legal_mask(self) -> np.ndarray:
        if self._last_mask_cache is not None:
            return self._last_mask_cache
        # 尝试拉一次
        try:
            asp = self.env.get_action_space('Blue')
            if hasattr(asp, 'get_action_mask'):
                m = np.asarray(asp.get_action_mask(), dtype=np.float32)
                if m.size == self.action_space.n:
                    return m
        except Exception:
            pass
        return np.ones(self.action_space.n, dtype=np.float32)

    # ---------- 观测 / 事实 / 奖励 ----------
    def _encode_observation(self, raw_obs) -> np.ndarray:
        # BlueTableWrapper(output_mode='vector') 通常给向量
        if isinstance(raw_obs, (list, tuple, np.ndarray)):
            vec = np.asarray(raw_obs, dtype=np.float32).flatten()
        else:
            vec = np.zeros(self.obs_dim, dtype=np.float32)
        # 对齐到 obs_dim
        if vec.size < self.obs_dim:
            out = np.zeros(self.obs_dim, dtype=np.float32)
            out[:vec.size] = vec
            vec = out
        elif vec.size > self.obs_dim:
            vec = vec[:self.obs_dim]
        # 在线归一化
        if self._rms is not None:
            self._rms.update(vec[None, :])
            vec = self._rms.normalize(vec)
        return vec

    # ===== 从观测中提取“语义化事实” =====
    # ===== 从观测中提取“语义化事实” =====
    # ===== 从观测中提取“语义化事实”（v0.4, host-aware） =====
    def _extract_facts(self, raw_obs, reward: float = 0.0) -> Dict[str, Any]:
        """
        将 BlueTableWrapper 的向量观测转成 CSKG 用的高层事实（host-aware 版）：

        - suspicious_activity        : 是否存在扫描/利用等可疑行为
        - host_compromised           : 是否有任何主机被攻陷

        - enterprise_compromised     : 是否有 EnterpriseX 被攻陷
        - opserver_compromised       : Op_Server0 是否被攻陷
        - ophost_compromised         : 是否有 Op_HostX 被攻陷
        - user_compromised           : 是否有 UserX 被攻陷
        - only_user_compromised      : 只有 UserX 被攻陷（Enterprise / Op_Server0 / Op_Host 都没事）

        - critical_host_breached     : EnterpriseX / Op_Server0 是否被攻陷
        - critical_host              : 同上，兼容旧字段
        - host_discovered            : 向量里是否出现过任何非 0 状态
        - high_risk_state            : 是否处于高风险（关键主机被攻陷）

        - recent_reward              : 最近一步 env reward
        - bad_recent_reward          : recent_reward < -0.1
        - very_bad_recent_reward     : recent_reward < -1.0
        """

        facts = {
            "suspicious_activity": False,
            "host_compromised": False,

            "enterprise_compromised": False,
            "opserver_compromised": False,
            "ophost_compromised": False,
            "user_compromised": False,
            "only_user_compromised": False,  # ✅ 新增：先默认 False

            "critical_host_breached": False,
            "critical_host": False,
            "host_discovered": False,
            "high_risk_state": False,

            "recent_reward": float(reward),
            "bad_recent_reward": float(reward) < -0.1,
            "very_bad_recent_reward": float(reward) < -1.0,
        }

        try:
            import numpy as np
        except ImportError:
            np = None

        # ---------- 主流程：向量观测（训练 / eval 正常走这里） ----------
        if np is not None and isinstance(raw_obs, np.ndarray) and raw_obs.ndim == 1:
            vec = raw_obs.astype(int)

            if vec.shape[0] >= 52:
                # 13 hosts * 4 bits
                v = vec[:52].reshape(13, 4)

                host_order = [
                    "Defender",
                    "Enterprise0",
                    "Enterprise1",
                    "Enterprise2",
                    "Op_Host0",
                    "Op_Host1",
                    "Op_Host2",
                    "Op_Server0",
                    "User0",
                    "User1",
                    "User2",
                    "User3",
                    "User4",
                ]

                any_activity = False
                any_comp = False
                enterprise_comp = False
                opserver_comp = False
                ophost_comp = False
                user_comp = False
                critical_breached = False

                for idx, bits in enumerate(v):
                    b0, b1, b2, b3 = bits.tolist()
                    host_name = host_order[idx]

                    # Activity 语义
                    is_scan_or_exploit = (b0 == 1)
                    is_exploit = (b0 == 1 and b1 == 1)

                    # Compromised 语义
                    is_compromised = (b2 == 1) or (b3 == 1)

                    if is_scan_or_exploit or is_compromised:
                        any_activity = True

                    if is_compromised:
                        any_comp = True

                        if host_name.startswith("Enterprise"):
                            enterprise_comp = True
                            critical_breached = True

                        elif host_name == "Op_Server0":
                            opserver_comp = True
                            critical_breached = True

                        elif host_name.startswith("Op_Host"):
                            ophost_comp = True

                        elif host_name.startswith("User"):
                            user_comp = True

                # ✅ 把这些局部变量写回 facts
                facts["suspicious_activity"] = any_activity
                facts["host_compromised"] = any_comp

                facts["enterprise_compromised"] = enterprise_comp
                facts["opserver_compromised"] = opserver_comp
                facts["ophost_compromised"] = ophost_comp
                facts["user_compromised"] = user_comp

                # ✅ 关键：只有 User 被攻陷（且 Enterprise / Op_Server0 / Op_Host 都没事）
                only_user = user_comp and not (enterprise_comp or opserver_comp or ophost_comp)
                facts["only_user_compromised"] = only_user

                facts["critical_host_breached"] = critical_breached
                facts["critical_host"] = critical_breached

                facts["host_discovered"] = bool(v.any())
                # 高风险：只要核心主机被攻陷就认为 high_risk
                facts["high_risk_state"] = critical_breached or opserver_comp or enterprise_comp

                return facts

        # ---------- PrettyTable 兜底（主要用于你单独 debug BlueTableWrapper） ----------
        try:
            from prettytable import PrettyTable
        except ImportError:
            PrettyTable = None

        if PrettyTable is not None and isinstance(raw_obs, PrettyTable):
            any_activity = False
            any_comp = False
            enterprise_comp = False
            opserver_comp = False
            ophost_comp = False
            user_comp = False
            critical_breached = False

            for row in raw_obs.rows:
                row_dict = dict(zip(raw_obs.field_names, row))
                host = str(row_dict.get("Hostname", ""))
                activity = str(row_dict.get("Activity", "None"))
                compromised = str(row_dict.get("Compromised", "No"))

                if activity not in ("None", "", "Unknown"):
                    any_activity = True

                if compromised in ("User", "Privileged", "Yes"):
                    any_comp = True
                    if host.startswith("Enterprise"):
                        enterprise_comp = True
                        critical_breached = True
                    elif host == "Op_Server0":
                        opserver_comp = True
                        critical_breached = True
                    elif host.startswith("Op_Host"):
                        ophost_comp = True
                    elif host.startswith("User"):
                        user_comp = True

            facts["suspicious_activity"] = any_activity
            facts["host_compromised"] = any_comp
            facts["enterprise_compromised"] = enterprise_comp
            facts["opserver_compromised"] = opserver_comp
            facts["ophost_compromised"] = ophost_comp
            facts["user_compromised"] = user_comp

            # ✅ 兜底分支同样设置 only_user_compromised
            only_user = user_comp and not (enterprise_comp or opserver_comp or ophost_comp)
            facts["only_user_compromised"] = only_user

            facts["critical_host_breached"] = critical_breached
            facts["critical_host"] = critical_breached
            facts["host_discovered"] = any_activity or any_comp
            facts["high_risk_state"] = critical_breached or opserver_comp or enterprise_comp
            return facts

        # ---------- 兜底：看不到结构就只用 reward 做一点点弱信号 ----------
        if reward < -1.0:
            facts["suspicious_activity"] = True

        return facts

    def _extract_reward(self, result) -> float:
        base = float(getattr(result, 'reward', 0.0))
        if self.reward_mode == 'dense':
            # 简单时间惩罚，避免磨蹭
            base -= 0.01 * (self._episode_steps / max(1, self.max_steps))
        return base

    # ---------- 公共接口 ----------
    @property
    def observation_space(self):
        return {"shape": (self.obs_dim,), "dtype": "float32"}

    @property
    def action_dim(self):
        return self.action_space.n

    def reset(self):
        """重置环境（按 red_pool 策略重建 env），带重试"""
        self._episode_steps = 0

        if self.red_pool_cfg and self.red_sampling_mode in ('per_episode', 'round_robin'):
            self.close()
            self.env = self._create_environment()
            self._setup_action_space()

        last_err = None
        for attempt in range(self.reset_tries):
            try:
                result = self.env.reset(agent='Blue')
                self._last_result = result
                mask = self._extract_legal_mask_from_result(result)
                self._last_mask_cache = mask if mask is not None else None

                obs_vec = self._encode_observation(getattr(result, 'observation', None))
                obs = {
                    "obs_vec": obs_vec,
                    "facts": self._extract_facts(getattr(result, 'observation', None)),
                    "raw": getattr(result, 'observation', None)
                }
                return obs
            except Exception as e:
                last_err = e
                time.sleep(0.1)
        raise RuntimeError(f"环境重置失败（已重试 {self.reset_tries} 次）: {last_err}")

    def step(self, action_idx: Optional[int] = None):
        """训练：传入 int；baseline/eval：不传 action_idx"""
        self._episode_steps += 1
        try:
            if self.mode == "train" and action_idx is not None:
                ai = int(action_idx)
                # 越界兜底
                if ai < 0 or ai >= self.action_dim:
                    ai = 0
                result = self.env.step(agent='Blue', action=ai)
            else:
                result = self.env.step(agent='Blue')

            self._last_result = result
            mask = self._extract_legal_mask_from_result(result)
            self._last_mask_cache = mask if mask is not None else None

            raw_obs = getattr(result, 'observation', None)
            obs_vec = self._encode_observation(raw_obs)
            reward = self._extract_reward(result)
            done = bool(getattr(result, 'done', False) or self._episode_steps >= self.max_steps)

            info = {
                "legal_mask": self._current_legal_mask(),
                "steps": self._episode_steps,
                "success": bool(getattr(result, 'success', False)),
                "red_agent": self._current_red_agent,
                "reward_mode": self.reward_mode
            }

            obs = {
                "obs_vec": obs_vec,
                "facts": self._extract_facts(raw_obs),
                "raw": raw_obs
            }
            return obs, reward, done, info

        except Exception as e:
            # 兜底，不崩训练
            obs = {
                "obs_vec": np.zeros(self.obs_dim, dtype=np.float32),
                "facts": self._extract_facts(None),
                "raw": None
            }
            return obs, 0.0, True, {"error": str(e)}

    def close(self):
        if hasattr(self, 'env') and self.env is not None:
            try:
                # EnumActionWrapper -> BlueTableWrapper -> Base
                base = getattr(self.env, 'env', None)
                if base is not None and hasattr(base, 'env'):
                    base = base.env
                if base is not None and hasattr(base, 'close'):
                    base.close()
            except Exception:
                pass
