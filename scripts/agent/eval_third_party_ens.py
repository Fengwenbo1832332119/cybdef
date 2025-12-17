# scripts/agent/third_party_eval.py
# -*- coding: utf-8 -*-
"""
第三方环境（CybORG++ / PrimAITE）统一评估脚本。

当前版本基于你的多环境训练框架：

- 评估用的环境：通过 MultiEnvWrapper 包一层
    * --env cyborg     -> MultiEnvWrapper(env_names=["cyborg"])
    * --env ics/lot/... -> MultiEnvWrapper(env_names=["ics"] / ["lot"] / ["robotics"])
  这样就和 train_blue_multi_env.py 的结构完全一致：4 维语义动作头。

- 网络结构复用 scripts.agent.train_blue_multi_env.ActorCritic
- obs 向量化复用 train_blue_multi_env.to_obs_vector（通过 _to_obs_vec 封装）
- 支持：
    * 多个 checkpoint（目录 + --ckpt-range start:stop:step）
    * greedy 策略（argmax）评估
    * 动作合法性掩码（MultiEnvWrapper.current_action_mask / legal_mask 等兜底）
    * intercept_rate：关键资产是否被成功保护
    * false_positive_rate：无告警时误阻断比例

用法示例（PowerShell）：

    conda activate primaite311
    cd C:\\cybdef

    # 单个 ckpt，评估 CybORG（但底层是 MultiEnvWrapper 包着 cyborg）
    python scripts/agent/third_party_eval.py `
        --env cyborg `
        --model scripts/runs/quick_smoke_train/quick_smoke_train_1765246013/ac_multi_upd200.pt

    # 对同一个 run 目录下多个 ckpt 做曲线（cyborg）
    python scripts/agent/third_party_eval.py `
        --env cyborg `
        --model scripts/runs/quick_smoke_train/quick_smoke_train_1765246013 `
        --ckpt-range 0:200:25
"""

from __future__ import annotations

import argparse
import json
import pathlib
import random
import re
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import torch
from torch.distributions import Categorical  # 虽然用不到采样，保留也无所谓

# === 路径注入：和训练脚本保持一致 ===
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]  # C:\cybdef
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# === 工程内组件 ===
from scripts.envs.registry import ENV_REGISTRY  # 只用于 CLI choices
from scripts.agent.train_blue_multi_env import ActorCritic, to_obs_vector
from scripts.envs.multi_env_wrapper import MultiEnvWrapper


# ---------------------------------------------------------------------------
# 工具函数：obs 向量化 & 动作 mask
# ---------------------------------------------------------------------------

def _to_obs_vec(obs_raw: Any) -> np.ndarray:
    """
    统一 obs -> 1D float32，直接复用训练脚本里的 to_obs_vector。
    """
    return to_obs_vector(obs_raw)


def _get_action_mask(env: Any, obs_raw: Any, act_dim: int) -> Optional[np.ndarray]:
    """
    尽量不依赖具体 wrapper 名字地拿合法动作 mask。

    对 MultiEnvWrapper 来说：
    - 有 current_action_mask()，长度 = 全局 action_dim（这里是 4）
    """
    # 1) 优先用 env 对象上的方法
    for name in ("current_action_mask", "action_masks", "action_mask"):
        if hasattr(env, name):
            try:
                m = getattr(env, name)()
                m = np.asarray(m, dtype=np.float32).reshape(-1)
                if m.size == act_dim and m.sum() > 0:
                    return m > 0.5
            except Exception:
                pass

    # 2) 再尝试 obs 里的常见 key（一般 MultiEnvWrapper 不会放在 obs 里）
    if isinstance(obs_raw, dict):
        for k in ("legal_mask", "action_mask", "mask", "legal_actions"):
            if k in obs_raw:
                try:
                    m = np.asarray(obs_raw[k], dtype=np.float32).reshape(-1)
                    if m.size == act_dim and m.sum() > 0:
                        return m > 0.5
                except Exception:
                    pass

    # 3) 实在没找到就不用 mask（全 1）
    return np.ones(act_dim, dtype=bool)


def _apply_mask_to_logits(
    logits: torch.Tensor, mask: Optional[np.ndarray], device: torch.device
) -> torch.Tensor:
    if mask is None:
        return logits
    mask_t = torch.from_numpy(mask.astype(bool)).to(device)
    if mask_t.numel() != logits.numel():
        # 长度不匹配，宁可不用 mask 也别搞崩
        return logits
    # 非法动作设为 -1e9，避免 argmax 选到它
    return logits.masked_fill(~mask_t, -1e9)


# ---------------------------------------------------------------------------
# 工具函数：模型加载 & ckpt 解析
# ---------------------------------------------------------------------------

def _load_checkpoint(
    model_path: pathlib.Path, device: torch.device, obs_dim: int, act_dim: int
) -> ActorCritic:
    """
    关键点：
    - obs_dim / act_dim 必须和训练时一致
      * 这里 act_dim 直接用 MultiEnvWrapper.action_dim（语义动作头=4）
      * obs_dim 用 MultiEnvWrapper.obs_dim（只用 env=[env_name]，与你这次 quick_smoke 相同）
    """
    model = ActorCritic(obs_dim=obs_dim, act_dim=act_dim, hidden=256).to(device)
    ckpt = torch.load(model_path, map_location=device)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.load_state_dict(state)
    model.eval()
    return model


def _parse_range(expr: str) -> List[int]:
    """解析 start:stop:step 表达式，包含 stop。"""
    m = re.match(r"^(\d+):(\d+):(\d+)$", expr)
    if not m:
        raise ValueError("--ckpt-range 需要 start:stop:step 形式，例如 0:200:25")
    start, stop, step = map(int, m.groups())
    if step <= 0:
        raise ValueError("step 必须大于 0")
    if start > stop:
        raise ValueError("start 不能大于 stop")
    return list(range(start, stop + 1, step))


def _resolve_model_path(model_arg: str) -> pathlib.Path:
    """
    解析单个 checkpoint 路径：

    - 直接传入 ac_multi_upd200.pt / ac_upd200.pt：原样使用
    - 传入 run 目录：自动选择 update 最大的那个 checkpoint
    """
    path = pathlib.Path(model_arg)
    if path.is_file():
        return path
    if not path.exists() or not path.is_dir():
        raise FileNotFoundError(f"找不到模型文件或目录：{model_arg}")

    candidates = list(path.glob("ac_multi_upd*.pt")) + list(path.glob("ac_upd*.pt"))
    if not candidates:
        raise FileNotFoundError(
            f"目录 {model_arg} 中未找到 ac_multi_updXXX.pt / ac_updXXX.pt checkpoint"
        )

    def _upd_id(p: pathlib.Path) -> int:
        m = re.search(r"ac(?:_multi)?_upd(\d+)", p.name)
        return int(m.group(1)) if m else -1

    best = max(candidates, key=_upd_id)
    print(f"自动选用最新 checkpoint: {best}")
    return best


def _resolve_model_paths(model_arg: str, ckpt_range: Optional[str]) -> List[pathlib.Path]:
    """
    解析 CLI 输入为 checkpoint 列表。

    - 未指定 --ckpt-range：退化为单个 _resolve_model_path
    - 指定 --ckpt-range：要求 model_arg 是目录，
      按 start:stop:step 找 ac_multi_updXXX.pt / ac_updXXX.pt
    """
    if not ckpt_range:
        return [_resolve_model_path(model_arg)]

    dir_path = pathlib.Path(model_arg)
    if not dir_path.is_dir():
        raise FileNotFoundError(
            "--ckpt-range 只支持目录，--model 应该指向包含 ckpt 的 run 目录"
        )

    numbers = _parse_range(ckpt_range)

    def _find_checkpoint(update_id: int) -> Optional[pathlib.Path]:
        candidates = [
            dir_path / f"ac_multi_upd{update_id}.pt",
            dir_path / f"ac_multi_upd{update_id:03d}.pt",
            dir_path / f"ac_upd{update_id}.pt",
            dir_path / f"ac_upd{update_id:03d}.pt",
        ]
        for c in candidates:
            if c.exists():
                return c
        return None

    paths: List[pathlib.Path] = []
    missing: List[int] = []
    for n in numbers:
        p = _find_checkpoint(n)
        if p:
            paths.append(p)
        else:
            missing.append(n)

    if not paths:
        raise FileNotFoundError(
            f"目录 {model_arg} 中未找到范围 {ckpt_range} 内的 checkpoint"
        )
    if missing:
        print(f"⚠ 警告：以下 update 号未找到 checkpoint，将被跳过：{missing}")

    return paths


# ---------------------------------------------------------------------------
# 评估指标：intercept / false positive
# ---------------------------------------------------------------------------

def _is_true_alert(facts: Dict[str, Any], env_name: str) -> bool:
    """判断当前时刻是否存在“真实告警”（用于 false_positive 判断）。"""
    if not facts:
        return False

    if facts.get("suspicious_activity", False):
        return True

    if env_name == "cyborg":
        for k in (
            "host_compromised",
            "enterprise_compromised",
            "opserver_compromised",
            "critical_host_breached",
            "high_risk_state",
        ):
            if facts.get(k, False):
                return True
        return False

    # ICS / LOT / Robotics：节点 / 完整性 / 流量相关
    for k in (
        "critical_node_down",
        "node_down",
        "integrity_lost",
        "attack_detected",
        "dos_detected",
        "ransomware_detected",
        "manipulation_detected",
        "nmne_detected",
        "nmne_high",
        "traffic_spike",
        "traffic_recent_spike",
    ):
        if facts.get(k, False):
            return True

    return False


def _is_intercept_ok(facts: Dict[str, Any], env_name: str) -> bool:
    """
    判断一局结束时是否“成功拦截关键风险”：
    - cyborg：企业级 / 关键主机不能被成功入侵
    - ics/lot/robotics：关键节点不能 down，也不能出现严重完整性 / 攻击信号
    """
    if not facts:
        return True  # 没信息就当作没出事

    if env_name == "cyborg":
        bad = (
            facts.get("critical_host_breached", False)
            or facts.get("high_risk_state", False)
            or facts.get("enterprise_compromised", False)
        )
        return not bad

    # ICS / LOT / Robotics
    bad = (
        facts.get("critical_node_down", False)
        or facts.get("integrity_lost", False)
        or facts.get("attack_detected", False)
        or facts.get("dos_detected", False)
        or facts.get("ransomware_detected", False)
        or facts.get("manipulation_detected", False)
    )
    return not bad


# ---------------------------------------------------------------------------
# 单次评估（给指定 env + ckpt）
# ---------------------------------------------------------------------------

def evaluate_once(
    env_name: str,
    model_path: pathlib.Path,
    episodes: int,
    num_steps: int,
    seed: int = 42,
) -> Dict[str, float]:
    if env_name not in ENV_REGISTRY:
        raise KeyError(f"未知环境：{env_name}，可选：{list(ENV_REGISTRY.keys())}")

    # 固定随机种子（便于复现）
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # === 关键区别：这里用 MultiEnvWrapper 包一层 ===
    # 只启用一个场景（["cyborg"] / ["ics"] / ...），
    # 但内部仍然是 4 维语义动作头。
    env = MultiEnvWrapper(
        env_names=[env_name],
        weights=[1.0],
        mode="eval",
        use_semantic_intents=True,
    )

    obs_raw = env.reset()
    obs = _to_obs_vec(obs_raw)
    facts = obs_raw.get("facts", {}) if isinstance(obs_raw, dict) else {}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # MultiEnvWrapper 已经帮你统一了 obs_dim / action_dim
    obs_dim = int(obs.shape[0])
    act_dim = int(env.action_dim)  # 对当前结构来说 = 4

    model = _load_checkpoint(model_path, device, obs_dim=obs_dim, act_dim=act_dim)

    ep_rewards: List[float] = []
    ep_lengths: List[int] = []
    intercept_flags: List[bool] = []

    false_positive_actions = 0
    checked_actions = 0

    for ep in range(episodes):
        obs_raw = env.reset()
        obs = _to_obs_vec(obs_raw)
        facts = obs_raw.get("facts", {}) if isinstance(obs_raw, dict) else {}

        done = False
        step = 0
        ep_reward = 0.0

        while not done and step < num_steps:
            step += 1
            obs_tensor = torch.from_numpy(obs).to(device).unsqueeze(0)

            with torch.no_grad():
                logits, _ = model(obs_tensor)
                logits = logits.squeeze(0)

                # 合法动作 mask（4 维语义动作）
                mask = _get_action_mask(env, obs_raw, act_dim)
                logits = _apply_mask_to_logits(logits, mask, device)

                # 评估阶段：greedy（argmax），不要采样
                action_idx = int(torch.argmax(logits).item())

            # 在执行前，用“旧 facts”判断当前是否有真实告警
            has_alert = _is_true_alert(facts, env_name)

            # env.step：MultiEnvWrapper 会把 0~3 映射到具体动作，
            # info 里有 intent_name / mapped_action_name
            next_obs_raw, reward_env, done, info = env.step(action_idx)
            ep_reward += float(reward_env)

            # 统计误杀：在无真实告警时做 Block/Restore intent
            intent_name = None
            if isinstance(info, dict):
                intent_name = info.get("intent_name")  # "Monitor" / "Block" / "Restore" / "Sleep"
            if intent_name in ("Block", "Restore"):
                checked_actions += 1
                if not has_alert:
                    false_positive_actions += 1

            obs_raw = next_obs_raw
            obs = _to_obs_vec(next_obs_raw)
            facts = next_obs_raw.get("facts", {}) if isinstance(next_obs_raw, dict) else {}

        ep_rewards.append(ep_reward)
        ep_lengths.append(step)
        intercept_flags.append(_is_intercept_ok(facts, env_name))

    # 环境收尾
    if hasattr(env, "close"):
        try:
            env.close()
        except Exception:
            pass

    return {
        "reward_mean": float(np.mean(ep_rewards)),
        "reward_var": float(np.var(ep_rewards)),
        "length_mean": float(np.mean(ep_lengths)),
        "length_var": float(np.var(ep_lengths)),
        "intercept_rate": float(np.mean(intercept_flags)),
        "false_positive_rate": float(
            false_positive_actions / max(checked_actions, 1)
        ),
    }


def aggregate_runs(run_metrics: List[Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    summary: Dict[str, Dict[str, float]] = {}
    for key in run_metrics[0].keys():
        arr = np.array([m[key] for m in run_metrics], dtype=np.float32)
        summary[key] = {"mean": float(np.mean(arr)), "var": float(np.var(arr))}
    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="第三方环境统一评估（CybORG++ / PrimAITE，经 MultiEnvWrapper）")
    parser.add_argument(
        "--env",
        choices=list(ENV_REGISTRY.keys()),
        default="cyborg",
        help="要评估的环境：cyborg / ics / lot / robotics",
    )
    parser.add_argument(
        "--model",
        required=True,
        help="待评估的策略权重（单个 ckpt，或包含 ckpt 的 run 目录）",
    )
    parser.add_argument(
        "--ckpt-range",
        help="当 --model 指向目录时，使用 start:stop:step（含 stop）批量评估多份 checkpoint，例如 0:200:25",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=5,
        help="每次评估的回合数（对同一个 ckpt）",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=3,
        help="重复次数（用于统计 mean/var）",
    )
    parser.add_argument(
        "--num-steps",
        type=int,
        nargs="+",
        default=[30, 50, 100],
        help="每回合的最大步数列表（会分别跑一次：30 / 50 / 100）",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="随机种子（不同 repeat 会加偏移）",
    )

    args = parser.parse_args()
    model_paths = _resolve_model_paths(args.model, args.ckpt_range)

    all_results: Dict[str, Dict[str, Dict[str, Any]]] = {}

    for model_path in model_paths:
        model_key = model_path.name
        print(f"\n===== 评估 checkpoint: {model_key} =====")
        all_results[model_key] = {}

        for num_steps in args.num_steps:
            print(f"\n>>> env={args.env} | num_steps={num_steps}")
            run_metrics: List[Dict[str, float]] = []
            for r in range(args.repeats):
                metrics = evaluate_once(
                    env_name=args.env,
                    model_path=model_path,
                    episodes=args.episodes,
                    num_steps=num_steps,
                    seed=args.seed + r * 7,
                )
                print(f"[Run {r+1}/{args.repeats}] {metrics}")
                run_metrics.append(metrics)

            summary = aggregate_runs(run_metrics)
            print("-- 汇总（mean ± var） --")
            for k, v in summary.items():
                print(f"{k:20s}: {v['mean']:.4f} ± {v['var']:.4f}")
            all_results[model_key][str(num_steps)] = {
                "runs": run_metrics,
                "summary": summary,
            }

    logs_dir = pathlib.Path(__file__).resolve().parent / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_path = logs_dir / f"third_party_eval_{args.env}_{int(torch.randint(0, 1_000_000, (1,)).item())}.json"
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "env": args.env,
                "models": [str(p) for p in model_paths],
                "episodes": args.episodes,
                "repeats": args.repeats,
                "num_steps": args.num_steps,
                "results": all_results,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    print(f"\n📝 日志写入: {log_path}")


if __name__ == "__main__":
    main()
