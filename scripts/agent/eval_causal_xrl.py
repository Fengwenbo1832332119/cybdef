"""Causal XRL 行为评估与对比脚本

结合 ``eval_ppo_cskg_behavior.py`` 与 ``eval_ppo_cskg_compare.py`` 的功能：
- 支持一次评估多个 checkpoint（--ckpt a.pt b.pt ...），并汇总平均 EnvReward / TotalReward / 步长
- 同时输出行为级 JSONL（包含 facts / explain / prior / 掩码 / MSE），便于复盘
- 可选关闭知识桥（--with-plain），对同一 ckpt 做「带 KB」与「plain_no_kb」对比
- 最终在终端打印对比矩阵，并把汇总写入 CSV 方便画图
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import pathlib
import sys
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.distributions import Categorical

ROOT = pathlib.Path(__file__).resolve().parents[2]
THIRD = ROOT / "third_party" / "CybORG"

for p in (ROOT, THIRD):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import yaml  # noqa: E402
from scripts.agent.train_causal_xrl import (  # noqa: E402
    MultiHeadActorCritic,
    extract_graph,
    to_obs_vector,
)
from scripts.common.validation import MSEApproximation  # noqa: E402
from scripts.cskg.reasoner import KnowledgeBridge  # noqa: E402
from scripts.envs.cyborg_wrapper import CybORGWrapper  # noqa: E402

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass
class StepRecord:
    mode: str
    step: int
    head_idx: int
    action_idx: int
    action_name: str
    env_reward: float
    total_reward: float
    done: bool
    legal_mask_sum: float
    kb_mask_sum: float
    combined_mask_sum: float
    prior_chosen: float
    top_prior: List[Tuple[str, float]]
    mse_mode: str
    mse: float
    evidence_ids: List[str]
    facts: Dict[str, Any]
    explain: Dict[str, Any]


def _infer_hparams(state_dict: Dict[str, torch.Tensor]) -> Tuple[int, int, int]:
    """从 state_dict 推断 hidden、node_dim、num_heads，用于与训练一致地重建模型。"""
    hidden = 128
    node_dim = 32
    combine_w = state_dict.get("encoder.combine.weight")
    node_encoder_w = state_dict.get("encoder.node_encoder.weight")
    if combine_w is not None:
        hidden = combine_w.shape[0]
    if node_encoder_w is not None:
        node_dim = node_encoder_w.shape[1]

    head_ids = [
        int(k.split(".")[1])
        for k in state_dict.keys()
        if k.startswith("actor_heads.") and k.split(".")[1].isdigit()
    ]
    num_heads = max(head_ids) + 1 if head_ids else 1
    return hidden, node_dim, num_heads


def _load_rule_coef() -> float:
    cfg_path = ROOT / "scripts" / "configs" / "ppo.yaml"
    if not cfg_path.exists():
        return 0.0
    with cfg_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    return float(cfg.get("rule_coef", 0.0))


def _build_env() -> CybORGWrapper:
    env_yaml = ROOT / "scripts" / "configs" / "env.yaml"
    return CybORGWrapper(str(env_yaml))


def _build_kb() -> KnowledgeBridge:
    cskg_yaml = ROOT / "scripts" / "configs" / "cskg.yaml"
    seed_graph = ROOT / "scripts" / "configs" / "seed_graph.json"
    return KnowledgeBridge(
        seed_graph_path=str(seed_graph),
        cskg_rules_path=str(cskg_yaml),
        recent_steps=10,
    )


def _prepare_logger(ckpt_path: str, mode: str) -> pathlib.Path:
    run_dir = ROOT / "scripts" / "runs" / "causal_xrl"
    run_dir.mkdir(parents=True, exist_ok=True)
    tag = pathlib.Path(ckpt_path).stem
    ts = int(time.time())
    return run_dir / f"eval_{tag}_{mode}_{ts}.jsonl"


def _select_head(strategy: str, fixed_head: int, num_heads: int, step: int) -> int:
    if strategy == "fixed":
        return fixed_head % num_heads
    if strategy == "round_robin":
        return step % num_heads
    raise ValueError(f"未知的 head 策略: {strategy}")


def _to_serializable(obj: Any) -> Any:
    import numpy as _np

    if isinstance(obj, (_np.floating,)):
        return float(obj)
    if isinstance(obj, (_np.integer,)):
        return int(obj)
    if isinstance(obj, _np.ndarray):
        return obj.tolist()
    if isinstance(obj, torch.Tensor):
        return _to_serializable(obj.detach().cpu().numpy())
    if isinstance(obj, (list, tuple)):
        return [_to_serializable(v) for v in obj]
    if isinstance(obj, dict):
        return {k: _to_serializable(v) for k, v in obj.items()}
    return obj


@torch.no_grad()
def run_episode(
    env: CybORGWrapper,
    model: MultiHeadActorCritic,
    mse_solver: MSEApproximation,
    rule_coef: float,
    head_strategy: str,
    fixed_head: int,
    *,
    kb: Optional[KnowledgeBridge],
    mode: str,
) -> Tuple[List[StepRecord], float, float]:
    obs_raw = env.reset()
    facts = obs_raw.get("facts", {}) if isinstance(obs_raw, dict) else {}
    if kb is not None and hasattr(kb, "reset_episode"):
        kb.reset_episode()

    step_logs: List[StepRecord] = []
    total_env = 0.0
    total_reward = 0.0
    step = 0

    action_names = env.action_space.names
    act_dim = env.action_dim

    done = False
    while not done:
        head_idx = _select_head(head_strategy, fixed_head, len(model.actor_heads), step)

        obs_vec = to_obs_vector(obs_raw)
        graph_spec = extract_graph(facts)
        obs_t = torch.from_numpy(obs_vec).to(DEVICE)

        kb_explain: Dict[str, Any] = {}
        if kb is not None:
            if hasattr(kb, "update_from_facts"):
                kb.update_from_facts(facts)
            prior_np, _ = kb.prior_logits(facts, action_names)
            mask_np, _ = kb.query_action_mask(facts, action_names)
        else:
            prior_np = np.zeros(act_dim, dtype=np.float32)
            mask_np = np.ones(act_dim, dtype=np.float32)

        try:
            legal_mask_np = env._current_legal_mask().astype(np.float32)
        except Exception:
            legal_mask_np = np.ones(act_dim, dtype=np.float32)

        prior_np = np.asarray(prior_np, dtype=np.float32)
        kb_mask_np = np.asarray(mask_np, dtype=np.float32)
        combined_mask = (kb_mask_np * legal_mask_np).astype(np.float32)
        if combined_mask.sum() <= 0:
            combined_mask[0] = 1.0

        logits, _, attn = model(obs_t, graph_spec=graph_spec, head_idx=head_idx)
        logits = logits.squeeze(0) if logits.dim() > 1 else logits
        if rule_coef != 0.0:
            logits = logits + torch.from_numpy(prior_np).to(DEVICE) * rule_coef
        else:
            logits = logits + torch.from_numpy(prior_np).to(DEVICE)

        mask_t = torch.from_numpy(combined_mask).to(DEVICE)
        logits = logits.clone()
        logits[mask_t == 0] = -1e9

        dist = Categorical(logits=logits)
        action = dist.sample()
        a_idx = int(action.item())
        a_name = action_names[a_idx]

        obs_raw, r_env, done, info = env.step(a_idx)
        total_env += float(r_env)

        shaped_reward = float(r_env)
        if kb is not None and hasattr(kb, "step_update"):
            try:
                shaped_reward = kb.step_update(facts, a_name, shaped_reward)
            except Exception:
                # KB shaping 出错不影响主流程
                pass
        total_reward += float(shaped_reward)

        if kb is not None:
            try:
                kb_explain = kb.explain_decision(facts, action_names)
            except Exception:
                kb_explain = {}

        attn_np = None if attn is None else attn.detach().cpu().numpy()
        mse_res = mse_solver.approximate(
            target_mask=combined_mask,
            prediction=dist.probs.detach().cpu().numpy(),
            attention=attn_np,
        )

        top_idx = np.argsort(prior_np)[-3:][::-1]
        step_logs.append(
            StepRecord(
                mode=mode,
                step=step + 1,
                head_idx=head_idx,
                action_idx=a_idx,
                action_name=a_name,
                env_reward=float(r_env),
                total_reward=float(shaped_reward),
                done=bool(done),
                legal_mask_sum=float(legal_mask_np.sum()),
                kb_mask_sum=float(kb_mask_np.sum()),
                combined_mask_sum=float(combined_mask.sum()),
                prior_chosen=float(prior_np[a_idx]),
                top_prior=[(action_names[i], float(prior_np[i])) for i in top_idx],
                mse_mode=mse_res.mode,
                mse=float(mse_res.mse),
                evidence_ids=mse_res.evidence_ids,
                facts=facts,
                explain=kb_explain,
            )
        )

        facts = obs_raw.get("facts", {}) if isinstance(obs_raw, dict) else {}
        step += 1

    return step_logs, total_env, total_reward


def main() -> None:
    parser = argparse.ArgumentParser(description="评估 / 对比 causal XRL checkpoint，输出行为日志与平均得分")
    parser.add_argument("--ckpt", nargs="+", required=True, help="train_causal_xrl.py 生成的 checkpoint 路径，可传多个")
    parser.add_argument("--episodes", type=int, default=5, help="每种模式评估回合数，默认 5")
    parser.add_argument(
        "--head-strategy",
        choices=["fixed", "round_robin"],
        default="fixed",
        help="多头策略选择：固定 head 或按步轮询",
    )
    parser.add_argument("--head", type=int, default=0, help="head-strategy=fixed 时使用的 head 索引")
    parser.add_argument(
        "--with-plain",
        action="store_true",
        help="除带 KB 外，再跑一遍 plain_no_kb（无 prior/mask）用于对比",
    )
    args = parser.parse_args()

    env = _build_env()
    obs_dim = env.obs_dim
    act_dim = env.action_dim  # 暂时没用，但保留以防后续扩展

    rule_coef = _load_rule_coef()
    mse_solver = MSEApproximation()

    all_results: List[Dict[str, Any]] = []

    # === 对每一个 ckpt 单独加载模型并评估 ===
    for ckpt_path_raw in args.ckpt:
        ckpt_path = os.path.abspath(ckpt_path_raw)
        if not os.path.exists(ckpt_path):
            print(f"❌ ckpt 不存在：{ckpt_path}")
            continue

        ckpt = torch.load(ckpt_path, map_location="cpu")
        state_dict = ckpt.get("model", ckpt)
        hidden, node_dim, num_heads = _infer_hparams(state_dict)

        model = MultiHeadActorCritic(
            obs_dim=obs_dim,
            act_dim=act_dim,
            hidden=hidden,
            num_heads=num_heads,
            node_dim=node_dim,
        )
        model.load_state_dict(state_dict)
        model.to(DEVICE)
        model.eval()

        print(
            f"\n==============================\n"
            f"🔍 评估模型: {ckpt_path}\n"
            f"heads={num_heads}, hidden={hidden}, node_dim={node_dim}, rule_coef={rule_coef}, device={DEVICE}\n"
            f"=============================="
        )

        modes: Sequence[str] = ["cskg"]
        if args.with_plain:
            modes.append("plain_no_kb")

        for mode in modes:
            use_kb = mode == "cskg"
            kb = _build_kb() if use_kb else None
            log_path = _prepare_logger(ckpt_path, mode)

            env_scores: List[float] = []
            total_scores: List[float] = []
            step_counts: List[int] = []

            with log_path.open("w", encoding="utf-8") as f:
                for ep in range(1, args.episodes + 1):
                    step_logs, env_r, total_r = run_episode(
                        env=env,
                        model=model,
                        mse_solver=mse_solver,
                        rule_coef=rule_coef,
                        head_strategy=args.head_strategy,
                        fixed_head=args.head,
                        kb=kb,
                        mode=mode,
                    )
                    env_scores.append(env_r)
                    total_scores.append(total_r)
                    step_counts.append(len(step_logs))

                    for rec in step_logs:
                        f.write(
                            json.dumps(
                                {
                                    "mode": rec.mode,
                                    "episode": ep,
                                    "step": rec.step,
                                    "head_idx": rec.head_idx,
                                    "action_idx": rec.action_idx,
                                    "action_name": rec.action_name,
                                    "env_reward": rec.env_reward,
                                    "total_reward": rec.total_reward,
                                    "done": rec.done,
                                    "legal_mask_sum": rec.legal_mask_sum,
                                    "kb_mask_sum": rec.kb_mask_sum,
                                    "combined_mask_sum": rec.combined_mask_sum,
                                    "prior_chosen": rec.prior_chosen,
                                    "top_prior": rec.top_prior,
                                    "mse_mode": rec.mse_mode,
                                    "mse": rec.mse,
                                    "evidence_ids": rec.evidence_ids,
                                    "facts": rec.facts,
                                    "explain": rec.explain,
                                },
                                default=_to_serializable,
                                ensure_ascii=False,
                            )
                            + "\n"
                        )

                    print(
                        f"[{mode}] Episode {ep:02d} EnvReward={env_r:.3f} TotalReward={total_r:.3f} "
                        f"steps={len(step_logs)}"
                    )

                env_avg = float(np.mean(env_scores)) if env_scores else 0.0
                total_avg = float(np.mean(total_scores)) if total_scores else 0.0
                mean_steps = float(np.mean(step_counts)) if step_counts else 0.0

                print(
                    f"\n✅ {mode} 评估完成，日志写入 {log_path}\n"
                    f"平均 EnvReward={env_avg:.3f}, 平均 TotalReward={total_avg:.3f}, 平均步长={mean_steps:.2f}, episodes={len(env_scores)}"
                )

                all_results.append(
                    {
                        "ckpt": pathlib.Path(ckpt_path).stem,
                        "mode": mode,
                        "mean_env_reward": env_avg,
                        "mean_total_reward": total_avg,
                        "mean_steps": mean_steps,
                    }
                )

    env.close()

    # === 汇总所有 ckpt + mode 的结果，打印矩阵并写 CSV ===
    if all_results:
        print("\n================ 总体对比汇总 ================")
        all_results.sort(key=lambda x: (x["ckpt"], x["mode"]))
        header = f"{'ckpt':15s} {'mode':12s} {'mean_env':>10s} {'mean_total':>12s} {'mean_steps':>12s}"
        print(header)
        print("-" * len(header))
        for r in all_results:
            print(
                f"{r['ckpt']:15s} {r['mode']:12s} "
                f"{r['mean_env_reward']:10.3f} {r['mean_total_reward']:12.3f} {r['mean_steps']:12.2f}"
            )

        out_dir = ROOT / "scripts" / "runs" / "causal_xrl"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_csv = out_dir / f"eval_compare_{int(time.time())}.csv"
        with out_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["ckpt", "mode", "mean_env_reward", "mean_total_reward", "mean_steps"])
            for r in all_results:
                writer.writerow(
                    [
                        r["ckpt"],
                        r["mode"],
                        f"{r['mean_env_reward']:.6f}",
                        f"{r['mean_total_reward']:.6f}",
                        f"{r['mean_steps']:.6f}",
                    ]
                )

        print(f"\n📄 已将汇总结果写入: {out_csv}")

    print("\n✅ 对比评估完成")


if __name__ == "__main__":
    main()
