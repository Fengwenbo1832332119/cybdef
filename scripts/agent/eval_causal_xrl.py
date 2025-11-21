"""Causal XRL 行为评估脚本。

评估 train_causal_xrl.py 训练出的 GNN+多头策略：
- 载入 MultiHeadActorCritic checkpoint（含 GNN 状态编码、知识桥 prior/mask）
- 按 head 策略（固定/轮询）运行若干回合，记录 EnvReward/TotalReward、掩码统计、prior、MSE 近似等
- 将每一步的详细记录写入 scripts/runs/causal_xrl/eval_*.jsonl，结尾输出平均分
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

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
    return KnowledgeBridge(seed_graph_path=str(seed_graph), cskg_rules_path=str(cskg_yaml), recent_steps=10)


def _prepare_logger(ckpt_path: str) -> pathlib.Path:
    run_dir = ROOT / "scripts" / "runs" / "causal_xrl"
    run_dir.mkdir(parents=True, exist_ok=True)
    tag = pathlib.Path(ckpt_path).stem
    ts = int(time.time())
    return run_dir / f"eval_{tag}_{ts}.jsonl"


def _select_head(strategy: str, fixed_head: int, num_heads: int, step: int) -> int:
    if strategy == "fixed":
        return fixed_head % num_heads
    if strategy == "round_robin":
        return step % num_heads
    raise ValueError(f"未知的 head 策略: {strategy}")


@torch.no_grad()
def run_episode(
    env: CybORGWrapper,
    kb: KnowledgeBridge,
    model: MultiHeadActorCritic,
    mse_solver: MSEApproximation,
    rule_coef: float,
    head_strategy: str,
    fixed_head: int,
) -> Tuple[List[StepRecord], float, float]:
    obs_raw = env.reset()
    facts = obs_raw.get("facts", {}) if isinstance(obs_raw, dict) else {}
    if hasattr(kb, "reset_episode"):
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

        if hasattr(kb, "update_from_facts"):
            kb.update_from_facts(facts)
        prior_np, _ = kb.prior_logits(facts, action_names)
        mask_np, _ = kb.query_action_mask(facts, action_names)

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
        if hasattr(kb, "step_update"):
            try:
                shaped_reward = kb.step_update(facts, a_name, shaped_reward)
            except Exception:
                pass
        total_reward += float(shaped_reward)

        kb_explain = {}
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
            )
        )

        facts = obs_raw.get("facts", {}) if isinstance(obs_raw, dict) else {}
        step += 1

    return step_logs, total_env, total_reward


def main() -> None:
    parser = argparse.ArgumentParser(description="评估 causal XRL checkpoint，输出行为日志与平均得分")
    parser.add_argument("--ckpt", required=True, help="train_causal_xrl.py 生成的 checkpoint 路径")
    parser.add_argument("--episodes", type=int, default=5, help="评估回合数，默认 5")
    parser.add_argument(
        "--head-strategy",
        choices=["fixed", "round_robin"],
        default="fixed",
        help="多头策略选择：固定 head 或按步轮询",
    )
    parser.add_argument("--head", type=int, default=0, help="head-strategy=fixed 时使用的 head 索引")
    args = parser.parse_args()

    ckpt_path = os.path.abspath(args.ckpt)
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"ckpt 不存在：{ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location="cpu")
    state_dict = ckpt.get("model", ckpt)

    env = _build_env()
    obs_dim = env.obs_dim
    act_dim = env.action_dim

    hidden, node_dim, num_heads = _infer_hparams(state_dict)

    model = MultiHeadActorCritic(
        obs_dim=obs_dim, act_dim=act_dim, hidden=hidden, num_heads=num_heads, node_dim=node_dim
    )
    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()

    rule_coef = _load_rule_coef()
    print(
        f"✅ ckpt 加载完成：heads={num_heads}, hidden={hidden}, node_dim={node_dim}, "
        f"rule_coef={rule_coef}, device={DEVICE}"
    )

    kb = _build_kb()
    mse_solver = MSEApproximation()

    log_path = _prepare_logger(ckpt_path)
    env_scores: List[float] = []
    total_scores: List[float] = []

    with log_path.open("w", encoding="utf-8") as f:
        for ep in range(1, args.episodes + 1):
            step_logs, env_r, total_r = run_episode(
                env=env,
                kb=kb,
                model=model,
                mse_solver=mse_solver,
                rule_coef=rule_coef,
                head_strategy=args.head_strategy,
                fixed_head=args.head,
            )
            env_scores.append(env_r)
            total_scores.append(total_r)

            for rec in step_logs:
                f.write(
                    json.dumps(
                        {
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
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )

            print(
                f"[Episode {ep:02d}] EnvReward={env_r:.3f} TotalReward={total_r:.3f} "
                f"steps={len(step_logs)}"
            )

    env_avg = float(np.mean(env_scores)) if env_scores else 0.0
    total_avg = float(np.mean(total_scores)) if total_scores else 0.0
    print(
        f"\n✅ 评估完成，日志写入 {log_path}\n"
        f"平均 EnvReward={env_avg:.3f}, 平均 TotalReward={total_avg:.3f}, episodes={len(env_scores)}"
    )

    env.close()


if __name__ == "__main__":
    main()