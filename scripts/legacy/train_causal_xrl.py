# scripts/agent/train_causal_xrl.py
# -*- coding: utf-8 -*-
"""
Causal XRL training with PPO + KnowledgeBridge + PolicySpeak.

This script demonstrates:
- GNN state encoding (optional graph features from env facts)
- MSE approximation (greedy / gradient-mask / attention-threshold) against
  KnowledgeBridge masks
- PolicySpeak validation with violation penalties
- CVaR/variance risk terms and tail-risk logging
- Multi-strategy support via multi-head actors and simple parallel rollouts

Loss: L = L_PPO + λ|MSE| + μ·Violation + γ·Risk
"""

from __future__ import annotations

import importlib.util
import json
import os
import pathlib
import sys
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

ROOT = pathlib.Path(__file__).resolve().parents[2]
THIRD = ROOT / "third_party" / "CybORG"

for p in (ROOT, THIRD):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from scripts.cskg.reasoner import KnowledgeBridge  # noqa: E402
from scripts.common.validation import MSEApproximation, PolicySpeakValidator  # noqa: E402

if importlib.util.find_spec("envs.cyborg_wrapper") is not None:
    from envs.cyborg_wrapper import CybORGWrapper  # type: ignore  # noqa: E402
else:
    from scripts.envs.cyborg_wrapper import CybORGWrapper  # type: ignore  # noqa: E402

import yaml

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def to_obs_vector(obs_raw: Any) -> np.ndarray:
    if isinstance(obs_raw, dict):
        if "obs_vec" in obs_raw:
            obs_raw = obs_raw["obs_vec"]
        else:
            for key in ["obs", "observation", "vector", "state"]:
                if key in obs_raw:
                    obs_raw = obs_raw[key]
                    break
    if isinstance(obs_raw, dict):
        raise TypeError(f"无法从 obs 字典中提取向量，请检查 keys: {list(obs_raw.keys())}")
    return np.array(obs_raw, dtype=np.float32).reshape(-1)


def extract_graph(facts: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    graph = facts.get("graph") if isinstance(facts, dict) else None
    if graph is None:
        return None
    node_feat = graph.get("node_feat") or graph.get("node_features")
    edge_index = graph.get("edge_index") or graph.get("edges")
    if node_feat is None or edge_index is None:
        return None
    return {"node_feat": node_feat, "edge_index": edge_index}


class GNNStateEncoder(nn.Module):
    def __init__(self, obs_dim: int, hidden: int = 128, node_dim: int = 32) -> None:
        super().__init__()
        self.obs_proj = nn.Linear(obs_dim, hidden)
        self.node_encoder = nn.Linear(node_dim, hidden)
        self.gate = nn.Linear(hidden, 1)
        self.combine = nn.Linear(hidden * 2, hidden)

    def forward(self, obs_vec: torch.Tensor, graph_spec: Optional[Dict[str, Any]] = None):
        if obs_vec.dim() == 1:
            obs_vec = obs_vec.unsqueeze(0)
        obs_feat = torch.tanh(self.obs_proj(obs_vec))

        batch_size = obs_feat.size(0)
        graph_summary = torch.zeros(batch_size, obs_feat.size(-1), device=obs_feat.device)
        attn_weights: Optional[torch.Tensor] = None

        if graph_spec is not None:
            node_feat = torch.as_tensor(graph_spec.get("node_feat", []), dtype=torch.float32, device=obs_feat.device)
            edge_index = torch.as_tensor(graph_spec.get("edge_index", []), dtype=torch.long, device=obs_feat.device)
            if node_feat.dim() == 1:
                node_feat = node_feat.unsqueeze(0)
            if node_feat.numel() > 0:
                encoded = torch.tanh(self.node_encoder(node_feat))
                if edge_index.numel() > 0:
                    edge_index = edge_index.view(-1, 2)
                    src = edge_index[:, 0].clamp(max=encoded.size(0) - 1)
                    dst = edge_index[:, 1].clamp(max=encoded.size(0) - 1)
                    agg = torch.zeros_like(encoded)
                    agg.index_add_(0, dst, encoded[src])
                else:
                    agg = encoded
                node_repr = encoded + agg
                scores = self.gate(torch.tanh(node_repr)).squeeze(-1)
                attn_weights = torch.softmax(scores, dim=0)
                pooled = torch.sum(node_repr * attn_weights.unsqueeze(-1), dim=0)
                graph_summary = pooled.unsqueeze(0).expand(batch_size, -1)

        combined = torch.cat([obs_feat, graph_summary], dim=-1)
        state = torch.tanh(self.combine(combined))
        return state, attn_weights


class MultiHeadActorCritic(nn.Module):
    def __init__(
        self, obs_dim: int, act_dim: int, hidden: int = 128, num_heads: int = 1, node_dim: int = 32
    ) -> None:
        super().__init__()
        self.encoder = GNNStateEncoder(obs_dim, hidden=hidden, node_dim=node_dim)
        self.actor_heads = nn.ModuleList(
            [nn.Linear(hidden, act_dim) for _ in range(max(1, num_heads))]
        )
        self.critic = nn.Sequential(nn.Linear(hidden, hidden), nn.Tanh(), nn.Linear(hidden, 1))

    def forward(
        self, obs_vec: torch.Tensor, graph_spec: Optional[Dict[str, Any]] = None, head_idx: int = 0
    ):
        state, attn = self.encoder(obs_vec, graph_spec)
        head = self.actor_heads[head_idx % len(self.actor_heads)]
        logits = head(state)
        value = self.critic(state).squeeze(-1)
        return logits, value, attn


@dataclass
class Transition:
    obs: np.ndarray
    graph: Optional[Dict[str, Any]]
    action: int
    logp: float
    value: float
    reward: float
    done: float
    head_idx: int


class RiskTracker:
    def __init__(self, alpha: float = 0.2) -> None:
        self.alpha = alpha
        self.history: List[Dict[str, float]] = []

    def record_batch(self, returns: Sequence[float]) -> float:
        if len(returns) == 0:
            return 0.0
        ret_np = np.asarray(returns, dtype=np.float32)
        tail_q = float(np.quantile(ret_np, 1 - self.alpha))
        self.history.append({"tail_quantile": tail_q, "mean": float(ret_np.mean())})
        return tail_q

    def risk_value(self, values: torch.Tensor) -> torch.Tensor:
        if values.numel() <= 1:
            return torch.tensor(0.0, device=values.device)
        variance = torch.var(values)
        tail_k = max(1, int(self.alpha * values.numel()))
        tail_values = torch.topk(-values, k=tail_k).values
        cvar = tail_values.mean()
        return variance + cvar


def main() -> None:
    global DEVICE

    ENV_YAML = ROOT / "scripts" / "configs" / "env.yaml"
    CSKG_YAML = ROOT / "scripts" / "configs" / "cskg.yaml"
    SEED_GRAPH = ROOT / "scripts" / "configs" / "seed_graph.json"
    PPO_YAML = ROOT / "scripts" / "configs" / "ppo.yaml"

    RUN_NAME = f"causal_xrl_{int(time.time())}"
    OUT_DIR = ROOT / "scripts" / "runs" / "causal_xrl" / RUN_NAME
    os.makedirs(OUT_DIR, exist_ok=True)

    if PPO_YAML.exists():
        with PPO_YAML.open("r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
    else:
        cfg = {}

    num_updates = int(cfg.get("num_updates", 50))
    rollout_steps = int(cfg.get("horizon", 128))
    ppo_epochs = int(cfg.get("ppo_epochs", 3))
    batch_size = int(cfg.get("mini_batch_size", 64))
    gamma = float(cfg.get("gamma", 0.99))
    lam = float(cfg.get("gae_lambda", 0.95))
    clip_ratio = float(cfg.get("clip_range", 0.2))

    lr = float(cfg.get("pi_lr", 3e-4))
    entropy_coef = float(cfg.get("entropy_coef", 0.01))
    value_coef = float(cfg.get("value_coef", 0.5))
    rule_coef = float(cfg.get("rule_coef", 0.0))
    max_grad_norm = float(cfg.get("max_grad_norm", 0.5))
    mse_lambda = float(cfg.get("mse_lambda", 1.0))
    violation_mu = float(cfg.get("violation_mu", 0.5))
    risk_gamma = float(cfg.get("risk_gamma", 0.2))
    risk_alpha = float(cfg.get("risk_alpha", 0.2))
    mse_mode = str(cfg.get("mse_mode", "greedy"))

    num_envs = int(cfg.get("num_envs", 2))
    num_heads = int(cfg.get("num_heads", 2))
    node_dim = int(cfg.get("node_dim", 32))

    device_cfg = str(cfg.get("device", "cuda")).lower()
    if device_cfg == "cuda" and torch.cuda.is_available():
        DEVICE = torch.device("cuda")
    else:
        DEVICE = torch.device("cpu")

    print(f"📋 Config from {PPO_YAML}")
    print(
        f"updates={num_updates}, horizon={rollout_steps}, batch={batch_size}, epochs={ppo_epochs},"
        f" mse_mode={mse_mode}, λ={mse_lambda}, μ={violation_mu}, γ={risk_gamma}"
    )
    print(f"num_envs={num_envs}, num_heads={num_heads}, device={DEVICE}")

    envs = [CybORGWrapper(str(ENV_YAML)) for _ in range(num_envs)]
    obs_dim = envs[0].obs_dim
    act_dim = envs[0].action_dim

    kbs = [
        KnowledgeBridge(
            seed_graph_path=str(SEED_GRAPH),
            cskg_rules_path=str(CSKG_YAML),
            recent_steps=10,
        )
        for _ in range(num_envs)
    ]

    model = MultiHeadActorCritic(
        obs_dim, act_dim, hidden=128, num_heads=num_heads, node_dim=node_dim
    ).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Validation utilities shared with other scripts
    mse_solver = MSEApproximation(default_mode=mse_mode)
    policyspeak = PolicySpeakValidator()
    risk_tracker = RiskTracker(alpha=risk_alpha)

    risk_log = (OUT_DIR / "tail_risk.jsonl").open("w", encoding="utf-8")
    ps_log = (OUT_DIR / "policyspeak.jsonl").open("w", encoding="utf-8")
    mse_log = (OUT_DIR / "mse.jsonl").open("w", encoding="utf-8")

    obs_raw_list: List[Dict[str, Any]] = []
    facts_list: List[Dict[str, Any]] = []
    for env in envs:
        obs_raw = env.reset()
        obs_raw_list.append(obs_raw)
        facts_list.append(obs_raw.get("facts", {}))

    global_step = 0

    for upd in range(1, num_updates + 1):
        transitions: List[Transition] = []
        mse_buf: List[float] = []
        violation_buf: List[float] = []
        ep_reward_env = [0.0 for _ in range(num_envs)]
        ep_reward_total = [0.0 for _ in range(num_envs)]

        steps = 0
        while steps < rollout_steps:
            for env_idx, env in enumerate(envs):
                if steps >= rollout_steps:
                    break
                facts = facts_list[env_idx]
                obs_vec = to_obs_vector(obs_raw_list[env_idx])
                graph_spec = extract_graph(facts)

                obs_tensor = torch.from_numpy(obs_vec).to(DEVICE)
                head_idx = env_idx % num_heads
                logits, value, attn = model(obs_tensor, graph_spec=graph_spec, head_idx=head_idx)
                action_names = env.action_space.names

                kb = kbs[env_idx]
                if hasattr(kb, "update_from_facts"):
                    kb.update_from_facts(facts)
                prior_np, _ = kb.prior_logits(facts, action_names)
                mask_np, _ = kb.query_action_mask(facts, action_names)

                try:
                    legal_mask_np = env._current_legal_mask().astype(np.float32)
                except Exception:
                    legal_mask_np = np.ones(act_dim, dtype=np.float32)

                prior_np = np.asarray(prior_np, dtype=np.float32)
                mask_np = np.asarray(mask_np, dtype=np.float32)
                combined_mask_np = (mask_np * legal_mask_np).astype(np.float32)
                if combined_mask_np.sum() <= 0:
                    combined_mask_np[0] = 1.0

                logits = logits.squeeze(0) if logits.dim() > 1 else logits
                logits = logits + torch.from_numpy(prior_np).to(DEVICE) * (rule_coef if rule_coef != 0 else 1.0)
                combined_mask_t = torch.from_numpy(combined_mask_np).to(DEVICE)
                logits[combined_mask_t == 0] = -1e9

                dist = Categorical(logits=logits)
                action = dist.sample()
                logp = dist.log_prob(action)

                action_idx = int(action.item())
                action_name = action_names[action_idx]
                next_obs_raw, reward_env, done, info = env.step(action_idx)

                shaped_reward = reward_env
                if hasattr(kb, "step_update"):
                    shaped_reward = kb.step_update(facts, action_name, float(reward_env))

                kb_explain = {}
                try:
                    kb_explain = kb.explain_decision(facts, action_names)
                except Exception:
                    kb_explain = {}

                attn_np = None if attn is None else attn.detach().cpu().numpy()
                mse_res = mse_solver.approximate(
                    target_mask=combined_mask_np,
                    prediction=dist.probs.detach().cpu().numpy(),
                    attention=attn_np,
                )
                mse_buf.append(mse_res.mse)
                mse_log.write(
                    json.dumps(
                        {
                            "step": global_step,
                            "update": upd,
                            "env_idx": env_idx,
                            "mode": mse_res.mode,
                            "mse": mse_res.mse,
                            "evidence_ids": mse_res.evidence_ids,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )

                justification = ""
                if isinstance(info, dict):
                    justification = info.get("reason", "") or info.get("explanation", "")
                if not justification:
                    active_rules = kb_explain.get("active_mask_rules") or kb_explain.get(
                        "active_prior_rules", []
                    )
                    if active_rules:
                        justification = json.dumps(active_rules, ensure_ascii=False)

                ps_statement = policyspeak.build_statement(
                    action=action_name,
                    violation=bool(info.get("violation", False) if isinstance(info, dict) else False),
                    justification=justification,
                )
                violation_penalty, ps_aligned = policyspeak.violation_penalty(kb_explain, [ps_statement])
                violation_buf.append(violation_penalty)

                ps_log.write(
                    json.dumps(
                        {
                            "step": global_step,
                            "update": upd,
                            "env_idx": env_idx,
                            "payload": policyspeak.loggable_payload(kb_explain, ps_aligned),
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )

                reward_total = shaped_reward
                ep_reward_env[env_idx] += float(reward_env)
                ep_reward_total[env_idx] += float(reward_total)

                transitions.append(
                    Transition(
                        obs=obs_vec.copy(),
                        graph=graph_spec,
                        action=action_idx,
                        logp=float(logp.item()),
                        value=float(value.squeeze().item()),
                        reward=float(reward_total),
                        done=float(done),
                        head_idx=head_idx,
                    )
                )

                obs_raw_list[env_idx] = next_obs_raw
                facts_list[env_idx] = next_obs_raw.get("facts", {})
                if done:
                    obs_raw_list[env_idx] = env.reset()
                    facts_list[env_idx] = obs_raw_list[env_idx].get("facts", {})

                global_step += 1
                steps += 1

        if not transitions:
            continue

        rewards = np.array([t.reward for t in transitions], dtype=np.float32)
        values = np.array([t.value for t in transitions], dtype=np.float32)
        dones = np.array([t.done for t in transitions], dtype=np.float32)

        values_ext = np.concatenate([values, np.array([0.0], dtype=np.float32)], axis=0)
        adv = np.zeros_like(rewards)
        gae = 0.0
        for t in reversed(range(len(rewards))):
            next_v = values_ext[t + 1]
            delta = rewards[t] + gamma * next_v * (1.0 - dones[t]) - values[t]
            gae = delta + gamma * lam * (1.0 - dones[t]) * gae
            adv[t] = gae
        returns = adv + values
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        num_samples = len(transitions)
        idxs = np.arange(num_samples)

        obs_arr = [t.obs for t in transitions]
        graph_arr = [t.graph for t in transitions]
        act_arr = [t.action for t in transitions]
        logp_old_arr = [t.logp for t in transitions]
        head_arr = [t.head_idx for t in transitions]

        adv_tensor = torch.from_numpy(adv).to(DEVICE)
        ret_tensor = torch.from_numpy(returns).to(DEVICE)
        logp_old_tensor = torch.from_numpy(np.array(logp_old_arr, dtype=np.float32)).to(DEVICE)

        for _ in range(ppo_epochs):
            np.random.shuffle(idxs)
            for start in range(0, num_samples, batch_size):
                end = start + batch_size
                batch_idx = idxs[start:end]

                logits_list: List[torch.Tensor] = []
                values_list: List[torch.Tensor] = []
                for bi in batch_idx:
                    obs_t = torch.from_numpy(obs_arr[bi]).to(DEVICE)
                    graph_spec = graph_arr[bi]
                    logits_b, value_b, _ = model(obs_t, graph_spec=graph_spec, head_idx=head_arr[bi])
                    logits_list.append(logits_b.squeeze(0) if logits_b.dim() > 1 else logits_b)
                    values_list.append(value_b if value_b.dim() == 0 else value_b.squeeze(0))

                logits_batch = torch.stack(logits_list)
                values_pred = torch.stack(values_list)

                b_act = torch.from_numpy(np.array([act_arr[i] for i in batch_idx], dtype=np.int64)).to(DEVICE)
                b_logp_old = logp_old_tensor[batch_idx]
                b_adv = adv_tensor[batch_idx]
                b_ret = ret_tensor[batch_idx]

                dist = Categorical(logits=logits_batch)
                logp = dist.log_prob(b_act)
                entropy = dist.entropy().mean()

                ratio = torch.exp(logp - b_logp_old)
                surr1 = ratio * b_adv
                surr2 = torch.clamp(ratio, 1.0 - clip_ratio, 1.0 + clip_ratio) * b_adv
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = ((values_pred - b_ret) ** 2).mean()

                mse_term = torch.tensor(np.mean(mse_buf) if mse_buf else 0.0, device=DEVICE)
                violation_term = torch.tensor(
                    np.mean(violation_buf) if violation_buf else 0.0, device=DEVICE
                )
                risk_term = risk_tracker.risk_value(b_ret)

                loss = (
                    policy_loss
                    + value_coef * value_loss
                    - entropy_coef * entropy
                    + mse_lambda * torch.abs(mse_term)
                    + violation_mu * violation_term
                    + risk_gamma * risk_term
                )

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()

        tail_q = risk_tracker.record_batch(returns.tolist())
        risk_log.write(json.dumps({"update": upd, "tail_quantile": tail_q}) + "\n")

        mean_env_r = np.mean(ep_reward_env)
        mean_train_r = np.mean(ep_reward_total)
        print(
            f"[UPD {upd:03d}] steps={len(transitions):4d} Env_R={mean_env_r:.3f} "
            f"Train_R={mean_train_r:.3f} | MSE={np.mean(mse_buf) if mse_buf else 0:.4f} "
            f"Violation={np.mean(violation_buf) if violation_buf else 0:.4f} "
            f"TailQ={tail_q:.4f}"
        )

        if upd % 25 == 0:
            ckpt_path = OUT_DIR / f"model_upd{upd:03d}.pt"
            torch.save(
                {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "update": upd,
                    "global_step": global_step,
                },
                ckpt_path,
            )
            print(f" 💾 已保存 checkpoint: {ckpt_path}")

    risk_log.close()
    ps_log.close()
    mse_log.close()
    for env in envs:
        env.close()
    print("✅ 训练结束")


if __name__ == "__main__":
    main()