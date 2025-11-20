# scripts/agent/train_ppo_cskg_old.py
# -*- coding: utf-8 -*-
"""
PPO + CSKG(KnowledgeBridge) 联合训练脚本

- 使用 CybORGWrapper 包装环境
- 使用 KnowledgeBridge 注入：
    - 动作掩码（action_mask）
    - 先验 logits（prior_logits）
    - 奖励塑形（reward_shaping）
- 保留 PPO 足够自由度，让策略自己学，而不是被规则“锁死”
"""

import os, sys, json, time, pathlib, random
from collections import deque
from typing import Any, Dict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

# ===== 路径注入 =====
ROOT = pathlib.Path(__file__).resolve().parents[2]  # C:\cybdef
THIRD = ROOT / "third_party" / "CybORG"

for p in (ROOT, THIRD):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

# ===== 项目内 import =====
# env wrapper
try:
    from envs.cyborg_wrapper import CybORGWrapper
except ImportError:
    from scripts.envs.cyborg_wrapper import CybORGWrapper

# CSKG reasoner
from scripts.cskg.reasoner import KnowledgeBridge

import yaml

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def to_serializable(obj):
    """
    递归把 numpy 类型、ndarray 等，转成 Python 原生类型，方便 json.dumps
    """
    import numpy as _np

    if isinstance(obj, (_np.floating,)):
        return float(obj)
    if isinstance(obj, (_np.integer,)):
        return int(obj)
    if isinstance(obj, _np.ndarray):
        return obj.tolist()
    if isinstance(obj, (list, tuple)):
        return [to_serializable(v) for v in obj]
    if isinstance(obj, dict):
        return {k: to_serializable(v) for k, v in obj.items()}
    return obj


# ===== 简单 Actor-Critic 网络 =====
class ActorCritic(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden=128):
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, act_dim),
        )
        self.critic = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1),
        )

    def forward(self, obs):
        logits = self.actor(obs)
        value = self.critic(obs)
        return logits, value.squeeze(-1)


# ===== GAE 计算 =====
def compute_gae(rewards, values, dones, gamma=0.99, lam=0.95):
    """
    rewards, values, dones: np.ndarray, shape [T]
    """
    T = len(rewards)
    adv = np.zeros(T, dtype=np.float32)
    gae = 0.0
    for t in reversed(range(T)):
        next_v = values[t + 1] if t + 1 < T else 0.0
        delta = rewards[t] + gamma * next_v * (1.0 - dones[t]) - values[t]
        gae = delta + gamma * lam * (1.0 - dones[t]) * gae
        adv[t] = gae
    return adv


def to_obs_vector(obs_raw: Any) -> np.ndarray:
    """
    把 env.reset()/step() 返回的各种结构，统一转成 1D np.array(float32)

    支持几种常见形式：
    - dict:
        - 包含 "obs_vec"（BlueTableWrapper 风格）
        - 或包含 "obs"/"observation"/"vector"/"state" 之一
    - 其它：直接 np.array(...)
    """
    if isinstance(obs_raw, dict):
        # 优先走你包装好的 obs_vec
        if "obs_vec" in obs_raw:
            obs_raw = obs_raw["obs_vec"]
        else:
            # 通用兜底：兼容老版本
            for key in ["obs", "observation", "vector", "state"]:
                if key in obs_raw:
                    obs_raw = obs_raw[key]
                    break

    # 如果还是字典，说明没法拿到向量
    if isinstance(obs_raw, dict):
        raise TypeError(
            f"无法从 obs 字典中提取向量，请检查 keys: {list(obs_raw.keys())}"
        )

    obs_np = np.array(obs_raw, dtype=np.float32).reshape(-1)
    return obs_np


# ===== 主训练函数 =====
def main():
    global DEVICE

    # --- 配置路径 ---
    ENV_YAML = ROOT / "scripts" / "configs" / "env.yaml"
    CSKG_YAML = ROOT / "scripts" / "configs" / "cskg.yaml"
    SEED_GRAPH = ROOT / "scripts" / "configs" / "seed_graph.json"
    PPO_YAML = ROOT / "scripts" / "configs" / "ppo.yaml"

    RUN_NAME = f"ppo_cskg_{int(time.time())}"
    OUT_DIR = ROOT / "scripts" / "runs" / "ppo_cskg" / RUN_NAME
    os.makedirs(OUT_DIR, exist_ok=True)

    # --- 从 ppo.yaml 读取超参 ---
    if PPO_YAML.exists():
        with open(PPO_YAML, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
    else:
        cfg = {}
        print(f"⚠ 未找到 {PPO_YAML}，将使用代码内默认超参")

    num_updates = int(cfg.get("num_updates", 100))        # 训练轮数（原 total_episodes）
    rollout_steps = int(cfg.get("horizon", 256))          # 每轮采样步数（原 rollout_steps）
    ppo_epochs = int(cfg.get("ppo_epochs", 4))
    batch_size = int(cfg.get("mini_batch_size", 64))
    gamma = float(cfg.get("gamma", 0.99))
    lam = float(cfg.get("gae_lambda", 0.95))
    clip_ratio = float(cfg.get("clip_range", 0.2))

    lr_pi = float(cfg.get("pi_lr", 3e-4))
    lr_vf = float(cfg.get("vf_lr", lr_pi))  # 目前仍共用一个 optimizer
    lr = lr_pi

    entropy_coef = float(cfg.get("entropy_coef", 0.01))
    value_coef = float(cfg.get("value_coef", 0.5))
    rule_coef = float(cfg.get("rule_coef", 0.0))          # 用于缩放 prior logits
    mask_alpha = float(cfg.get("mask_alpha", 1.0))        # 目前先预留，不强行使用
    max_grad_norm = float(cfg.get("max_grad_norm", 0.5))

    device_cfg = str(cfg.get("device", "cuda")).lower()

    # --- 设备选择：优先按 ppo.yaml，但要保证可用 ---
    if device_cfg == "cuda" and torch.cuda.is_available():
        DEVICE = torch.device("cuda")
    else:
        DEVICE = torch.device("cpu")

    print(f"📋 PPO 配置来自: {PPO_YAML}")
    print(f"   num_updates={num_updates}, horizon={rollout_steps}, "
          f"mini_batch_size={batch_size}, ppo_epochs={ppo_epochs}")
    print(f"   gamma={gamma}, gae_lambda={lam}, clip_range={clip_ratio}")
    print(f"   pi_lr={lr_pi}, vf_lr={lr_vf}, entropy_coef={entropy_coef}, value_coef={value_coef}")
    print(f"   rule_coef={rule_coef}, mask_alpha={mask_alpha}, max_grad_norm={max_grad_norm}")
    print(f"   device={DEVICE}")

    # 固定随机种子
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # --- 初始化环境 ---
    env = CybORGWrapper(str(ENV_YAML))
    obs_dim = env.obs_dim
    act_dim = env.action_dim

    print(f"✅ PPO+CSKG 训练初始化完成")
    print(f"   obs_dim={obs_dim}, act_dim={act_dim}")
    print(f"   日志目录: {OUT_DIR}")

    # --- 初始化 CSKG ---
    kb = KnowledgeBridge(
        seed_graph_path=str(SEED_GRAPH),
        cskg_rules_path=str(CSKG_YAML),
        recent_steps=10,
    )

    # --- 初始化策略网络 ---
    ac = ActorCritic(obs_dim, act_dim).to(DEVICE)
    optimizer = optim.Adam(ac.parameters(), lr=lr)

    # --- 可解释日志：前 N 次 update 详细记录 ---
    explain_log_path = OUT_DIR / "policy_explain_upd1_5.jsonl"
    explain_log_f = open(explain_log_path, "w", encoding="utf-8")

    global_step = 0

    # ===== 训练主循环：以 num_updates 为外层轮数 =====
    for upd in range(1, num_updates + 1):
        # env.reset() 返回 dict: {"obs_vec", "facts", "raw", ...}
        obs_raw = env.reset()
        if hasattr(kb, "reset_episode"):
            kb.reset_episode()

        # 神经网络用的向量观测
        obs_vec = to_obs_vector(obs_raw)
        # 规则引擎用的语义 facts（来自 wrapper，而不是再从 obs_vec 反推）
        facts = obs_raw.get("facts", {})

        # rollout buffer
        obs_buf = []
        act_buf = []
        logp_buf = []
        rew_buf = []
        val_buf = []
        done_buf = []

        ep_reward_env = 0.0
        ep_reward_total = 0.0

        last_reward_env = 0.0  # 如果你在 _extract_facts 里想用 recent_reward，可以从这里喂

        # 一次 update 内采样 rollout_steps 步（可能跨 episode，中途 done 就重置）
        steps_collected = 0
        while steps_collected < rollout_steps:
            global_step += 1

            # === 策略网络前向 ===
            obs_tensor = torch.from_numpy(obs_vec).to(DEVICE).unsqueeze(0)
            logits, value = ac(obs_tensor)  # [1, act_dim], [1]
            logits = logits.squeeze(0)  # [act_dim]
            value = value.squeeze(0)  # scalar

            action_names = env.action_space.names

            # === CSKG: 先用 facts 更新内部状态（可选）===
            if hasattr(kb, "update_from_facts"):
                kb.update_from_facts(facts)

            # === 从 KB 获取先验与掩码 ===
            prior_np = kb.prior_logits(facts, action_names)
            # 有些版本可能返回 (prior, debug_info)
            if isinstance(prior_np, tuple):
                prior_np = prior_np[0]
            prior_np = np.array(prior_np, dtype=np.float32)

            mask_res = kb.query_action_mask(facts, action_names)
            if isinstance(mask_res, tuple):
                rule_mask_np = np.array(mask_res[0], dtype=np.float32)
            else:
                rule_mask_np = np.array(mask_res, dtype=np.float32)

            # 环境自带合法掩码
            try:
                legal_mask_np = env._current_legal_mask().astype(np.float32)
            except Exception:
                # 如果没有该接口，就假设全部合法
                legal_mask_np = np.ones(act_dim, dtype=np.float32)

            if rule_mask_np.shape[0] != act_dim:
                raise ValueError(f"rule_mask 维度异常: {rule_mask_np.shape[0]} vs act_dim={act_dim}")
            if prior_np.shape[0] != act_dim:
                raise ValueError(f"prior 维度异常: {prior_np.shape[0]} vs act_dim={act_dim}")
            if legal_mask_np.shape[0] != act_dim:
                raise ValueError(f"legal_mask 维度异常: {legal_mask_np.shape[0]} vs act_dim={act_dim}")

            # 融合掩码（环境 × 规则）
            combined_mask_np = (legal_mask_np * rule_mask_np).astype(np.float32)
            if combined_mask_np.sum() <= 0:
                # 极端情况：全 0，就放开一个 no-op（Sleep=0）
                combined_mask_np[0] = 1.0

            # ==== logits + prior + mask ====
            logits = logits.clone()
            prior = torch.from_numpy(prior_np).to(DEVICE)

            # 融合先验（带 rule_coef）
            if rule_coef != 0.0:
                logits = logits + rule_coef * prior
            else:
                logits = logits + prior

            # 掩码：combined_mask == 0 的动作视为不可选
            combined_mask_t = torch.from_numpy(combined_mask_np).to(DEVICE)
            logits[combined_mask_t == 0] = -1e9

            dist = Categorical(logits=logits)
            action = dist.sample()
            logp = dist.log_prob(action)

            action_idx = int(action.item())
            action_name = action_names[action_idx]

            # === 与环境交互 ===
            next_obs_raw, reward_env, done, info = env.step(action_idx)

            # === 修正：正确的奖励塑形 ===
            env_reward = float(reward_env)  # 环境原始奖励（真实性能）

            # CSKG奖励塑形（基于环境奖励）
            if hasattr(kb, "step_update"):
                shaped_reward = kb.step_update(facts, action_name, env_reward)
            else:
                shaped_reward = env_reward

            # 关键设计：训练用塑形奖励，评估用环境奖励
            if env.mode == "train":
                r_total = shaped_reward  # PPO用CSKG指导的训练信号
            else:
                r_total = env_reward  # 评估时用真实环境奖励

            # === 修正：正确的KB状态更新 ===
            next_facts = next_obs_raw.get("facts", {})
            # 如果有专门的KB状态更新方法，在这里调用（但step_update可能已经处理了）
            # 注意：我们不再重复调用step_update，因为它已经返回了塑形奖励

            last_reward_env = env_reward  # 用于_extract_facts的recent_reward

            # ==== 写入 rollout buffer（用 r_total 来训练 PPO） ====
            obs_buf.append(obs_vec.copy())
            act_buf.append(action_idx)
            logp_buf.append(float(logp.item()))
            rew_buf.append(float(r_total))  # PPO用训练信号
            val_buf.append(float(value.item()))
            done_buf.append(float(done))

            # 分别记录两种奖励用于分析
            ep_reward_env += env_reward  # 真实环境表现
            ep_reward_total += r_total  # 实际训练信号

            steps_collected += 1

            # === 可解释日志：前 5 次 update 详细记录 ===
            if upd <= 5:
                top_idx = np.argsort(prior_np)[-3:][::-1]
                top_prior = [
                    [action_names[i], float(prior_np[i])]
                    for i in top_idx
                ]
                try:
                    explain = kb.explain_decision(facts, action_names)
                except Exception:
                    explain = {}

                explain_rec = {
                    "update": upd,
                    "step": steps_collected,
                    "global_step": global_step,
                    "action_idx": action_idx,
                    "action_name": action_name,
                    "reward_env": float(env_reward),  # 记录环境奖励
                    "reward_shaped": float(shaped_reward),  # 记录塑形奖励
                    "reward_total": float(r_total),  # 记录实际训练信号
                    "legal_mask_sum": float(legal_mask_np.sum()),
                    "rule_mask_sum": float(rule_mask_np.sum()),
                    "combined_mask_sum": float(combined_mask_np.sum()),
                    "top_prior": top_prior,
                    "fact": facts,
                    "explain": explain,
                }
                explain_log_f.write(
                    json.dumps(to_serializable(explain_rec), ensure_ascii=False) + "\n"
                )

            # === 准备下一步 ===
            obs_vec = to_obs_vector(next_obs_raw)  # 仅用于 NN
            facts = next_facts  # 下一步规则使用的 facts

            # 如果 episode 结束，重置环境 + KB，但继续本次 update 直到收满 horizon
            if done and steps_collected < rollout_steps:
                obs_raw = env.reset()
                if hasattr(kb, "reset_episode"):
                    kb.reset_episode()
                obs_vec = to_obs_vector(obs_raw)
                facts = obs_raw.get("facts", {})
                last_reward_env = 0.0

        # ===== 一次 update 结束：PPO 更新 =====
        T = len(rew_buf)
        if T == 0:
            continue

        rewards = np.array(rew_buf, dtype=np.float32)
        values = np.array(val_buf, dtype=np.float32)
        dones = np.array(done_buf, dtype=np.float32)

        # 末值 bootstrap = 0（这里简单处理）
        values_ext = np.concatenate([values, np.array([0.0], dtype=np.float32)], axis=0)

        adv = compute_gae(rewards, values_ext, dones, gamma=gamma, lam=lam)
        returns = adv + values

        # 标准化优势
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        # 转 tensor
        obs_tensor = torch.from_numpy(np.array(obs_buf, dtype=np.float32)).to(DEVICE)
        act_tensor = torch.from_numpy(np.array(act_buf, dtype=np.int64)).to(DEVICE)
        logp_old_tensor = torch.from_numpy(np.array(logp_buf, dtype=np.float32)).to(DEVICE)
        adv_tensor = torch.from_numpy(adv).to(DEVICE)
        ret_tensor = torch.from_numpy(returns).to(DEVICE)

        # 多 epoch 打乱训练
        num_samples = T
        idxs = np.arange(num_samples)

        for _ in range(ppo_epochs):
            np.random.shuffle(idxs)
            for start in range(0, num_samples, batch_size):
                end = start + batch_size
                batch_idx = idxs[start:end]

                b_obs = obs_tensor[batch_idx]
                b_act = act_tensor[batch_idx]
                b_logp_old = logp_old_tensor[batch_idx]
                b_adv = adv_tensor[batch_idx]
                b_ret = ret_tensor[batch_idx]

                logits, values_pred = ac(b_obs)
                dist = Categorical(logits=logits)
                logp = dist.log_prob(b_act)
                entropy = dist.entropy().mean()

                ratio = torch.exp(logp - b_logp_old)
                surr1 = ratio * b_adv
                surr2 = torch.clamp(ratio, 1.0 - clip_ratio, 1.0 + clip_ratio) * b_adv
                policy_loss = -torch.min(surr1, surr2).mean()

                value_loss = ((values_pred - b_ret) ** 2).mean()

                loss = policy_loss + value_coef * value_loss - entropy_coef * entropy

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(ac.parameters(), max_grad_norm)
                optimizer.step()

        # 打印时显示两种奖励
        print(
            f"[UPD {upd:03d}] steps={T:4d}  "
            f"Env_R={ep_reward_env:.3f}  Shaped_R={ep_reward_total:.3f}"
        )

        # 简单保存 checkpoint（每 25 次 update）
        if upd % 25 == 0:
            ckpt_path = OUT_DIR / f"ac_upd{upd:03d}.pt"
            torch.save(
                {
                    "model": ac.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "update": upd,
                    "global_step": global_step,
                },
                ckpt_path,
            )
            print(f" 💾 已保存 checkpoint: {ckpt_path}")

    explain_log_f.close()
    env.close()
    print("✅ 训练结束")


if __name__ == "__main__":
    main()
