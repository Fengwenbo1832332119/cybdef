# Agent scripts usage

This repository provides three PPO training entry points plus a unified evaluator. Use the modules directly so imports resolve correctly.

## Training

Each launcher already binds to its matching config under `scripts/configs/`.

- **Plain PPO (B0)**
  ```bash
  python -m scripts.agent.train_ppo_plain
  ```

- **PPO + CSKG (B1)**
  ```bash
  python -m scripts.agent.train_ppo_cskg --config scripts/configs/b1.yaml
  ```
  Omit `--config` to fall back to `b1.yaml` automatically.

- **PPO + CSKG + RAG-LLM 解释 (B2)**
  ```bash
  python -m scripts.agent.train_ppo_rag
  ```

Checkpoints and run metadata are written under `scripts/runs/<run_prefix>/<run_id>/`. Default checkpoints save every 25 updates as `ac_updXXX.pt` and are suitable for downstream evaluation.

## Evaluation

After training, pick the checkpoint you want to assess and run the unified evaluator. It supports step-count sweeps; by default it will evaluate at 30/50/100 steps per episode.

```bash
python -m scripts.agent.eval_benchmark \
  --model scripts/runs/ppo_cskg/ppo_cskg_exp_<timestamp>/ac_upd025.pt \
  --config scripts/configs/b1.yaml \
  --episodes 5 \
  --repeats 3 \
  --num-steps 30 50 100
```

Replace `--model` with the checkpoint from your chosen variant:
- B0 plain: checkpoint from `scripts/agent/train_ppo_plain.py` run (outputs under `scripts/runs/ppo_plain/...`).
- B1 CSKG: checkpoint from `scripts/agent/train_ppo_cskg.py` (defaults to `ppo_cskg` prefix).
- B2 RAG: checkpoint from `scripts/agent/train_ppo_rag.py` (defaults to `ppo_rag` prefix).

Evaluation logs are stored in `scripts/logs/` as JSON containing per-step summaries and aggregated mean/variance.

### Legacy CybORG behaviour replay
If you need the original CybORG evaluation that sweeps adversaries (BLine/Meander/Sleep) and step counts, enable legacy mode:

```bash
python -m scripts.agent.eval_benchmark \
  --legacy-cyborg \
  --num-steps 30 50 100 \
  --legacy-episodes 100
```
This uses the competition-style `LoadBanditBlueAgent` against each built-in Red agent and writes a plain-text log under `scripts/logs/legacy_eval_<timestamp>.txt`.