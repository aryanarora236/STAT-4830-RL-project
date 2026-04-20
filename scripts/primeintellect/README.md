# PrimeIntellect Training Playbook

End-to-end recipe for running the full BC → REINFORCE → Eval pipeline on a PrimeIntellect GPU pod. Everything here is driven by `scripts/poker_train.py` (already in the repo); these helpers only wrap pod launch, environment bootstrap, and result retrieval.

## 1. One-time local setup

```bash
# CLI (once per laptop)
curl -fsSL https://get.primeintellect.ai | sh
prime login
prime config set-ssh-key-path ~/.ssh/id_ed25519   # or whichever key you registered
```

Export the Hugging Face token that will be copied onto the pod (Qwen downloads need it):

```bash
export HF_TOKEN=hf_xxx   # put in ~/.zshrc if you want it permanent
```

## 2. Pick a GPU

```bash
# Cheapest Ampere+ GPU that fits Qwen-1.5B + 4-bit LoRA comfortably:
prime availability list --gpu-type A100_80GB --regions united_states
# Bigger runs (Qwen-7B BC, or longer RL) — prefer H100:
prime availability list --gpu-type H100_80GB --regions united_states
```

Note the short `id` from the row you want (e.g. `346663`). Total runtime for the default pipeline on H100_80GB is ~2 hours (~$4-8 depending on the row you pick), A100_80GB is ~3 hours.

## 3. (Optional) Create a persistent disk for checkpoints

Only bother if you expect to iterate across multiple pod sessions. Otherwise keep everything on the pod's ephemeral disk and `scp` results back before termination.

```bash
prime availability disks --regions united_states
prime disks create --id <provider-id> --size 100 --name poker-checkpoints
```

## 4. Launch the pod

```bash
prime pods create \
  --id <availability-id> \
  --gpu-count 1 \
  --disk-size 100 \
  --name stat4830-poker \
  --env HF_TOKEN=$HF_TOKEN \
  --env WANDB_DISABLED=true
  # optionally: --disks <disk-id-from-step-3>
```

`prime pods list` to see the pod id once it's `ACTIVE`, then:

```bash
prime pods ssh <pod-id>
```

## 5. Bootstrap the pod (inside the SSH session)

```bash
curl -sSL https://raw.githubusercontent.com/aryanarora236/STAT-4830-RL-project/main/scripts/primeintellect/bootstrap.sh | bash
cd /root/STAT-4830-RL-project
source .venv/bin/activate
```

If you'd rather read the script before running it, `cat scripts/primeintellect/bootstrap.sh` after cloning manually.

## 6. Run the pipeline

The canonical run for the final presentation is:

```bash
bash scripts/primeintellect/run_full.sh
```

That's equivalent to:

```bash
python scripts/poker_train.py \
  --phase full \
  --model Qwen/Qwen2.5-Coder-1.5B-Instruct \
  --seed 42 \
  --episodes 1000 --bc-source agent --bc-task-mix mixed \
  --bc-epochs 3 --bc-grad-accum 4 --bc-weight-decay 0.01 \
  --rl-iterations 30 --rl-batch-size 12 \
  --rl-sample-temperature 0.2 --rl-top-p 0.9 \
  --rl-adv-clip 2.0 --ema-gamma 0.9 \
  --eval-episodes 100 --eval-by-street \
  --eval-json experiments/results/final_eval.json \
  2>&1 | tee checkpoints/poker_full_$(date +%Y%m%d_%H%M%S).log
```

Individual phases (useful if BC is done and you only need to re-run RL):

```bash
bash scripts/primeintellect/run_full.sh bc     # BC only
bash scripts/primeintellect/run_full.sh rl     # RL from ./checkpoints/poker_bc
bash scripts/primeintellect/run_full.sh eval   # eval of ./checkpoints/poker_rl
```

## 7. Pull results back to your laptop

From the laptop (not the pod):

```bash
PI_POD=<pod-id>     # from `prime pods list`
# Structured metrics for report tables:
scp root@$PI_POD:/root/STAT-4830-RL-project/experiments/results/final_eval.json \
    experiments/results/final_eval.json

# Training curves and logs:
scp root@$PI_POD:/root/STAT-4830-RL-project/checkpoints/poker_rl/training_curves.png \
    figures/poker_rl_training_curves.png
scp root@$PI_POD:/root/STAT-4830-RL-project/checkpoints/poker_full_*.log \
    docs/results/

# LoRA adapter weights (small, ~20MB each) for the Colab demo:
scp -r root@$PI_POD:/root/STAT-4830-RL-project/checkpoints/poker_bc ./checkpoints/poker_bc
scp -r root@$PI_POD:/root/STAT-4830-RL-project/checkpoints/poker_rl ./checkpoints/poker_rl
```

Or do it in one shot from the pod itself by pushing to a new branch:

```bash
cd /root/STAT-4830-RL-project
git checkout -b results/final_$(date +%Y%m%d)
git add experiments/results/final_eval.json docs/results/poker_full_*.log figures/*.png
git -c user.email=$GIT_EMAIL -c user.name="$GIT_USER" commit -m "final training results"
git push -u origin HEAD
```

## 8. Terminate the pod

```bash
# From the laptop
prime pods delete <pod-id>
# If you attached a disk and want to keep weights around:
prime disks list
# prime disks delete <disk-id>   # only when you're truly done
```

## Troubleshooting

| Symptom | Fix |
|---|---|
| `CUDA OOM` during BC | Drop `--batch-size` to 2 and raise `--bc-grad-accum` to 8. |
| `CUDA OOM` during RL | Drop `--rl-batch-size` to 6 and/or `--max-new-tokens` to 768. |
| Qwen download 401 | `HF_TOKEN` missing from pod env. `export HF_TOKEN=...` and re-run. |
| Tests fail on `bitsandbytes` import | Base Docker image lacks CUDA libs; recreate pod from a CUDA-enabled image (default PrimeIntellect PyTorch images work). |
| `unsloth` ImportError when `--unsloth` set | `pip install unsloth` in the venv, then re-run. Skip the flag to stay on the standard transformers+peft path. |
| RL loss wildly negative | Healthy — clipped-advantage REINFORCE produces signed losses. Watch `reward_ema` and `accuracy_ema`, not raw loss. |

## What gets produced

- `checkpoints/poker_bc/` — LoRA adapters after BC
- `checkpoints/poker_rl/iter_{5,10,15,20,…}/` — RL checkpoints every 5 iters
- `checkpoints/poker_rl/training_curves.png` — reward + accuracy EMA plot
- `experiments/results/final_eval.json` — per-suite (All/Preflop/Postflop), per-agent metrics: exact_match_rate, type_match_rate, avg_reward, avg_steps, confusion_matrix, by_action, by_street
- `docs/results/poker_full_*.log` — stdout tee for reproducibility
