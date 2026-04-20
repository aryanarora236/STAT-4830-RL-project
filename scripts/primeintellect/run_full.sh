#!/usr/bin/env bash
# Drive the poker BC → REINFORCE → Eval pipeline on a PrimeIntellect pod.
#
# Usage:
#   bash scripts/primeintellect/run_full.sh            # BC + RL + Eval (default)
#   bash scripts/primeintellect/run_full.sh bc         # BC only
#   bash scripts/primeintellect/run_full.sh rl         # RL only (from ./checkpoints/poker_bc)
#   bash scripts/primeintellect/run_full.sh eval       # Eval only (./checkpoints/poker_rl)
#
# Hyperparameters are set for the final presentation run. Override via env vars:
#   MODEL=Qwen/Qwen2.5-Coder-1.5B-Instruct
#   BC_EPISODES=1000
#   RL_ITERS=30
#   EVAL_EPISODES=100
#   SEED=42

set -euo pipefail

PHASE="${1:-full}"
MODEL="${MODEL:-Qwen/Qwen2.5-Coder-1.5B-Instruct}"
SEED="${SEED:-42}"

# Defaults are tuned so `bash run_full.sh` finishes in ~90 min on an H100 80GB.
# Scale BC_EPISODES / RL_ITERS / EVAL_EPISODES up when you have more time.
BC_EPISODES="${BC_EPISODES:-500}"
BC_EPOCHS="${BC_EPOCHS:-2}"
BC_GRAD_ACCUM="${BC_GRAD_ACCUM:-4}"
BC_WEIGHT_DECAY="${BC_WEIGHT_DECAY:-0.01}"
BC_OUTPUT="${BC_OUTPUT:-./checkpoints/poker_bc}"
BC_UNSLOTH_FLAG=""
[[ "${USE_UNSLOTH:-1}" == "1" ]] && BC_UNSLOTH_FLAG="--unsloth"

RL_ITERS="${RL_ITERS:-20}"
RL_BATCH="${RL_BATCH:-8}"
RL_LR="${RL_LR:-5e-6}"
RL_TEMP="${RL_TEMP:-0.2}"
RL_TOP_P="${RL_TOP_P:-0.9}"
RL_ADV_CLIP="${RL_ADV_CLIP:-2.0}"
RL_TASK_MODE="${RL_TASK_MODE:-all}"     # all | preflop | postflop
RL_OUTPUT="${RL_OUTPUT:-./checkpoints/poker_rl}"
EMA_GAMMA="${EMA_GAMMA:-0.9}"

EVAL_EPISODES="${EVAL_EPISODES:-50}"
EVAL_MODEL="${EVAL_MODEL:-$RL_OUTPUT}"
EVAL_JSON="${EVAL_JSON:-experiments/results/final_eval_$(date +%Y%m%d_%H%M%S).json}"

TS="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="docs/results/poker_${PHASE}_${TS}.log"
mkdir -p "$(dirname "$LOG_FILE")" experiments/results checkpoints figures

echo "=== run_full.sh phase=$PHASE model=$MODEL seed=$SEED ==="
echo "Log: $LOG_FILE"
echo "Started: $(date -u +%FT%TZ)"

case "$PHASE" in
  bc)
    python scripts/poker_train.py \
      --phase bc \
      --model "$MODEL" \
      --seed "$SEED" \
      --episodes "$BC_EPISODES" \
      --bc-source agent --bc-task-mix mixed \
      --bc-epochs "$BC_EPOCHS" \
      --bc-grad-accum "$BC_GRAD_ACCUM" \
      --bc-weight-decay "$BC_WEIGHT_DECAY" \
      --bc-output "$BC_OUTPUT" \
      $BC_UNSLOTH_FLAG \
      2>&1 | tee "$LOG_FILE"
    ;;

  rl)
    python scripts/poker_train.py \
      --phase rl \
      --model "$BC_OUTPUT" \
      --seed "$SEED" \
      --rl-iterations "$RL_ITERS" \
      --rl-batch-size "$RL_BATCH" \
      --rl-lr "$RL_LR" \
      --rl-sample-temperature "$RL_TEMP" \
      --rl-top-p "$RL_TOP_P" \
      --rl-adv-clip "$RL_ADV_CLIP" \
      --rl-task-mode "$RL_TASK_MODE" \
      --ema-gamma "$EMA_GAMMA" \
      --rl-output "$RL_OUTPUT" \
      2>&1 | tee "$LOG_FILE"
    ;;

  eval)
    python scripts/poker_train.py \
      --phase eval \
      --model "$EVAL_MODEL" \
      --seed "$SEED" \
      --eval-episodes "$EVAL_EPISODES" \
      --eval-by-street \
      --eval-json "$EVAL_JSON" \
      2>&1 | tee "$LOG_FILE"
    ;;

  full)
    python scripts/poker_train.py \
      --phase full \
      --model "$MODEL" \
      --seed "$SEED" \
      --episodes "$BC_EPISODES" \
      --bc-source agent --bc-task-mix mixed \
      --bc-epochs "$BC_EPOCHS" \
      --bc-grad-accum "$BC_GRAD_ACCUM" \
      --bc-weight-decay "$BC_WEIGHT_DECAY" \
      --bc-output "$BC_OUTPUT" \
      $BC_UNSLOTH_FLAG \
      --rl-iterations "$RL_ITERS" \
      --rl-batch-size "$RL_BATCH" \
      --rl-lr "$RL_LR" \
      --rl-sample-temperature "$RL_TEMP" \
      --rl-top-p "$RL_TOP_P" \
      --rl-adv-clip "$RL_ADV_CLIP" \
      --rl-task-mode "$RL_TASK_MODE" \
      --ema-gamma "$EMA_GAMMA" \
      --rl-output "$RL_OUTPUT" \
      --eval-episodes "$EVAL_EPISODES" \
      --eval-by-street \
      --eval-json "$EVAL_JSON" \
      2>&1 | tee "$LOG_FILE"
    ;;

  *)
    echo "Unknown phase: $PHASE (expected: full | bc | rl | eval)" >&2
    exit 1
    ;;
esac

echo "Finished: $(date -u +%FT%TZ)"
echo "Artifacts:"
echo "  log   : $LOG_FILE"
[[ "$PHASE" == "bc"   || "$PHASE" == "full" ]] && echo "  BC    : $BC_OUTPUT"
[[ "$PHASE" == "rl"   || "$PHASE" == "full" ]] && echo "  RL    : $RL_OUTPUT  (+ iter_* subdirs, training_curves.png)"
[[ "$PHASE" == "eval" || "$PHASE" == "full" ]] && echo "  eval  : $EVAL_JSON"
