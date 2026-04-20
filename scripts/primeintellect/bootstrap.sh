#!/usr/bin/env bash
# Bootstrap a fresh PrimeIntellect pod for STAT 4830 poker RLM training.
#
# Assumes:
#   - Ubuntu 22.04+ base image with CUDA drivers (PrimeIntellect default PyTorch pods qualify)
#   - HF_TOKEN env var set (passed via `prime pods create --env HF_TOKEN=...`)
#
# Idempotent: re-running is safe after SSH reconnection.

set -euo pipefail

REPO_URL="${REPO_URL:-https://github.com/aryanarora236/STAT-4830-RL-project.git}"
REPO_DIR="${REPO_DIR:-/root/STAT-4830-RL-project}"
VENV_DIR="${VENV_DIR:-$REPO_DIR/.venv}"

echo "[bootstrap] updating apt index"
apt-get update -qq
apt-get install -y --no-install-recommends git python3-venv python3-dev build-essential >/dev/null

if [[ ! -d "$REPO_DIR/.git" ]]; then
  echo "[bootstrap] cloning $REPO_URL → $REPO_DIR"
  git clone --depth 1 "$REPO_URL" "$REPO_DIR"
else
  echo "[bootstrap] repo present, pulling latest"
  git -C "$REPO_DIR" pull --ff-only
fi

cd "$REPO_DIR"

if [[ ! -d "$VENV_DIR" ]]; then
  echo "[bootstrap] creating venv"
  python3 -m venv "$VENV_DIR"
fi

# shellcheck disable=SC1090
source "$VENV_DIR/bin/activate"

echo "[bootstrap] installing deps (this takes 3-5 min)"
pip install --upgrade pip wheel >/dev/null
pip install -r requirements.txt
pip install matplotlib pytest >/dev/null

# Optional fast path
if [[ "${INSTALL_UNSLOTH:-0}" == "1" ]]; then
  echo "[bootstrap] installing unsloth (optional fast path)"
  pip install unsloth
fi

echo "[bootstrap] sanity check"
python - <<'PY'
import torch
assert torch.cuda.is_available(), "no CUDA device visible"
dev = torch.cuda.get_device_name(0)
cap = torch.cuda.get_device_capability(0)
mem_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
print(f"GPU: {dev} | compute {cap[0]}.{cap[1]} | {mem_gb:.1f} GB")
print(f"bf16: {cap[0] >= 8}, fp16 fallback: {cap[0] < 8}")
PY

echo "[bootstrap] running unit tests"
python -m pytest tests/ -q --tb=line

if [[ -z "${HF_TOKEN:-}" ]]; then
  echo "[bootstrap] WARNING: HF_TOKEN not set — Qwen download may rate-limit."
else
  echo "[bootstrap] HF_TOKEN present"
  python -c "from huggingface_hub import login; import os; login(token=os.environ['HF_TOKEN'])"
fi

mkdir -p checkpoints experiments/results docs/results figures

echo
echo "[bootstrap] done. Activate with:"
echo "  source $VENV_DIR/bin/activate"
echo "Then run:"
echo "  bash scripts/primeintellect/run_full.sh"
