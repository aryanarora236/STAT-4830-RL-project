"""
Apply the shaped-reward patch to src/poker/training.py, run a short RL
session starting from Exp A's best_by_eval, then play live vs Slumbot.

The training patch is the same one used in `scripts/modal_shaped_reward.py`:
  +tool_bonus_real_code  for genuine code (not wrapped fallback)
  +tool_bonus_parsed_stats for parsing opponent stats
  -fallback_penalty for the wrapped-action fallback

Restores training.py after training completes.
"""

import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent.parent
TRAINING_FILE = ROOT / "src/poker/training.py"
BACKUP_FILE   = ROOT / "src/poker/training.py.preshape.bak"
START_CKPT    = "docs/results/poker_rl_expA_partial_20260422/poker_rl_expA_evalselect_20260421_long/best_by_eval"
NEW_CKPT_DIR  = "checkpoints/shaped_rl_local"

# These mirror SHAPED_REWARD_PATCHES in scripts/modal_shaped_reward.py
SHAPING_PATCHES = [
    (
        "        for _ in range(self.batch_size):\n            attempted += 1\n            context, question, correct_answer = self.task_generator()",
        "        for _ in range(self.batch_size):\n            attempted += 1\n            was_wrapped = False  # SHAPED REWARD\n            context, question, correct_answer = self.task_generator()",
    ),
    (
        "                    if attempt_idx == 2:\n                        wrapped_action_code_count += 1\n                        raw_type, raw_amt = parse_action(response_text)",
        "                    if attempt_idx == 2:\n                        wrapped_action_code_count += 1\n                        was_wrapped = True  # SHAPED REWARD\n                        raw_type, raw_amt = parse_action(response_text)",
    ),
]


def shaping_reward_patch(tool_bonus_code, tool_bonus_stats, fallback_pen):
    return (
        "            reward = compute_poker_reward_simple(predicted, correct_action)\n            if reward > 0:\n                nonzero_reward_count += 1",
        (
            "            base_reward = compute_poker_reward_simple(predicted, correct_action)\n"
            "            # SHAPED REWARD: reward real code + stat parsing, penalize fallback\n"
            "            tool_bonus = 0.0\n"
            "            if not was_wrapped:\n"
            f"                tool_bonus += {tool_bonus_code}\n"
            "            if parsed_stats:\n"
            f"                tool_bonus += {tool_bonus_stats}\n"
            f"            fallback_penalty = {fallback_pen} if was_wrapped else 0.0\n"
            "            reward = base_reward + tool_bonus - fallback_penalty\n"
            "            if reward > 0:\n"
            "                nonzero_reward_count += 1"
        ),
    )


def apply_patches(tool_bonus_code, tool_bonus_stats, fallback_pen):
    src = TRAINING_FILE.read_text()
    if "SHAPED REWARD" in src:
        print("[patch] already applied — skipping")
        return False

    shutil.copyfile(TRAINING_FILE, BACKUP_FILE)

    patches = list(SHAPING_PATCHES) + [shaping_reward_patch(tool_bonus_code, tool_bonus_stats, fallback_pen)]
    for old, new in patches:
        if old not in src:
            raise RuntimeError(f"Patch insertion point not found:\n{old[:120]}")
        src = src.replace(old, new, 1)
    TRAINING_FILE.write_text(src)
    print(f"[patch] applied (+{tool_bonus_code} real_code, +{tool_bonus_stats} stats, -{fallback_pen} fallback)")
    return True


def restore_training():
    if BACKUP_FILE.exists():
        shutil.copyfile(BACKUP_FILE, TRAINING_FILE)
        BACKUP_FILE.unlink()
        print("[patch] restored original training.py")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--user", required=True)
    parser.add_argument("--password", required=True)
    parser.add_argument("--rl-iterations", type=int, default=15)
    parser.add_argument("--hands", type=int, default=10)
    parser.add_argument("--tool-bonus-code",  type=float, default=0.3)
    parser.add_argument("--tool-bonus-stats", type=float, default=0.2)
    parser.add_argument("--fallback-penalty", type=float, default=0.2)
    args = parser.parse_args()

    os.chdir(ROOT)
    print(f"[run] starting in {ROOT}")
    print(f"[run] starting checkpoint: {START_CKPT}")
    print(f"[run] new checkpoint dir:  {NEW_CKPT_DIR}")
    print(f"[run] RL iters: {args.rl_iterations} | hands: {args.hands}\n")

    try:
        applied = apply_patches(args.tool_bonus_code, args.tool_bonus_stats, args.fallback_penalty)

        # ── Train ───────────────────────────────────────────────────────────────
        train_cmd = [
            sys.executable, "scripts/poker_train.py",
            "--phase", "rl",
            "--model", START_CKPT,
            "--rl-iterations", str(args.rl_iterations),
            "--rl-output", NEW_CKPT_DIR,
            "--rl-batch-size", "4",
        ]
        print("[run] training command:", " ".join(train_cmd))
        t0 = time.time()
        result = subprocess.run(train_cmd, env={**os.environ, "PYTHONUNBUFFERED": "1"})
        print(f"[run] training finished in {time.time()-t0:.0f}s (exit={result.returncode})")
        if result.returncode != 0:
            print("[run] training failed — aborting")
            return

    finally:
        if applied:
            restore_training()

    # Pick the latest iter checkpoint
    ckpts = sorted(Path(NEW_CKPT_DIR).glob("iter_*"), key=lambda p: int(p.name.split("_")[1]))
    target = ckpts[-1] if ckpts else Path(NEW_CKPT_DIR)
    print(f"\n[run] testing checkpoint against Slumbot: {target}")

    out_dir = "docs/results/slumbot_rl_shaped_session_20260513"
    slumbot_cmd = [
        sys.executable, "scripts/run_slumbot_rl_session.py",
        "--user", args.user,
        "--password", args.password,
        "--hands", str(args.hands),
        "--ckpt", str(target),
        "--out-dir", out_dir,
    ]
    print("[run] slumbot command:", " ".join(slumbot_cmd))
    subprocess.run(slumbot_cmd)
    print(f"\n[run] done — results in {out_dir}/")


if __name__ == "__main__":
    main()
