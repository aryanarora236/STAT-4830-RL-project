"""
Modal app: run the shaped-reward poker REINFORCE experiment on an A10G.

Usage from repo root:
    modal run scripts/modal_shaped_reward.py

Or with custom shaping coefficients (for iteration):
    modal run scripts/modal_shaped_reward.py --tool-bonus 0.6 --fallback-penalty 0.4 --iterations 30

Produces experiments/results/modal_shaped_leaderboard.json locally after the
remote run finishes.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import modal

REPO_ROOT = Path(__file__).resolve().parent.parent

app = modal.App("stat4830-shaped-reward")

# Full-precision (bf16) path — avoids bitsandbytes complications on Modal.
# Qwen-1.5B in bf16 = ~3GB; with LoRA + backprop on A10G 24GB we have plenty.
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install(
        "torch==2.5.1",
        "transformers>=4.45",
        "peft>=0.13",
        "trl>=0.11",
        "datasets>=2.19",
        "accelerate>=0.30",
        "matplotlib",
        "numpy",
    )
    .add_local_dir(str(REPO_ROOT), "/workspace", ignore=["**/.venv/**", "**/__pycache__/**", "**/.git/**", "**/checkpoints/**"])
)


SHAPED_REWARD_PATCHES = [
    (
        "        for _ in range(self.batch_size):\n            attempted += 1\n            context, question, correct_answer = self.task_generator()",
        "        for _ in range(self.batch_size):\n            attempted += 1\n            was_wrapped = False  # SHAPED REWARD\n            context, question, correct_answer = self.task_generator()",
    ),
    (
        "                    if attempt_idx == 2:\n                        wrapped_action_code_count += 1\n                        raw_type, raw_amt = parse_action(response_text)",
        "                    if attempt_idx == 2:\n                        wrapped_action_code_count += 1\n                        was_wrapped = True  # SHAPED REWARD\n                        raw_type, raw_amt = parse_action(response_text)",
    ),
    # Note: actual shaping coefficients are injected as format-string placeholders
    # so a single Modal script can try multiple shaping configs.
    (
        "            reward = compute_poker_reward_simple(predicted, correct_answer)\n            if reward > 0:\n                nonzero_reward_count += 1",
        """            base_reward = compute_poker_reward_simple(predicted, correct_answer)
            # SHAPED REWARD: reward real code + stat parsing, penalize fallback
            tool_bonus = 0.0
            if not was_wrapped:
                tool_bonus += {tool_bonus_real_code}
            if parsed_stats:
                tool_bonus += {tool_bonus_parsed_stats}
            fallback_penalty = {fallback_penalty} if was_wrapped else 0.0
            reward = base_reward + tool_bonus - fallback_penalty
            if reward > 0:
                nonzero_reward_count += 1""",
    ),
]


VERBOSE_ROLLOUT_PATCH = (
    """            pred_type, pred_amt = parse_action(predicted)
            trajectories.append(PokerTrajectory(""",
    """            # VERBOSE ROLLOUT LOG
            print(f"  >> rollout {len(trajectories)+1}/{self.batch_size}  was_wrapped={was_wrapped}  parsed_stats={parsed_stats}  exec_ok={bool(exec_result.ok) if 'exec_result' in dir() else '?'}", flush=True)
            print(f"     correct={correct_answer!r}  predicted={predicted!r}  base_r={base_reward:.2f}  shaped_r={reward:.2f}", flush=True)
            _resp_preview = (response_text or '')[:240].replace(chr(10), ' / ')
            print(f"     response[:240]={_resp_preview!r}", flush=True)
            _code_preview = (code or '')[:240].replace(chr(10), ' / ')
            print(f"     code[:240]={_code_preview!r}", flush=True)
            pred_type, pred_amt = parse_action(predicted)
            trajectories.append(PokerTrajectory(""",
)


def apply_patch(path: str, tool_bonus_real_code: float, tool_bonus_parsed_stats: float, fallback_penalty: float, verbose_rollouts: bool = False) -> None:
    """Apply the 3-point shaped reward patch with configurable coefficients."""
    with open(path, "r") as f:
        src = f.read()

    if "SHAPED REWARD" in src:
        print("[patch] already present, skipping")
    else:
        for i, (old, new) in enumerate(SHAPED_REWARD_PATCHES, 1):
            if "{tool_bonus_real_code}" in new:
                new = new.format(
                    tool_bonus_real_code=tool_bonus_real_code,
                    tool_bonus_parsed_stats=tool_bonus_parsed_stats,
                    fallback_penalty=fallback_penalty,
                )
            if old not in src:
                raise RuntimeError(f"Patch {i} insertion point not found in {path}")
            src = src.replace(old, new, 1)
        with open(path, "w") as f:
            f.write(src)
        print(
            f"[patch] applied with tool_bonus=+{tool_bonus_real_code} (code), "
            f"+{tool_bonus_parsed_stats} (stats), fallback_penalty=-{fallback_penalty}"
        )

    if verbose_rollouts:
        with open(path, "r") as f:
            src = f.read()
        if "VERBOSE ROLLOUT LOG" in src:
            print("[verbose-patch] already present, skipping")
            return
        old, new = VERBOSE_ROLLOUT_PATCH
        if old not in src:
            raise RuntimeError("verbose rollout patch insertion point not found")
        src = src.replace(old, new, 1)
        with open(path, "w") as f:
            f.write(src)
        print("[verbose-patch] applied — each rollout will log response/code/predicted")


@app.function(image=image, gpu="A10G", timeout=4500)
def train_and_eval(
    iterations: int = 30,
    batch_size: int = 4,
    tool_bonus_real_code: float = 0.3,
    tool_bonus_parsed_stats: float = 0.2,
    fallback_penalty: float = 0.2,
    eval_episodes: int = 20,
    eval_seed: int = 20260422,
    checkpoint_every: int = 5,
    run_tag: str = "default",
    sample_temperature: float = 0.2,
    sample_top_p: float = 0.9,
    rl_lr: float = 5e-6,
    verbose_rollouts: bool = False,
    max_new_tokens: int = 512,
):
    """
    Full pipeline:
      1. Patch src/poker/training.py with shaped reward.
      2. Copy Exp B's best_by_eval as the starting checkpoint.
      3. Run --phase rl training via subprocess (full precision on A10G).
      4. Eval each saved checkpoint on held-out tasks.
      5. Return the leaderboard dict.
    """
    import os
    import shutil
    import subprocess
    import sys
    import time

    os.chdir("/workspace")
    sys.path.insert(0, "/workspace")
    os.environ["PYTHONUNBUFFERED"] = "1"

    # 1. Apply patch
    apply_patch(
        "src/poker/training.py",
        tool_bonus_real_code=tool_bonus_real_code,
        tool_bonus_parsed_stats=tool_bonus_parsed_stats,
        fallback_penalty=fallback_penalty,
        verbose_rollouts=verbose_rollouts,
    )

    # 2. Copy starting checkpoint (Exp B's best_by_eval)
    src_ckpt = "docs/results/poker_rl_expB_mixed20_20260422/poker_rl_expB_mixed20_20260422/best_by_eval"
    dst_ckpt = "checkpoints/poker_rl_expB_best"
    os.makedirs("checkpoints", exist_ok=True)
    if os.path.exists(dst_ckpt):
        shutil.rmtree(dst_ckpt)
    shutil.copytree(src_ckpt, dst_ckpt)
    print(f"[setup] starting checkpoint copied to {dst_ckpt}")

    # 3. Launch training via the CLI
    rl_output = f"checkpoints/poker_rl_shaped_{run_tag}"
    cmd = [
        "python", "-u", "scripts/poker_train.py",
        "--phase", "rl",
        "--rl-model", dst_ckpt,       # --phase rl reads --rl-model (not --model)
        "--bc-output", dst_ckpt,       # belt-and-suspenders in case of fallback
        "--seed", "20260422",
        "--rl-iterations", str(iterations),
        "--rl-batch-size", str(batch_size),
        "--rl-lr", str(rl_lr),
        "--rl-sample-temperature", str(sample_temperature),
        "--rl-top-p", str(sample_top_p),
        "--rl-adv-clip", "2.0",
        "--ema-gamma", "0.9",
        "--max-new-tokens", str(max_new_tokens),
        "--full-precision",  # bf16 on A10G, skip bitsandbytes
        "--rl-output", rl_output,
    ]
    print(f"[train] launching: {' '.join(cmd)}")
    t0 = time.time()
    result = subprocess.run(cmd, capture_output=False, text=True)
    t_train = time.time() - t0
    print(f"[train] finished in {t_train/60:.1f} min, exit code {result.returncode}")
    if result.returncode != 0:
        raise RuntimeError(f"training failed with exit code {result.returncode}")

    # 4. Held-out eval on each saved checkpoint
    import random
    import torch

    from src.training import load_model
    from src.poker.agents import PokerLocalLLMAgent
    from src.poker.tasks import generate_poker_task
    from src.poker.rewards import parse_action

    def held_out_eval(model, tokenizer, n_episodes: int, seed: int) -> float:
        random.seed(seed)
        torch.manual_seed(seed)
        agent = PokerLocalLLMAgent(
            model=model,
            tokenizer=tokenizer,
            name="eval",
            max_steps=1,
            max_new_tokens=384,
            temperature=0.1,
        )
        correct = 0
        for i in range(n_episodes):
            context, question, answer = generate_poker_task()
            pred, _ = agent.run_episode(context, question, answer)
            if parse_action(pred)[0] == parse_action(answer)[0]:
                correct += 1
            if (i + 1) % 5 == 0:
                print(f"    [eval] {i+1}/{n_episodes} running_acc={correct/(i+1):.2f}")
        return correct / n_episodes

    results = [
        {
            "iteration": 0,
            "held_out_acc": 0.20,
            "source": "baseline (Exp B leaderboard)",
            "run_tag": run_tag,
        }
    ]

    import glob
    ckpt_dirs = sorted(
        glob.glob(f"{rl_output}/iter_*"),
        key=lambda p: int(p.rsplit("_", 1)[-1]),
    )
    print(f"[eval] found {len(ckpt_dirs)} checkpoints")

    for ckpt in ckpt_dirs:
        it = int(ckpt.rsplit("_", 1)[-1])
        print(f"\n--- Evaluating iter_{it} ({eval_episodes} eps) ---")
        t_eval_start = time.time()
        model, tokenizer = load_model(ckpt, load_in_4bit=False)
        model.eval()
        acc = held_out_eval(model, tokenizer, eval_episodes, eval_seed)
        t_eval = time.time() - t_eval_start
        print(f"[eval] iter {it}: held_out_acc={acc:.3f}  ({t_eval:.0f}s)")
        results.append({
            "iteration": it,
            "held_out_acc": acc,
            "source": ckpt,
            "run_tag": run_tag,
        })
        del model, tokenizer
        torch.cuda.empty_cache()

    # 5. Return + also write JSON in container
    out_path = f"experiments/results/modal_shaped_leaderboard_{run_tag}.json"
    os.makedirs("experiments/results", exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    # Print final table
    print("\n" + "=" * 66)
    print(f"SHAPED-REWARD LEADERBOARD  (run_tag={run_tag})")
    print(f"  shaping:  real_code=+{tool_bonus_real_code}  stats=+{tool_bonus_parsed_stats}  fallback=-{fallback_penalty}")
    print(f"  training: {iterations} iters, batch {batch_size}, eval {eval_episodes} eps/checkpoint")
    print("=" * 66)
    print(f"{'iter':>5} | {'held-out acc':>12} | {'delta vs baseline':>18}")
    for row in results:
        delta = (row["held_out_acc"] - 0.20) * 100
        sign = "+" if delta >= 0 else ""
        print(f"{row['iteration']:>5} | {row['held_out_acc']:>12.1%} | {sign}{delta:>17.1f} pp")
    print("=" * 66)

    return results


@app.local_entrypoint()
def main(
    iterations: int = 30,
    batch_size: int = 4,
    tool_bonus: float = 0.3,
    stats_bonus: float = 0.2,
    fallback_penalty: float = 0.2,
    eval_episodes: int = 20,
    run_tag: str = "default",
    sample_temperature: float = 0.2,
    sample_top_p: float = 0.9,
    rl_lr: float = 5e-6,
    verbose_rollouts: bool = False,
    max_new_tokens: int = 512,
):
    print(f"[local] dispatching to Modal: run_tag={run_tag}  temp={sample_temperature}  lr={rl_lr}  max_new_tokens={max_new_tokens}  verbose={verbose_rollouts}")
    results = train_and_eval.remote(
        iterations=iterations,
        batch_size=batch_size,
        tool_bonus_real_code=tool_bonus,
        tool_bonus_parsed_stats=stats_bonus,
        fallback_penalty=fallback_penalty,
        eval_episodes=eval_episodes,
        run_tag=run_tag,
        sample_temperature=sample_temperature,
        sample_top_p=sample_top_p,
        rl_lr=rl_lr,
        verbose_rollouts=verbose_rollouts,
        max_new_tokens=max_new_tokens,
    )

    out_local = Path(f"experiments/results/modal_shaped_leaderboard_{run_tag}.json")
    out_local.parent.mkdir(parents=True, exist_ok=True)
    with open(out_local, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[local] wrote {out_local}")
    print("\n[local] final leaderboard:")
    for row in results:
        delta = (row["held_out_acc"] - 0.20) * 100
        sign = "+" if delta >= 0 else ""
        print(f"  iter {row['iteration']:>3}  held_out_acc={row['held_out_acc']:>6.1%}  delta={sign}{delta:.1f} pp")
