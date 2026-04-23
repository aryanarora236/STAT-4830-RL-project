"""
Modal inspection: load a LoRA checkpoint and dump full rollout transcripts.

Used to capture real samples of what the agent writes (response text,
extracted code, sandbox stdout, parsed action, whether correct) for
presentation slides and debugging.

Usage from repo root:
    modal run --detach scripts/modal_inspect_rollouts.py

Default: loads Exp B's best_by_eval (the BEFORE-shaped-reward checkpoint that
both v1 and v2 started from). 8 rollouts on fresh poker tasks, full logging.
"""

from __future__ import annotations

from pathlib import Path
import modal

REPO_ROOT = Path(__file__).resolve().parent.parent

app = modal.App("stat4830-rollout-inspect")

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
    )
    .add_local_dir(
        str(REPO_ROOT),
        "/workspace",
        ignore=["**/.venv/**", "**/__pycache__/**", "**/.git/**", "**/checkpoints/**"],
    )
)


@app.function(image=image, gpu="A10G", timeout=1800)
def inspect_rollouts(
    checkpoint_relpath: str = "docs/results/poker_rl_expB_mixed20_20260422/poker_rl_expB_mixed20_20260422/best_by_eval",
    n_rollouts: int = 8,
    seed: int = 20260423,
    max_new_tokens: int = 512,
    temperature: float = 0.2,
    max_steps: int = 3,
):
    import os
    import random
    import sys

    os.chdir("/workspace")
    sys.path.insert(0, "/workspace")

    import torch
    from src.training import load_model
    from src.poker.tasks import generate_poker_task
    from src.poker.rewards import parse_action
    from src.poker.agents import PokerLocalLLMAgent

    random.seed(seed)
    torch.manual_seed(seed)

    print(f"[inspect] loading {checkpoint_relpath}")
    model, tokenizer = load_model(checkpoint_relpath, load_in_4bit=False)
    model.eval()

    agent = PokerLocalLLMAgent(
        model=model,
        tokenizer=tokenizer,
        name="inspect",
        max_steps=max_steps,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
    )

    rollouts = []
    correct = 0

    for i in range(n_rollouts):
        context, question, answer = generate_poker_task()
        pred, transcript = agent.run_episode(context, question, answer)

        corr_type = parse_action(answer)[0]
        pred_type = parse_action(pred)[0]
        match = pred_type == corr_type

        if match:
            correct += 1

        print("\n" + "=" * 80)
        print(f"ROLLOUT {i+1}/{n_rollouts}  seed_offset={i}")
        print("=" * 80)
        print(f"[ground truth answer]: {answer!r}")
        print(f"[parsed predicted]:    {pred!r}")
        print(f"[action type match]:   {match}")
        print(f"[num steps taken]:     {len(transcript)}")
        print()

        # Show the context hero has seen (truncated for log clarity)
        print("--- CONTEXT (first 600 chars of", len(context), "total) ---")
        print(context[:600])
        if len(context) > 600:
            print(f"... ({len(context) - 600} more chars)")
        print()

        for step_idx, step in enumerate(transcript):
            print(f"--- STEP {step_idx+1} ---")
            step_type = step.get("step") or step.get("type") or "?"
            print(f"step marker: {step_type}")

            resp = step.get("response_text") or step.get("response") or ""
            if resp:
                print(">> MODEL RESPONSE (first 800 chars of", len(resp), "total):")
                print(resp[:800])
                if len(resp) > 800:
                    print(f"... ({len(resp) - 800} more chars)")

            code = step.get("code") or ""
            if code:
                print(">> EXTRACTED CODE (first 800 chars of", len(code), "total):")
                print(code[:800])
                if len(code) > 800:
                    print(f"... ({len(code) - 800} more chars)")

            exec_result = step.get("exec_result") or {}
            if exec_result:
                print(">> SANDBOX EXEC:")
                print("  ok:", exec_result.get("ok"))
                stdout = (exec_result.get("stdout") or "")[:400]
                stderr = (exec_result.get("stderr") or "")[:400]
                print("  stdout:", repr(stdout))
                if stderr:
                    print("  stderr:", repr(stderr))
            print()

        rollouts.append({
            "rollout_idx": i,
            "predicted": pred,
            "correct_answer": answer,
            "match": match,
            "num_steps": len(transcript),
            "context_preview": context[:400],
            "steps": [
                {
                    "type": s.get("step") or s.get("type"),
                    "response_text": (s.get("response_text") or s.get("response") or "")[:2000],
                    "code": (s.get("code") or "")[:2000],
                    "exec_ok": (s.get("exec_result") or {}).get("ok"),
                    "stdout": ((s.get("exec_result") or {}).get("stdout") or "")[:400],
                    "stderr": ((s.get("exec_result") or {}).get("stderr") or "")[:400],
                }
                for s in transcript
            ],
        })

    print("\n" + "=" * 80)
    print(f"SUMMARY: {correct}/{n_rollouts} correct action-type match = {correct/n_rollouts:.1%}")
    print(f"Checkpoint: {checkpoint_relpath}")
    print("=" * 80)

    return {
        "checkpoint": checkpoint_relpath,
        "n_rollouts": n_rollouts,
        "correct": correct,
        "accuracy": correct / n_rollouts,
        "rollouts": rollouts,
    }


@app.local_entrypoint()
def main(
    checkpoint: str = "docs/results/poker_rl_expB_mixed20_20260422/poker_rl_expB_mixed20_20260422/best_by_eval",
    n: int = 8,
    temperature: float = 0.2,
):
    print(f"[local] dispatching rollout inspection: checkpoint={checkpoint} n={n} temp={temperature}")
    result = inspect_rollouts.remote(
        checkpoint_relpath=checkpoint,
        n_rollouts=n,
        temperature=temperature,
    )
    import json
    out = Path("experiments/results/inspect_rollouts.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\n[local] wrote {out}")
    print(f"accuracy: {result['accuracy']:.1%}  ({result['correct']}/{result['n_rollouts']})")
