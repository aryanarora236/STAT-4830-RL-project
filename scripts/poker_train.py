"""
Poker training entry point.

Usage examples:
    # Phase 1: behavior cloning with heuristic reasoning traces
    python scripts/poker_train.py --phase bc --model Qwen/Qwen2.5-Coder-1.5B-Instruct --episodes 500

    # Phase 2: REINFORCE from BC checkpoint
    python scripts/poker_train.py --phase rl --model ./checkpoints/poker_bc --rl-iterations 20

    # Evaluate a trained checkpoint
    python scripts/poker_train.py --phase eval --model ./checkpoints/poker_rl --eval-episodes 20

    # Full pipeline: BC -> RL -> Eval
    python scripts/poker_train.py --phase full --model Qwen/Qwen2.5-Coder-1.5B-Instruct
"""

import argparse
import json
import os
import sys
from typing import Any, Dict, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.training import load_model, LORA_PRESETS, set_training_seed
from src.poker.tasks import (
    generate_poker_task,
    generate_preflop_task,
    generate_postflop_task,
    bc_agent_task_generators,
)
from src.poker.training import (
    collect_poker_trajectories,
    collect_poker_trajectories_with_traces,
    PokerBCTrainer,
    PokerReinforceTrainer,
)
from src.poker.agents import PokerHeuristicAgent, PokerLocalLLMAgent
from src.poker.evaluation import PokerEvaluationFramework


def run_bc(args):
    """Phase 1: Behavior cloning on poker trajectories."""
    print("=" * 60)
    print("POKER PHASE 1: BEHAVIOR CLONING")
    print("=" * 60)

    # 1) Collect trajectories
    if args.bc_source == "traces":
        print(f"\nCollecting {args.episodes} heuristic trace trajectories...")
        trajectories = collect_poker_trajectories_with_traces(num_episodes=args.episodes)
    else:
        task_gens = bc_agent_task_generators(args.bc_task_mix)
        print(
            f"\nCollecting {args.episodes} trajectories via PokerHeuristicAgent "
            f"(bc_task_mix={args.bc_task_mix}, {len(task_gens)} generator(s))..."
        )
        heuristic_agent = PokerHeuristicAgent()
        trajectories = collect_poker_trajectories(
            heuristic_agent,
            num_episodes=args.episodes,
            task_generators=task_gens,
        )

    successful = sum(1 for t in trajectories if t.is_correct and t.code)
    print(f"Collected: {len(trajectories)} total, {successful} successful")

    # 2) Load base model (fresh LoRA or checkpoint)
    print(f"\nLoading model: {args.model}")
    model, tokenizer = load_model(
        model_id_or_path=args.model,
        load_in_4bit=not args.full_precision,
        lora_r=args.lora_r,
        lora_preset=args.lora_targets,
        use_unsloth=args.unsloth,
    )

    # 3) Train BC
    trainer = PokerBCTrainer(
        model=model,
        tokenizer=tokenizer,
        output_dir=args.bc_output,
        num_epochs=args.bc_epochs,
        batch_size=args.batch_size,
        learning_rate=args.bc_lr,
        max_length=args.max_length,
        gradient_accumulation_steps=args.bc_grad_accum,
        weight_decay=args.bc_weight_decay,
        seed=args.seed if args.seed >= 0 else None,
    )
    result = trainer.train(trajectories)
    print(f"\nBC training complete: {result}")
    return model, tokenizer


def run_rl(args, model=None, tokenizer=None):
    """Phase 2: REINFORCE on poker tasks."""
    print("\n" + "=" * 60)
    print("POKER PHASE 2: REINFORCE")
    print("=" * 60)

    if model is None:
        load_path = args.rl_model or args.bc_output
        print(f"\nLoading model from: {load_path}")
        model, tokenizer = load_model(
            model_id_or_path=load_path,
            load_in_4bit=not args.full_precision,
            use_unsloth=args.unsloth,
        )

    if args.rl_task_mode == "preflop":
        rl_task_generator = generate_preflop_task
    elif args.rl_task_mode == "postflop":
        rl_task_generator = generate_postflop_task
    else:
        rl_task_generator = generate_poker_task

    trainer = PokerReinforceTrainer(
        model=model,
        tokenizer=tokenizer,
        output_dir=args.rl_output,
        batch_size=args.rl_batch_size,
        learning_rate=args.rl_lr,
        max_new_tokens=args.max_new_tokens,
        max_length=args.max_length,
        baseline_ema=args.rl_baseline_ema,
        advantage_clip=args.rl_adv_clip,
        task_generator=rl_task_generator,
        ema_gamma=args.ema_gamma,
        sample_temperature=args.rl_sample_temperature,
        sample_top_p=args.rl_top_p,
    )

    history = trainer.train(num_iterations=args.rl_iterations)
    trainer.plot_training()
    if history:
        first = history[0]
        last = history[-1]
        print("\nREINFORCE summary:")
        print(f"  Accuracy: {first['accuracy']:.1%} -> {last['accuracy']:.1%}")
        print(f"  Reward:   {first['avg_reward']:.3f} -> {last['avg_reward']:.3f}")
        print(f"  Loss:     {first['loss']:.4f} -> {last['loss']:.4f}")

    return model, tokenizer, history


def _eval_one(name, task_gen, agents, episodes) -> Tuple[str, PokerEvaluationFramework]:
    print(f"\n--- {name} ---")
    fw = PokerEvaluationFramework(
        agents=agents,
        task_generator=task_gen,
        num_episodes=episodes,
    )
    fw.run_evaluation()
    fw.display_results()
    return name, fw


def run_eval(args, model=None, tokenizer=None):
    """Evaluate local poker model on generated poker tasks."""
    print("\n" + "=" * 60)
    print("POKER EVALUATION")
    print("=" * 60)

    load_path = args.eval_model or args.rl_output
    if model is None:
        print(f"\nLoading model from: {load_path}")
        model, tokenizer = load_model(
            model_id_or_path=load_path,
            load_in_4bit=not args.full_precision,
        )

    local_agent = PokerLocalLLMAgent(
        model=model,
        tokenizer=tokenizer,
        max_steps=args.eval_max_steps,
        max_new_tokens=args.max_new_tokens,
        temperature=args.eval_temperature,
    )
    heuristic = PokerHeuristicAgent()
    agents = [local_agent] if args.no_heuristic_baseline else [local_agent, heuristic]

    suites: Dict[str, Any] = {}
    name, fw = _eval_one("All Streets", generate_poker_task, agents, args.eval_episodes)
    suites[name] = fw.export_run_summary()
    if args.eval_by_street:
        name, fw = _eval_one("Preflop", generate_preflop_task, agents, args.eval_episodes)
        suites[name] = fw.export_run_summary()
        name, fw = _eval_one("Postflop", generate_postflop_task, agents, args.eval_episodes)
        suites[name] = fw.export_run_summary()

    if args.eval_json:
        meta: Dict[str, Any] = {
            "checkpoint": load_path,
            "eval_episodes": args.eval_episodes,
            "eval_by_street": args.eval_by_street,
            "eval_temperature": args.eval_temperature,
            "seed": args.seed,
            "no_heuristic_baseline": args.no_heuristic_baseline,
        }
        out_path = args.eval_json
        parent = os.path.dirname(os.path.abspath(out_path))
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump({"meta": meta, "suites": suites}, f, indent=2)
        print(f"\nWrote evaluation report to {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Poker RLM Training Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--phase",
        choices=["bc", "rl", "full", "eval"],
        default="full",
        help="bc: behavior cloning, rl: REINFORCE, full: bc+rl+eval, eval: evaluate checkpoint",
    )
    parser.add_argument(
        "--model",
        default="Qwen/Qwen2.5-Coder-1.5B-Instruct",
        help="Base model ID or checkpoint path",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="RNG seed for task sampling and BC trainer (use -1 to skip set_training_seed)",
    )

    # Shared/model
    parser.add_argument("--full-precision", action="store_true", help="Disable 4-bit quantization")
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-targets", choices=list(LORA_PRESETS.keys()), default="attention",
                        help="LoRA target preset: attention, mlp, mlp+head, attention+mlp, all, head")
    parser.add_argument("--unsloth", action="store_true", help="Use unsloth for faster training (2-5x)")
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--max-new-tokens", type=int, default=1024)

    # BC
    parser.add_argument("--episodes", type=int, default=500, help="Trajectory count for BC")
    parser.add_argument("--bc-source", choices=["traces", "agent"], default="traces")
    parser.add_argument(
        "--bc-task-mix",
        choices=["all", "mixed", "preflop", "postflop"],
        default="all",
        help="When bc_source=agent: which task distribution to sample (mixed = street-balanced mix)",
    )
    parser.add_argument("--bc-epochs", type=int, default=3)
    parser.add_argument("--bc-lr", type=float, default=2e-4)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument(
        "--bc-grad-accum",
        type=int,
        default=2,
        help="Gradient accumulation for poker BC (increases effective batch size)",
    )
    parser.add_argument(
        "--bc-weight-decay",
        type=float,
        default=0.0,
        help="AdamW weight decay during BC; try 0.01 if overfitting small trace sets",
    )
    parser.add_argument("--bc-output", default="./checkpoints/poker_bc")

    # RL
    parser.add_argument("--rl-model", default=None, help="Checkpoint to start RL from (default: BC output)")
    parser.add_argument("--rl-iterations", type=int, default=20)
    parser.add_argument("--rl-batch-size", type=int, default=12)
    parser.add_argument("--rl-lr", type=float, default=5e-6)
    parser.add_argument("--rl-baseline-ema", type=float, default=0.95)
    parser.add_argument("--rl-adv-clip", type=float, default=2.0)
    parser.add_argument("--rl-task-mode", choices=["all", "preflop", "postflop"], default="all")
    parser.add_argument("--ema-gamma", type=float, default=0.9, help="EMA decay for reward/accuracy tracking")
    parser.add_argument(
        "--rl-sample-temperature",
        type=float,
        default=0.15,
        help="Rollout sampling temperature for poker REINFORCE (higher = more exploration)",
    )
    parser.add_argument(
        "--rl-top-p",
        type=float,
        default=0.9,
        help="Nucleus sampling top_p during poker REINFORCE rollouts",
    )
    parser.add_argument("--rl-output", default="./checkpoints/poker_rl")

    # Eval
    parser.add_argument("--eval-model", default=None, help="Checkpoint to evaluate (default: RL output)")
    parser.add_argument("--eval-episodes", type=int, default=20)
    parser.add_argument("--eval-max-steps", type=int, default=5)
    parser.add_argument("--eval-temperature", type=float, default=0.1)
    parser.add_argument("--eval-by-street", action="store_true")
    parser.add_argument(
        "--eval-json",
        default=None,
        metavar="PATH",
        help="Write structured metrics (per suite, per agent) to this JSON file",
    )
    parser.add_argument("--no-heuristic-baseline", action="store_true")

    args = parser.parse_args()

    if args.seed >= 0:
        set_training_seed(args.seed)

    if args.phase == "bc":
        run_bc(args)
    elif args.phase == "rl":
        run_rl(args)
    elif args.phase == "eval":
        run_eval(args)
    else:
        model, tokenizer = run_bc(args)
        model, tokenizer, _ = run_rl(args, model, tokenizer)
        run_eval(args, model, tokenizer)


if __name__ == "__main__":
    main()
