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
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.training import load_model
from src.poker.tasks import generate_poker_task, generate_preflop_task, generate_postflop_task
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
        print(f"\nCollecting {args.episodes} trajectories via PokerHeuristicAgent...")
        heuristic_agent = PokerHeuristicAgent()
        trajectories = collect_poker_trajectories(
            heuristic_agent,
            num_episodes=args.episodes,
            task_generators=[generate_poker_task],
        )

    successful = sum(1 for t in trajectories if t.is_correct and t.code)
    print(f"Collected: {len(trajectories)} total, {successful} successful")

    # 2) Load base model (fresh LoRA or checkpoint)
    print(f"\nLoading model: {args.model}")
    model, tokenizer = load_model(
        model_id_or_path=args.model,
        load_in_4bit=not args.full_precision,
        lora_r=args.lora_r,
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
    )

    history = trainer.train(num_iterations=args.rl_iterations)
    if history:
        first = history[0]
        last = history[-1]
        print("\nREINFORCE summary:")
        print(f"  Accuracy: {first['accuracy']:.1%} -> {last['accuracy']:.1%}")
        print(f"  Reward:   {first['avg_reward']:.3f} -> {last['avg_reward']:.3f}")
        print(f"  Loss:     {first['loss']:.4f} -> {last['loss']:.4f}")

    return model, tokenizer, history


def _eval_one(name, task_gen, agents, episodes):
    print(f"\n--- {name} ---")
    fw = PokerEvaluationFramework(
        agents=agents,
        task_generator=task_gen,
        num_episodes=episodes,
    )
    fw.run_evaluation()
    fw.display_results()


def run_eval(args, model=None, tokenizer=None):
    """Evaluate local poker model on generated poker tasks."""
    print("\n" + "=" * 60)
    print("POKER EVALUATION")
    print("=" * 60)

    if model is None:
        load_path = args.eval_model or args.rl_output
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

    _eval_one("All Streets", generate_poker_task, agents, args.eval_episodes)
    if args.eval_by_street:
        _eval_one("Preflop", generate_preflop_task, agents, args.eval_episodes)
        _eval_one("Postflop", generate_postflop_task, agents, args.eval_episodes)


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

    # Shared/model
    parser.add_argument("--full-precision", action="store_true", help="Disable 4-bit quantization")
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--max-new-tokens", type=int, default=1024)

    # BC
    parser.add_argument("--episodes", type=int, default=500, help="Trajectory count for BC")
    parser.add_argument("--bc-source", choices=["traces", "agent"], default="traces")
    parser.add_argument("--bc-epochs", type=int, default=3)
    parser.add_argument("--bc-lr", type=float, default=2e-4)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--bc-output", default="./checkpoints/poker_bc")

    # RL
    parser.add_argument("--rl-model", default=None, help="Checkpoint to start RL from (default: BC output)")
    parser.add_argument("--rl-iterations", type=int, default=20)
    parser.add_argument("--rl-batch-size", type=int, default=12)
    parser.add_argument("--rl-lr", type=float, default=5e-6)
    parser.add_argument("--rl-baseline-ema", type=float, default=0.95)
    parser.add_argument("--rl-adv-clip", type=float, default=2.0)
    parser.add_argument("--rl-task-mode", choices=["all", "preflop", "postflop"], default="all")
    parser.add_argument("--rl-output", default="./checkpoints/poker_rl")

    # Eval
    parser.add_argument("--eval-model", default=None, help="Checkpoint to evaluate (default: RL output)")
    parser.add_argument("--eval-episodes", type=int, default=20)
    parser.add_argument("--eval-max-steps", type=int, default=5)
    parser.add_argument("--eval-temperature", type=float, default=0.1)
    parser.add_argument("--eval-by-street", action="store_true")
    parser.add_argument("--no-heuristic-baseline", action="store_true")

    args = parser.parse_args()

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
