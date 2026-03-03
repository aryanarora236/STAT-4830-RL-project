"""
Training entry point for the RLM agent.

Usage:
    # Full pipeline: behavior cloning warm-start + REINFORCE
    python scripts/train.py --phase full --model Qwen/Qwen2.5-Coder-1.5B-Instruct

    # Phase 1 only: behavior cloning on heuristic trajectories
    python scripts/train.py --phase bc --model Qwen/Qwen2.5-Coder-1.5B-Instruct --episodes 500

    # Phase 2 only: REINFORCE from a BC checkpoint
    python scripts/train.py --phase rl --model ./checkpoints/bc --rl-iterations 20

    # Evaluate a trained checkpoint
    python scripts/train.py --phase eval --model ./checkpoints/rl
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils import generate_task, generate_kv_extraction_task, generate_multistep_task
from src.models import HeuristicMultiStepAgent, LocalLLMAgent, EvaluationFramework
from src.training import (
    load_model,
    collect_trajectories,
    BehaviorCloningTrainer,
    ReinforceTrainer,
)


def get_task_generators():
    """Return the three task generator functions."""
    return [
        lambda: generate_task(num_sentences=15, num_needles=1),
        lambda: generate_kv_extraction_task(num_entries=20),
        lambda: generate_multistep_task(num_sentences=15, num_keys=3),
    ]


def run_bc(args):
    """Phase 1: Behavior cloning on heuristic agent trajectories."""
    print("=" * 60)
    print("PHASE 1: BEHAVIOR CLONING")
    print("=" * 60)

    # Collect trajectories with the heuristic agent
    agent = HeuristicMultiStepAgent()
    task_gens = get_task_generators()

    print(f"\nCollecting {args.episodes} trajectories with HeuristicMultiStepAgent...")
    trajectories = collect_trajectories(agent, task_gens, args.episodes)

    successful = sum(1 for t in trajectories if t.is_correct)
    print(f"Collected: {len(trajectories)} total, {successful} successful")

    # Load model
    print(f"\nLoading model: {args.model}")
    model, tokenizer = load_model(
        args.model,
        load_in_4bit=not args.full_precision,
        lora_r=args.lora_r,
    )

    # Train
    bc_trainer = BehaviorCloningTrainer(
        model,
        tokenizer,
        output_dir=args.bc_output,
        num_epochs=args.bc_epochs,
        batch_size=args.batch_size,
        learning_rate=args.bc_lr,
    )
    result = bc_trainer.train(trajectories)
    print(f"\nBC training complete: {result}")

    return model, tokenizer


def run_rl(args, model=None, tokenizer=None):
    """Phase 2: REINFORCE training."""
    print("\n" + "=" * 60)
    print("PHASE 2: REINFORCE")
    print("=" * 60)

    if model is None:
        load_path = args.rl_model or args.bc_output
        print(f"\nLoading model from: {load_path}")
        model, tokenizer = load_model(
            load_path,
            load_in_4bit=not args.full_precision,
        )

    task_gens = get_task_generators()

    rl_trainer = ReinforceTrainer(
        model,
        tokenizer,
        task_gens,
        output_dir=args.rl_output,
        batch_size=args.rl_batch_size,
        learning_rate=args.rl_lr,
    )

    history = rl_trainer.train(num_iterations=args.rl_iterations)

    # Print summary
    if history:
        first = history[0]
        last = history[-1]
        print(f"\nTraining summary:")
        print(f"  Accuracy:  {first['accuracy']:.1%} -> {last['accuracy']:.1%}")
        print(f"  Reward:    {first['avg_reward']:.3f} -> {last['avg_reward']:.3f}")
        print(f"  Loss:      {first['loss']:.4f} -> {last['loss']:.4f}")

    return model, tokenizer, history


def run_eval(args, model=None, tokenizer=None):
    """Evaluate a trained model on all task types."""
    print("\n" + "=" * 60)
    print("EVALUATION")
    print("=" * 60)

    if model is None:
        load_path = args.eval_model or args.rl_output
        print(f"\nLoading model from: {load_path}")
        model, tokenizer = load_model(
            load_path,
            load_in_4bit=not args.full_precision,
        )

    agent = LocalLLMAgent(model, tokenizer, max_steps=5)

    # Also compare against heuristic baseline
    heuristic = HeuristicMultiStepAgent()

    task_configs = [
        ("Needle", lambda: generate_task(num_sentences=15, num_needles=1)),
        ("KV Extraction", lambda: generate_kv_extraction_task(num_entries=20)),
        ("Aggregation", lambda: generate_multistep_task(num_sentences=15, num_keys=3)),
    ]

    for name, gen in task_configs:
        print(f"\n--- {name} ---")
        eval_fw = EvaluationFramework(
            agents=[agent, heuristic],
            task_generator=gen,
            num_episodes=args.eval_episodes,
        )
        eval_fw.run_evaluation()
        eval_fw.display_results()


def main():
    parser = argparse.ArgumentParser(
        description="RLM Training Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--phase",
        choices=["bc", "rl", "full", "eval"],
        default="full",
        help="Training phase: bc (behavior cloning), rl (REINFORCE), "
             "full (bc+rl+eval), eval (evaluate checkpoint)",
    )
    parser.add_argument(
        "--model",
        default="Qwen/Qwen2.5-Coder-1.5B-Instruct",
        help="Base model ID (HuggingFace) or checkpoint path",
    )

    # BC arguments
    parser.add_argument("--episodes", type=int, default=500,
                        help="Number of trajectories to collect for BC")
    parser.add_argument("--bc-epochs", type=int, default=3)
    parser.add_argument("--bc-lr", type=float, default=2e-4)
    parser.add_argument("--bc-output", default="./checkpoints/bc")
    parser.add_argument("--batch-size", type=int, default=4)

    # RL arguments
    parser.add_argument("--rl-model", default=None,
                        help="Checkpoint to start RL from (default: BC output)")
    parser.add_argument("--rl-iterations", type=int, default=20)
    parser.add_argument("--rl-batch-size", type=int, default=8)
    parser.add_argument("--rl-lr", type=float, default=1e-5)
    parser.add_argument("--rl-output", default="./checkpoints/rl")

    # Eval arguments
    parser.add_argument("--eval-model", default=None,
                        help="Checkpoint to evaluate (default: RL output)")
    parser.add_argument("--eval-episodes", type=int, default=20)

    # Model arguments
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--full-precision", action="store_true",
                        help="Disable 4-bit quantization (uses more VRAM)")

    args = parser.parse_args()

    if args.phase == "bc":
        run_bc(args)
    elif args.phase == "rl":
        run_rl(args)
    elif args.phase == "eval":
        run_eval(args)
    elif args.phase == "full":
        model, tokenizer = run_bc(args)
        model, tokenizer, history = run_rl(args, model, tokenizer)
        run_eval(args, model, tokenizer)


if __name__ == "__main__":
    main()
