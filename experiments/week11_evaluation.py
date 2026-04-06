"""
Week 11 Experiment: Comprehensive Heuristic Evaluation + Trajectory Collection

Runs:
1. Heuristic agent evaluation (500 episodes) with detailed metrics
2. Trajectory collection with reasoning traces (500 episodes)
3. Analysis of adjustment rates, action distributions, opponent modeling
4. Saves results to experiments/results/
"""

import sys
import os
import json
import random
import time
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.poker.tasks import generate_poker_task, generate_poker_task_with_trace
from src.poker.agents import PokerHeuristicAgent
from src.poker.evaluation import PokerEvaluationFramework
from src.poker.rewards import parse_action, compute_poker_reward_simple
from src.poker.heuristic import HeuristicPokerBot, parse_opponent_stats
from src.poker.training import collect_poker_trajectories_with_traces

random.seed(42)

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


def run_heuristic_evaluation(num_episodes=500):
    """Run full evaluation of the heuristic agent."""
    print(f"\n{'='*60}")
    print(f"EXPERIMENT 1: Heuristic Agent Evaluation ({num_episodes} episodes)")
    print(f"{'='*60}")

    agent = PokerHeuristicAgent()
    framework = PokerEvaluationFramework(
        agents=[agent],
        task_generator=generate_poker_task,
        num_episodes=num_episodes,
    )

    start = time.time()
    framework.run_evaluation()
    elapsed = time.time() - start

    framework.display_results()
    framework.display_confusion_matrix("PokerHeuristicAgent")

    results = framework.results["PokerHeuristicAgent"]
    print(f"\nTime: {elapsed:.1f}s ({elapsed/num_episodes:.3f}s/episode)")

    return results


def run_detailed_heuristic_analysis(num_episodes=500):
    """Run heuristic bot directly to get detailed reasoning traces."""
    print(f"\n{'='*60}")
    print(f"EXPERIMENT 2: Detailed Heuristic Analysis ({num_episodes} episodes)")
    print(f"{'='*60}")

    bot = HeuristicPokerBot()

    # Track detailed metrics
    action_dist = Counter()
    action_by_street = defaultdict(Counter)
    adjustment_count = 0
    adjustment_types = Counter()
    tier_dist = Counter()
    hand_category_dist = Counter()
    adjustment_details = []

    # Track opponent archetype vs adjustment correlation
    adjustment_by_context = defaultdict(int)
    no_adjustment_by_context = defaultdict(int)

    for i in range(num_episodes):
        context, question, answer, trace_text = generate_poker_task_with_trace()

        # Parse the trace to extract details
        action_type, action_amt = parse_action(answer)
        action_dist[action_type] += 1

        # Detect street
        street = "preflop"
        if "(Flop)" in context:
            street = "flop"
        elif "(Turn)" in context:
            street = "turn"
        elif "(River)" in context:
            street = "river"
        action_by_street[street][action_type] += 1

        # Parse trace details
        if "Tier" in trace_text:
            import re
            tier_match = re.search(r"Tier (\d)", trace_text)
            if tier_match:
                tier_dist[int(tier_match.group(1))] += 1

        for cat in ["monster", "strong", "medium", "weak", "nothing"]:
            if f"({cat}," in trace_text:
                hand_category_dist[cat] += 1

        # Check for adjustments
        if "no adjustment" not in trace_text.lower() and "no exploitable" not in trace_text.lower():
            if "Adjustment:" in trace_text:
                adj_line = [l for l in trace_text.split("\n") if "Adjustment:" in l]
                if adj_line:
                    adj_text = adj_line[0].strip()
                    if "no adjustment" not in adj_text.lower() and "no exploitable" not in adj_text.lower():
                        adjustment_count += 1
                        adjustment_types[adj_text] += 1
                        adjustment_details.append({
                            "street": street,
                            "action": answer,
                            "adjustment": adj_text,
                        })

        if (i + 1) % 100 == 0:
            print(f"  Processed {i+1}/{num_episodes} episodes...")

    # Print results
    total = num_episodes
    print(f"\n--- Action Distribution ---")
    for action, count in action_dist.most_common():
        print(f"  {action:8s}: {count:4d} ({count/total:.1%})")

    print(f"\n--- Actions by Street ---")
    for street in ["preflop", "flop", "turn", "river"]:
        if street in action_by_street:
            total_street = sum(action_by_street[street].values())
            print(f"  {street}:")
            for action, count in action_by_street[street].most_common():
                print(f"    {action:8s}: {count:4d} ({count/total_street:.1%})")

    print(f"\n--- Preflop Tier Distribution ---")
    for tier in sorted(tier_dist.keys()):
        count = tier_dist[tier]
        print(f"  Tier {tier}: {count:4d} ({count/max(sum(tier_dist.values()),1):.1%})")

    print(f"\n--- Postflop Hand Categories ---")
    for cat in ["monster", "strong", "medium", "weak", "nothing"]:
        count = hand_category_dist.get(cat, 0)
        total_pp = sum(hand_category_dist.values()) or 1
        print(f"  {cat:10s}: {count:4d} ({count/total_pp:.1%})")

    print(f"\n--- Opponent Adjustments ---")
    print(f"  Total adjustments: {adjustment_count}/{total} ({adjustment_count/total:.1%})")
    print(f"\n  Adjustment types:")
    for adj, count in adjustment_types.most_common(10):
        print(f"    [{count:3d}] {adj}")

    results = {
        "num_episodes": num_episodes,
        "action_distribution": dict(action_dist),
        "action_by_street": {k: dict(v) for k, v in action_by_street.items()},
        "tier_distribution": dict(tier_dist),
        "hand_category_distribution": dict(hand_category_dist),
        "adjustment_rate": adjustment_count / total,
        "adjustment_count": adjustment_count,
        "adjustment_types": dict(adjustment_types),
        "sample_adjustments": adjustment_details[:20],
    }

    return results


def run_trajectory_collection(num_episodes=500):
    """Collect BC training trajectories with reasoning traces."""
    print(f"\n{'='*60}")
    print(f"EXPERIMENT 3: Trajectory Collection ({num_episodes} episodes)")
    print(f"{'='*60}")

    start = time.time()
    trajectories = collect_poker_trajectories_with_traces(num_episodes=num_episodes)
    elapsed = time.time() - start

    # Analyze trajectories
    correct = sum(1 for t in trajectories if t.is_correct)
    has_code = sum(1 for t in trajectories if t.code)
    parsed_stats = sum(1 for t in trajectories if t.parsed_stats)

    action_types = Counter(t.action_type for t in trajectories)
    avg_code_len = sum(len(t.code) for t in trajectories) / len(trajectories)

    print(f"\n  Collected: {len(trajectories)} trajectories in {elapsed:.1f}s")
    print(f"  Correct: {correct}/{len(trajectories)} ({correct/len(trajectories):.1%})")
    print(f"  Has code: {has_code}/{len(trajectories)}")
    print(f"  Parsed stats: {parsed_stats}/{len(trajectories)} ({parsed_stats/len(trajectories):.1%})")
    print(f"  Avg code length: {avg_code_len:.0f} chars")
    print(f"\n  Action distribution:")
    for action, count in action_types.most_common():
        print(f"    {action:8s}: {count:4d} ({count/len(trajectories):.1%})")

    # Check context lengths
    context_lens = [len(t.context) for t in trajectories]
    print(f"\n  Context length: min={min(context_lens)}, max={max(context_lens)}, avg={sum(context_lens)/len(context_lens):.0f}")

    results = {
        "num_trajectories": len(trajectories),
        "correct": correct,
        "has_code": has_code,
        "parsed_stats": parsed_stats,
        "action_distribution": dict(action_types),
        "avg_code_length": avg_code_len,
        "context_length_stats": {
            "min": min(context_lens),
            "max": max(context_lens),
            "avg": sum(context_lens) / len(context_lens),
        },
        "collection_time_sec": elapsed,
    }

    return results


def run_heuristic_agent_eval(num_episodes=200):
    """Run the PokerHeuristicAgent through the full REPL pipeline."""
    print(f"\n{'='*60}")
    print(f"EXPERIMENT 4: Heuristic Agent REPL Pipeline ({num_episodes} episodes)")
    print(f"{'='*60}")

    agent = PokerHeuristicAgent()

    correct = 0
    code_executed = 0
    parsed_stats_in_output = 0
    errors = 0
    total_steps = 0

    for i in range(num_episodes):
        context, question, answer = generate_poker_task()
        predicted, transcript = agent.run_episode(context, question, answer)

        pred_type, _ = parse_action(predicted)
        corr_type, _ = parse_action(answer)

        if pred_type == corr_type:
            correct += 1

        for step in transcript:
            exec_result = step.get("exec_result", {})
            if exec_result.get("ok"):
                code_executed += 1
            else:
                errors += 1
            stdout = exec_result.get("stdout", "")
            if "VPIP" in stdout or "vpip" in stdout:
                parsed_stats_in_output += 1
        total_steps += len(transcript)

        if (i + 1) % 50 == 0:
            print(f"  Processed {i+1}/{num_episodes}... acc={correct/(i+1):.1%}")

    total = num_episodes
    print(f"\n  Results:")
    print(f"  Action type accuracy: {correct}/{total} ({correct/total:.1%})")
    print(f"  Code execution success: {code_executed}/{total_steps} ({code_executed/total_steps:.1%})")
    print(f"  Errors: {errors}/{total_steps}")
    print(f"  Stats parsed in output: {parsed_stats_in_output}/{total} ({parsed_stats_in_output/total:.1%})")
    print(f"  Avg steps per episode: {total_steps/total:.1f}")

    return {
        "accuracy": correct / total,
        "code_success_rate": code_executed / total_steps,
        "stats_parsed_rate": parsed_stats_in_output / total,
        "avg_steps": total_steps / total,
        "errors": errors,
    }


if __name__ == "__main__":
    all_results = {}

    # Run all experiments
    all_results["heuristic_evaluation"] = run_heuristic_evaluation(500)
    all_results["detailed_analysis"] = run_detailed_heuristic_analysis(500)
    all_results["trajectory_collection"] = run_trajectory_collection(500)
    all_results["agent_repl_pipeline"] = run_heuristic_agent_eval(200)

    # Save results
    output_path = os.path.join(RESULTS_DIR, "week11_results.json")

    # Convert Counter objects and non-serializable types
    def make_serializable(obj):
        if isinstance(obj, Counter):
            return dict(obj)
        if isinstance(obj, defaultdict):
            return dict(obj)
        return obj

    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    print(f"\n{'='*60}")
    print(f"All results saved to {output_path}")
    print(f"{'='*60}")
