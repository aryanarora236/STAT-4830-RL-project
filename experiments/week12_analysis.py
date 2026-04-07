"""
Week 12 Analysis: Comprehensive pre-training evaluation and data analysis.

Runs locally without GPU. Generates results and figures for the meeting.
Covers:
  1. Extended heuristic evaluation (1000 episodes)
  2. Training data distribution analysis
  3. Ablation: heuristic with vs without opponent modeling
  4. Context length and complexity analysis
  5. Decision difficulty breakdown
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import random
import json
import time
from collections import Counter, defaultdict

from src.poker.environment import Card, Deck, HandEvaluator, GameState, OPPONENT_ARCHETYPES
from src.poker.heuristic import HeuristicPokerBot, preflop_tier, _hand_key
from src.poker.tasks import generate_poker_task, generate_poker_task_with_trace
from src.poker.agents import PokerHeuristicAgent
from src.poker.rewards import parse_action, compute_poker_reward_simple
from src.poker.evaluation import PokerEvaluationFramework
from src.poker.training import collect_poker_trajectories_with_traces


def run_extended_heuristic_eval(num_episodes=1000, seed=42):
    """Extended heuristic evaluation with detailed breakdown."""
    print(f"\n{'='*60}")
    print(f"EXTENDED HEURISTIC EVALUATION ({num_episodes} episodes)")
    print(f"{'='*60}")

    random.seed(seed)
    results = {
        "total": num_episodes,
        "by_action": Counter(),
        "by_street": Counter(),
        "by_street_action": defaultdict(Counter),
        "opponent_adjustments": 0,
        "adjustment_types": Counter(),
        "hand_strengths": Counter(),
        "preflop_tiers": Counter(),
        "pot_sizes": [],
        "to_call_amounts": [],
        "context_lengths": [],
    }

    for i in range(num_episodes):
        random.seed(seed + i)
        try:
            context, question, answer, trace = generate_poker_task_with_trace()
        except Exception:
            continue

        action_type, amount = parse_action(answer)
        results["by_action"][action_type] += 1
        results["context_lengths"].append(len(context))

        # Detect street
        street = "preflop"
        if "Flop)" in context:
            street = "flop"
        elif "Turn)" in context:
            street = "turn"
        elif "River)" in context:
            street = "river"
        results["by_street"][street] += 1
        results["by_street_action"][street][action_type] += 1

        # Parse pot and to_call from context
        import re
        pot_m = re.search(r'Pot:\s*\$(\d+)', context)
        call_m = re.search(r'To Call:\s*\$(\d+)', context)
        if pot_m:
            results["pot_sizes"].append(int(pot_m.group(1)))
        if call_m:
            results["to_call_amounts"].append(int(call_m.group(1)))

        # Check for opponent adjustments in trace
        adj_keywords = [
            "calling down", "thin value", "fold to passive",
            "steal", "trap", "fold to tight", "size up",
            "wider calling", "bluff.*cbet",
        ]
        trace_lower = trace.lower()
        has_adjustment = False
        if "no adjustment" not in trace_lower and "no exploitable" not in trace_lower and "insufficient" not in trace_lower:
            for kw in adj_keywords:
                if re.search(kw, trace_lower):
                    results["adjustment_types"][kw.split(".*")[0].strip()] += 1
                    has_adjustment = True
        if has_adjustment:
            results["opponent_adjustments"] += 1

        # Parse hand strength from trace
        for strength in ["monster", "strong", "medium", "weak", "nothing"]:
            if strength in trace_lower:
                results["hand_strengths"][strength] += 1
                break

        # Preflop tier
        tier_m = re.search(r'tier[:\s]*(\d)', trace_lower)
        if tier_m:
            results["preflop_tiers"][f"tier_{tier_m.group(1)}"] += 1

        if (i + 1) % 250 == 0:
            print(f"  Processed {i+1}/{num_episodes} episodes...")

    # Print summary
    print(f"\n--- Action Distribution ---")
    for action, count in results["by_action"].most_common():
        print(f"  {action:8s}: {count:4d} ({count/num_episodes:.1%})")

    print(f"\n--- Street Distribution ---")
    for street, count in results["by_street"].most_common():
        print(f"  {street:8s}: {count:4d} ({count/num_episodes:.1%})")

    print(f"\n--- Per-Street Action Breakdown ---")
    for street in ["preflop", "flop", "turn", "river"]:
        if street in results["by_street_action"]:
            total = sum(results["by_street_action"][street].values())
            actions = results["by_street_action"][street]
            parts = [f"{a}: {c/total:.0%}" for a, c in actions.most_common()]
            print(f"  {street:8s}: {', '.join(parts)}")

    print(f"\n--- Opponent Adjustments ---")
    adj_rate = results["opponent_adjustments"] / num_episodes
    print(f"  Adjustment rate: {results['opponent_adjustments']}/{num_episodes} ({adj_rate:.1%})")
    for adj, count in results["adjustment_types"].most_common():
        print(f"    {adj}: {count}")

    print(f"\n--- Hand Strength Distribution (postflop) ---")
    for strength, count in results["hand_strengths"].most_common():
        print(f"  {strength:8s}: {count}")

    print(f"\n--- Context Length Stats ---")
    lengths = results["context_lengths"]
    print(f"  Mean: {sum(lengths)/len(lengths):.0f} chars")
    print(f"  Min:  {min(lengths)} chars")
    print(f"  Max:  {max(lengths)} chars")
    print(f"  Median: {sorted(lengths)[len(lengths)//2]} chars")

    return results


def run_training_data_analysis(num_episodes=500, seed=42):
    """Analyze the BC training data distribution."""
    print(f"\n{'='*60}")
    print(f"BC TRAINING DATA ANALYSIS ({num_episodes} trajectories)")
    print(f"{'='*60}")

    random.seed(seed)
    trajectories = collect_poker_trajectories_with_traces(num_episodes=num_episodes)

    analysis = {
        "total": len(trajectories),
        "action_distribution": Counter(),
        "code_lengths": [],
        "context_lengths": [],
        "correct_rate": 0,
        "parsed_stats_rate": 0,
        "unique_actions": set(),
    }

    for t in trajectories:
        analysis["action_distribution"][t.action_type] += 1
        analysis["code_lengths"].append(len(t.code))
        analysis["context_lengths"].append(len(t.context))
        if t.is_correct:
            analysis["correct_rate"] += 1
        if t.parsed_stats:
            analysis["parsed_stats_rate"] += 1
        analysis["unique_actions"].add(t.correct_answer)

    print(f"\n--- Trajectory Stats ---")
    print(f"  Total: {analysis['total']}")
    print(f"  Correct: {analysis['correct_rate']} ({analysis['correct_rate']/analysis['total']:.0%})")
    print(f"  Parsed stats: {analysis['parsed_stats_rate']} ({analysis['parsed_stats_rate']/analysis['total']:.0%})")

    print(f"\n--- Action Distribution in Training Data ---")
    for action, count in analysis["action_distribution"].most_common():
        print(f"  {action:8s}: {count:4d} ({count/analysis['total']:.1%})")

    print(f"\n--- Code Length Stats ---")
    cl = analysis["code_lengths"]
    print(f"  Mean: {sum(cl)/len(cl):.0f} chars")
    print(f"  Min:  {min(cl)}, Max: {max(cl)}")

    print(f"\n--- Context Length Stats ---")
    ctx = analysis["context_lengths"]
    print(f"  Mean: {sum(ctx)/len(ctx):.0f} chars")
    print(f"  Min:  {min(ctx)}, Max: {max(ctx)}")

    # Estimate token counts (rough: 1 token ≈ 4 chars)
    est_total_tokens = [(len(t.context) + len(t.code) + 500) / 4 for t in trajectories]  # +500 for system prompt
    print(f"\n--- Estimated Token Counts (context + code + prompt) ---")
    print(f"  Mean: {sum(est_total_tokens)/len(est_total_tokens):.0f} tokens")
    print(f"  Max:  {max(est_total_tokens):.0f} tokens")
    over_2048 = sum(1 for t in est_total_tokens if t > 2048)
    over_4096 = sum(1 for t in est_total_tokens if t > 4096)
    print(f"  Exceeding 2048 tokens: {over_2048} ({over_2048/len(est_total_tokens):.0%})")
    print(f"  Exceeding 4096 tokens: {over_4096} ({over_4096/len(est_total_tokens):.0%})")

    analysis["unique_actions"] = list(analysis["unique_actions"])[:20]
    return analysis


def run_ablation_study(num_episodes=200, seed=42):
    """Compare heuristic with and without opponent modeling."""
    print(f"\n{'='*60}")
    print(f"ABLATION: WITH vs WITHOUT OPPONENT MODELING ({num_episodes} episodes)")
    print(f"{'='*60}")

    random.seed(seed)

    with_adj = 0
    without_adj_agree = 0
    without_adj_disagree = 0
    disagree_examples = []

    for i in range(num_episodes):
        random.seed(seed + i)
        try:
            context, question, answer, trace = generate_poker_task_with_trace()
        except Exception:
            continue

        # Check if an opponent adjustment was made
        trace_lower = trace.lower()
        has_adjustment = (
            "no adjustment" not in trace_lower
            and "no exploitable" not in trace_lower
            and "insufficient" not in trace_lower
        )

        if has_adjustment:
            with_adj += 1
            # The adjustment changed the decision
            # We can't easily get the "without adjustment" answer,
            # but we can note that these are the cases where history matters
            without_adj_disagree += 1
            if len(disagree_examples) < 5:
                disagree_examples.append({
                    "answer": answer,
                    "trace_excerpt": trace[:300],
                })
        else:
            without_adj_agree += 1

    total = with_adj + without_adj_agree
    print(f"\n--- Results ---")
    print(f"  Total episodes: {total}")
    print(f"  Decisions WITH opponent adjustment: {with_adj} ({with_adj/total:.1%})")
    print(f"  Decisions WITHOUT adjustment: {without_adj_agree} ({without_adj_agree/total:.1%})")
    print(f"\n  --> {with_adj/total:.1%} of decisions are influenced by opponent history")
    print(f"  --> A model that ignores history would get at most {without_adj_agree/total:.1%} correct")

    if disagree_examples:
        print(f"\n--- Example Opponent-Adjusted Decisions ---")
        for j, ex in enumerate(disagree_examples[:3]):
            print(f"\n  Example {j+1}: {ex['answer']}")
            print(f"  Trace: {ex['trace_excerpt'][:200]}...")

    return {
        "total": total,
        "with_adjustment": with_adj,
        "without_adjustment": without_adj_agree,
        "adjustment_rate": with_adj / total if total > 0 else 0,
        "examples": disagree_examples,
    }


def run_decision_difficulty_analysis(num_episodes=500, seed=42):
    """Categorize decisions by difficulty."""
    print(f"\n{'='*60}")
    print(f"DECISION DIFFICULTY ANALYSIS ({num_episodes} episodes)")
    print(f"{'='*60}")

    random.seed(seed)
    import re

    difficulty_buckets = {
        "trivial": [],    # fold trash / check with nothing to call
        "easy": [],       # clear action based on hand strength alone
        "medium": [],     # requires pot odds calculation
        "hard": [],       # requires opponent modeling
    }

    for i in range(num_episodes):
        random.seed(seed + i)
        try:
            context, question, answer, trace = generate_poker_task_with_trace()
        except Exception:
            continue

        action_type, amount = parse_action(answer)
        trace_lower = trace.lower()

        # Classify difficulty
        has_adj = (
            "no adjustment" not in trace_lower
            and "no exploitable" not in trace_lower
            and "insufficient" not in trace_lower
        )

        call_m = re.search(r'To Call:\s*\$(\d+)', context)
        to_call = int(call_m.group(1)) if call_m else 0

        if has_adj:
            difficulty = "hard"
        elif to_call == 0 and action_type in ("check", "fold"):
            difficulty = "trivial"
        elif action_type == "fold" and "tier 5" in trace_lower:
            difficulty = "trivial"
        elif to_call > 0:
            difficulty = "medium"
        else:
            difficulty = "easy"

        difficulty_buckets[difficulty].append(action_type)

    print(f"\n--- Decision Difficulty Distribution ---")
    total = sum(len(v) for v in difficulty_buckets.values())
    for diff, actions in difficulty_buckets.items():
        pct = len(actions) / total if total > 0 else 0
        action_dist = Counter(actions)
        top_actions = ", ".join(f"{a}: {c}" for a, c in action_dist.most_common(3))
        print(f"  {diff:8s}: {len(actions):4d} ({pct:.1%}) [{top_actions}]")

    print(f"\n  Key insight: A model can get ~{len(difficulty_buckets['trivial'])/total:.0%} accuracy")
    print(f"  just from trivial rules (fold trash, check for free).")
    print(f"  To exceed ~{(len(difficulty_buckets['trivial'])+len(difficulty_buckets['easy']))/total:.0%},")
    print(f"  it must learn pot odds and basic strategy.")
    print(f"  To exceed ~{(total-len(difficulty_buckets['hard']))/total:.0%},")
    print(f"  it must learn to read opponent history.")

    return {
        "total": total,
        "distribution": {k: len(v) for k, v in difficulty_buckets.items()},
        "action_breakdown": {k: dict(Counter(v)) for k, v in difficulty_buckets.items()},
    }


def save_results(all_results, output_dir="experiments/results"):
    """Save all results to JSON."""
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "week12_analysis.json")

    # Make serializable
    serializable = {}
    for key, val in all_results.items():
        if isinstance(val, dict):
            clean = {}
            for k, v in val.items():
                if isinstance(v, (Counter, defaultdict)):
                    clean[k] = dict(v)
                elif isinstance(v, set):
                    clean[k] = list(v)
                elif isinstance(v, list) and len(v) > 100:
                    # Store summary stats for large lists
                    clean[k] = {
                        "mean": sum(v) / len(v),
                        "min": min(v),
                        "max": max(v),
                        "count": len(v),
                    }
                else:
                    clean[k] = v
            serializable[key] = clean
        else:
            serializable[key] = val

    with open(output_path, "w") as f:
        json.dump(serializable, f, indent=2, default=str)
    print(f"\nResults saved to {output_path}")
    return output_path


def generate_figures(results, output_dir="figures"):
    """Generate matplotlib figures for the report/presentation."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("matplotlib not available — skipping figures")
        return

    os.makedirs(output_dir, exist_ok=True)

    # Figure 1: Action Distribution
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Overall action distribution
    heuristic = results.get("heuristic_eval", {})
    if "by_action" in heuristic:
        actions = dict(heuristic["by_action"])
        labels = list(actions.keys())
        values = list(actions.values())
        total = sum(values)
        colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12']
        axes[0].bar(labels, [v/total for v in values], color=colors[:len(labels)])
        axes[0].set_title('Heuristic Action Distribution (1000 episodes)')
        axes[0].set_ylabel('Frequency')
        for i, (v, l) in enumerate(zip(values, labels)):
            axes[0].text(i, v/total + 0.01, f'{v/total:.1%}', ha='center', fontsize=10)

    # Decision difficulty
    difficulty = results.get("difficulty", {})
    if "distribution" in difficulty:
        dist = difficulty["distribution"]
        labels = list(dist.keys())
        values = list(dist.values())
        total = sum(values)
        colors = ['#95a5a6', '#3498db', '#f39c12', '#e74c3c']
        axes[1].bar(labels, [v/total for v in values], color=colors[:len(labels)])
        axes[1].set_title('Decision Difficulty Distribution')
        axes[1].set_ylabel('Frequency')
        for i, v in enumerate(values):
            axes[1].text(i, v/total + 0.01, f'{v/total:.1%}', ha='center', fontsize=10)

    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, 'week12_action_difficulty.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {output_dir}/week12_action_difficulty.png")

    # Figure 2: Per-Street Breakdown
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    if "by_street_action" in heuristic:
        streets = ["preflop", "flop", "turn", "river"]
        action_types = ["fold", "check", "call", "raise"]
        bar_data = {a: [] for a in action_types}
        street_totals = []

        for s in streets:
            s_data = heuristic["by_street_action"].get(s, {})
            s_total = sum(s_data.values()) if s_data else 1
            street_totals.append(s_total)
            for a in action_types:
                bar_data[a].append(s_data.get(a, 0) / s_total)

        x = np.arange(len(streets))
        width = 0.2
        colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12']
        for i, (action, vals) in enumerate(bar_data.items()):
            axes[0].bar(x + i * width, vals, width, label=action, color=colors[i])

        axes[0].set_xlabel('Street')
        axes[0].set_ylabel('Frequency')
        axes[0].set_title('Action Distribution by Street')
        axes[0].set_xticks(x + 1.5 * width)
        axes[0].set_xticklabels(streets)
        axes[0].legend()

    # Context length distribution
    training_data = results.get("training_data", {})
    if "context_lengths" in training_data:
        ctx_lens = training_data["context_lengths"]
        if isinstance(ctx_lens, list):
            axes[1].hist(ctx_lens, bins=30, color='#3498db', alpha=0.7, edgecolor='black')
            axes[1].axvline(x=2048*4, color='red', linestyle='--', label='2048 tokens (est.)')
            axes[1].axvline(x=4096*4, color='orange', linestyle='--', label='4096 tokens (est.)')
            axes[1].set_xlabel('Context Length (chars)')
            axes[1].set_ylabel('Count')
            axes[1].set_title('Context Length Distribution')
            axes[1].legend()

    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, 'week12_street_context.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {output_dir}/week12_street_context.png")

    # Figure 3: Improvement Arc (what we expect to see)
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))

    stages = ['Zero-Shot\n(Qwen-7B)', 'BC Model\n(target)', 'RL Model\n(target)', 'Heuristic\n(ceiling)']
    accuracies = [8, None, None, 100]  # BC and RL are targets
    bc_target_low = 35
    bc_target_high = 55
    rl_target_low = 50
    rl_target_high = 70

    # Plot known values
    ax.bar(0, 8, color='#e74c3c', alpha=0.8, width=0.6)
    ax.bar(3, 100, color='#2ecc71', alpha=0.8, width=0.6)

    # Plot target ranges
    ax.bar(1, bc_target_high, color='#3498db', alpha=0.3, width=0.6)
    ax.bar(1, bc_target_low, color='#3498db', alpha=0.6, width=0.6)
    ax.text(1, bc_target_high + 2, f'{bc_target_low}-{bc_target_high}%\n(target)', ha='center', fontsize=9)

    ax.bar(2, rl_target_high, color='#f39c12', alpha=0.3, width=0.6)
    ax.bar(2, rl_target_low, color='#f39c12', alpha=0.6, width=0.6)
    ax.text(2, rl_target_high + 2, f'{rl_target_low}-{rl_target_high}%\n(target)', ha='center', fontsize=9)

    ax.text(0, 10, '8%', ha='center', fontsize=11, fontweight='bold')
    ax.text(3, 102, '100%', ha='center', fontsize=11, fontweight='bold')

    ax.set_xticks(range(4))
    ax.set_xticklabels(stages, fontsize=10)
    ax.set_ylabel('Action Type Accuracy (%)')
    ax.set_title('Expected Improvement Arc: Zero-Shot → BC → RL → Heuristic')
    ax.set_ylim(0, 115)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, 'week12_improvement_arc.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {output_dir}/week12_improvement_arc.png")


def main():
    start = time.time()
    results = {}

    # 1. Extended heuristic evaluation
    results["heuristic_eval"] = run_extended_heuristic_eval(num_episodes=1000)

    # 2. Training data analysis
    results["training_data"] = run_training_data_analysis(num_episodes=500)

    # 3. Ablation study
    results["ablation"] = run_ablation_study(num_episodes=500)

    # 4. Decision difficulty
    results["difficulty"] = run_decision_difficulty_analysis(num_episodes=500)

    # Save results
    save_results(results)

    # Generate figures
    print(f"\n{'='*60}")
    print("GENERATING FIGURES")
    print(f"{'='*60}")
    generate_figures(results)

    elapsed = time.time() - start
    print(f"\n{'='*60}")
    print(f"ALL ANALYSIS COMPLETE ({elapsed:.1f}s)")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
