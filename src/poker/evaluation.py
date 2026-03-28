"""
Evaluation framework for poker agents.

Compares agents on poker tasks, tracks per-action and per-street accuracy,
and measures opponent exploitation rate.
"""

from typing import List, Dict, Any, Callable, Tuple
from src.models import Agent
from src.poker.rewards import parse_action, compute_poker_reward_simple


class PokerEvaluationFramework:
    """Evaluate poker agents on generated tasks."""

    def __init__(
        self,
        agents: List[Agent],
        task_generator: Callable,
        num_episodes: int = 50,
    ):
        self.agents = agents
        self.task_generator = task_generator
        self.num_episodes = num_episodes
        self.results: Dict[str, Dict[str, Any]] = {}

    def run_evaluation(self):
        """Run all agents on the same set of tasks."""
        # Generate tasks once
        tasks = [self.task_generator() for _ in range(self.num_episodes)]

        for agent in self.agents:
            metrics = {
                "correct": 0,
                "type_match": 0,
                "total": 0,
                "by_action": {},       # action_type -> {correct, total}
                "by_street": {},       # street -> {correct, total}
                "total_reward": 0.0,
                "total_steps": 0,
                "predictions": [],
            }

            for context, question, answer in tasks:
                # Detect street from context
                street = "preflop"
                if "Flop)" in context:
                    street = "flop"
                elif "Turn)" in context:
                    street = "turn"
                elif "River)" in context:
                    street = "river"

                predicted, transcript = agent.run_episode(context, question, answer)

                # Exact match
                exact = (predicted.strip().lower() == answer.strip().lower())
                type_match = compute_poker_reward_simple(predicted, answer)

                metrics["total"] += 1
                if exact:
                    metrics["correct"] += 1
                if type_match >= 0.9:
                    metrics["type_match"] += 1
                metrics["total_reward"] += type_match
                metrics["total_steps"] += len(transcript)

                # Per-action breakdown
                _, _ = parse_action(answer)
                corr_type = parse_action(answer)[0]
                if corr_type not in metrics["by_action"]:
                    metrics["by_action"][corr_type] = {"correct": 0, "total": 0}
                metrics["by_action"][corr_type]["total"] += 1
                if type_match >= 0.9:
                    metrics["by_action"][corr_type]["correct"] += 1

                # Per-street breakdown
                if street not in metrics["by_street"]:
                    metrics["by_street"][street] = {"correct": 0, "total": 0}
                metrics["by_street"][street]["total"] += 1
                if type_match >= 0.9:
                    metrics["by_street"][street]["correct"] += 1

                metrics["predictions"].append({
                    "predicted": predicted,
                    "correct": answer,
                    "exact_match": exact,
                    "type_match": type_match,
                    "street": street,
                })

            self.results[agent.name] = metrics

    def display_results(self):
        """Print evaluation results."""
        for name, m in self.results.items():
            total = max(m["total"], 1)
            print(f"\n{'=' * 50}")
            print(f"Agent: {name}")
            print(f"{'=' * 50}")
            print(f"Exact match:  {m['correct']}/{total} ({m['correct']/total:.1%})")
            print(f"Type match:   {m['type_match']}/{total} ({m['type_match']/total:.1%})")
            print(f"Avg reward:   {m['total_reward']/total:.3f}")
            print(f"Avg steps:    {m['total_steps']/total:.1f}")

            if m["by_action"]:
                print(f"\nPer-action accuracy:")
                for action, counts in sorted(m["by_action"].items()):
                    t = max(counts["total"], 1)
                    print(f"  {action:8s}: {counts['correct']}/{counts['total']} ({counts['correct']/t:.0%})")

            if m["by_street"]:
                print(f"\nPer-street accuracy:")
                for street, counts in sorted(m["by_street"].items()):
                    t = max(counts["total"], 1)
                    print(f"  {street:8s}: {counts['correct']}/{counts['total']} ({counts['correct']/t:.0%})")

    def get_confusion_matrix(self, agent_name: str) -> Dict[str, Dict[str, int]]:
        """Get predicted vs correct action confusion matrix."""
        matrix: Dict[str, Dict[str, int]] = {}
        actions = ["fold", "check", "call", "raise"]
        for a in actions:
            matrix[a] = {b: 0 for b in actions}

        if agent_name not in self.results:
            return matrix

        for pred in self.results[agent_name]["predictions"]:
            pred_type = parse_action(pred["predicted"])[0]
            corr_type = parse_action(pred["correct"])[0]
            if pred_type in matrix and corr_type in matrix[pred_type]:
                matrix[corr_type][pred_type] += 1

        return matrix

    def display_confusion_matrix(self, agent_name: str):
        """Print confusion matrix."""
        matrix = self.get_confusion_matrix(agent_name)
        actions = ["fold", "check", "call", "raise"]

        print(f"\nConfusion matrix for {agent_name}:")
        header = f"{'':8s} | " + " | ".join(f"{a:6s}" for a in actions)
        print(header)
        print("-" * len(header))
        for true_action in actions:
            row = f"{true_action:8s} | "
            row += " | ".join(f"{matrix[true_action][pred]:6d}" for pred in actions)
            print(row)
