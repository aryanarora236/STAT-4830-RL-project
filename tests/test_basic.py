"""
Validation tests for the RLM retrieval system.

Tests:
- Week 4: Edge cases for deterministic agent (missing/multiple needles, long haystack)
- Week 6: Multi-step tasks (KV extraction, aggregation)
- Week 7: Reward computation and training loop
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils import (
    generate_task,
    generate_kv_extraction_task,
    generate_multistep_task,
    compute_reward,
)
from src.models import (
    DeterministicAgent,
    HeuristicMultiStepAgent,
    EvaluationFramework,
    TrainingLoop,
)


# ---- Week 4 tests ----

def test_missing_needle():
    """Missing needle (num_needles=0) should return 'Needle not found'."""
    print("=== Test: Missing Needle ===")
    haystack, question, correct = generate_task(num_sentences=10, num_needles=0)
    agent = DeterministicAgent()
    predicted, _ = agent.run_episode(haystack, question, correct)
    passed = predicted == correct or "not found" in predicted.lower()
    print(f"  Expected: {correct}  Got: {predicted}  {'PASS' if passed else 'FAIL'}")
    return passed


def test_multiple_needles():
    """Multiple needles should still return the correct value."""
    print("=== Test: Multiple Needles ===")
    haystack, question, correct = generate_task(num_sentences=10, num_needles=2)
    agent = DeterministicAgent()
    predicted, _ = agent.run_episode(haystack, question, correct)
    passed = predicted == correct
    print(f"  Expected: {correct}  Got: {predicted}  {'PASS' if passed else 'FAIL'}")
    return passed


def test_long_haystack():
    """Long haystack (30 sentences) should still work."""
    print("=== Test: Long Haystack ===")
    haystack, question, correct = generate_task(num_sentences=30, num_needles=1)
    agent = DeterministicAgent()
    predicted, transcript = agent.run_episode(haystack, question, correct)
    passed = predicted == correct
    runtime = transcript[0]["exec_result"]["runtime_sec"] if transcript else -1
    print(f"  Haystack: {len(haystack)} chars  Runtime: {runtime:.4f}s  {'PASS' if passed else 'FAIL'}")
    return passed


# ---- Week 6 tests ----

def test_kv_extraction():
    """HeuristicMultiStepAgent should filter log lines and extract a field."""
    print("=== Test: KV Extraction (Multi-Step) ===")
    context, question, correct = generate_kv_extraction_task(num_entries=20)
    agent = HeuristicMultiStepAgent()
    predicted, transcript = agent.run_episode(context, question, correct)
    passed = predicted == correct
    print(f"  Expected: {correct}  Got: {predicted}  Steps: {len(transcript)}  {'PASS' if passed else 'FAIL'}")
    return passed


def test_aggregation():
    """HeuristicMultiStepAgent should find METRIC_* keys and sum their values."""
    print("=== Test: Aggregation (Multi-Step) ===")
    context, question, correct = generate_multistep_task(num_sentences=15, num_keys=3)
    agent = HeuristicMultiStepAgent()
    predicted, transcript = agent.run_episode(context, question, correct)
    passed = predicted == correct
    print(f"  Expected: {correct}  Got: {predicted}  Steps: {len(transcript)}  {'PASS' if passed else 'FAIL'}")
    return passed


# ---- Week 7 tests ----

def test_reward_computation():
    """Reward function should penalise steps and reward correctness."""
    print("=== Test: Reward Computation ===")
    r_correct = compute_reward(is_correct=True, num_steps=1, max_steps=10)
    r_wrong = compute_reward(is_correct=False, num_steps=1, max_steps=10)
    r_many_steps = compute_reward(is_correct=True, num_steps=10, max_steps=10)

    checks = [
        r_correct > r_wrong,            # correct > wrong
        r_correct > r_many_steps,        # fewer steps > many steps
        r_wrong < 0.1,                   # wrong answer should be near 0
        r_correct > 0.9,                 # correct + few steps should be high
    ]
    passed = all(checks)
    print(f"  correct/1step={r_correct:.3f}  wrong/1step={r_wrong:.3f}  correct/10steps={r_many_steps:.3f}  {'PASS' if passed else 'FAIL'}")
    return passed


def test_training_loop():
    """Training loop should collect trajectories and compute stats."""
    print("=== Test: Training Loop ===")
    agent = DeterministicAgent()
    task_gen = lambda: generate_task(num_sentences=10, num_needles=1)
    trainer = TrainingLoop(agent=agent, task_generator=task_gen, batch_size=4)
    stats = trainer.train_step()
    passed = (
        stats["batch_size"] == 4
        and stats["accuracy"] >= 0.0
        and stats["buffer_size"] == 4
    )
    print(f"  Stats: {stats}  {'PASS' if passed else 'FAIL'}")
    return passed


# ---- Runner ----

def run_all_tests():
    """Run all test cases and display summary."""
    print("=" * 60)
    print("RUNNING ALL TESTS")
    print("=" * 60)

    tests = [
        ("Missing Needle", test_missing_needle),
        ("Multiple Needles", test_multiple_needles),
        ("Long Haystack", test_long_haystack),
        ("KV Extraction", test_kv_extraction),
        ("Aggregation", test_aggregation),
        ("Reward Computation", test_reward_computation),
        ("Training Loop", test_training_loop),
    ]

    results = []
    for name, fn in tests:
        try:
            results.append((name, fn()))
        except Exception as e:
            print(f"  ERROR: {e}")
            results.append((name, False))

    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    for name, passed in results:
        print(f"  {name}: {'PASS' if passed else 'FAIL'}")
    total = len(results)
    passed_count = sum(1 for _, p in results if p)
    print(f"\n  {passed_count}/{total} tests passed")
    print("=" * 60)
    return all(p for _, p in results)


if __name__ == "__main__":
    run_all_tests()
