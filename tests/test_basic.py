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
    LLMAgent,
    EvaluationFramework,
    TrainingLoop,
    extract_code_from_response,
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


# ---- Week 8 tests: LLM Agent ----

def test_code_extraction():
    """extract_code_from_response should parse code from markdown blocks."""
    print("=== Test: Code Extraction ===")

    # Case 1: ```python block
    resp1 = "Here is the code:\n```python\nprint('hello')\n```\nDone."
    code1 = extract_code_from_response(resp1)
    check1 = code1 == "print('hello')"

    # Case 2: plain ``` block
    resp2 = "```\nimport re\nm = re.search(r'KEY=(\\w+)', CONTEXT)\nprint(m.group(1))\n```"
    code2 = extract_code_from_response(resp2)
    check2 = code2 is not None and "re.search" in code2

    # Case 3: no code block at all (just explanation)
    resp3 = "The answer is 42. You should look at the data carefully."
    code3 = extract_code_from_response(resp3)
    check3 = code3 is None

    # Case 4: raw code without fences
    resp4 = "import re\nprint(re.findall(r'METRIC_\\w+=(\\d+)', CONTEXT))"
    code4 = extract_code_from_response(resp4)
    check4 = code4 is not None and "import re" in code4

    passed = check1 and check2 and check3 and check4
    print(f"  python block: {'OK' if check1 else 'FAIL'}")
    print(f"  plain block:  {'OK' if check2 else 'FAIL'}")
    print(f"  no code:      {'OK' if check3 else 'FAIL'}")
    print(f"  raw code:     {'OK' if check4 else 'FAIL'}")
    print(f"  {'PASS' if passed else 'FAIL'}")
    return passed


def _has_hf_token():
    """Check if HF_TOKEN is set for LLM integration tests."""
    return bool(os.environ.get("HF_TOKEN"))


def test_llm_agent_needle():
    """LLMAgent should find a needle in a haystack (requires HF_TOKEN)."""
    print("=== Test: LLM Agent - Needle ===")
    if not _has_hf_token():
        print("  SKIPPED (HF_TOKEN not set)")
        return True  # skip = pass

    haystack, question, correct = generate_task(num_sentences=10, num_needles=1)
    agent = LLMAgent(max_steps=5)
    predicted, transcript = agent.run_episode(haystack, question, correct)
    passed = predicted == correct
    print(f"  Expected: {correct}  Got: {predicted}  Steps: {len(transcript)}  {'PASS' if passed else 'FAIL'}")
    return passed


def test_llm_agent_kv():
    """LLMAgent should handle KV extraction tasks (requires HF_TOKEN)."""
    print("=== Test: LLM Agent - KV Extraction ===")
    if not _has_hf_token():
        print("  SKIPPED (HF_TOKEN not set)")
        return True

    context, question, correct = generate_kv_extraction_task(num_entries=15)
    agent = LLMAgent(max_steps=5)
    predicted, transcript = agent.run_episode(context, question, correct)
    passed = predicted == correct
    print(f"  Expected: {correct}  Got: {predicted}  Steps: {len(transcript)}  {'PASS' if passed else 'FAIL'}")
    return passed


def test_llm_agent_aggregation():
    """LLMAgent should handle aggregation tasks (requires HF_TOKEN)."""
    print("=== Test: LLM Agent - Aggregation ===")
    if not _has_hf_token():
        print("  SKIPPED (HF_TOKEN not set)")
        return True

    context, question, correct = generate_multistep_task(num_sentences=10, num_keys=3)
    agent = LLMAgent(max_steps=5)
    predicted, transcript = agent.run_episode(context, question, correct)
    passed = predicted == correct
    print(f"  Expected: {correct}  Got: {predicted}  Steps: {len(transcript)}  {'PASS' if passed else 'FAIL'}")
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
        ("Code Extraction", test_code_extraction),
        ("LLM Agent - Needle", test_llm_agent_needle),
        ("LLM Agent - KV", test_llm_agent_kv),
        ("LLM Agent - Aggregation", test_llm_agent_aggregation),
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
