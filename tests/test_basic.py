"""
Basic validation tests for the RLM needle-in-haystack retrieval system.

Tests edge cases and validates core functionality:
- Missing needles (num_needles=0)
- Multiple needles (num_needles>1)
- Long haystacks (scalability)
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils import generate_task
from src.models import DeterministicAgent


def test_missing_needle():
    """Test Case 1: Missing needle (num_needles=0)"""
    print("=== Test Case 1: Missing Needle ===")
    haystack_missing, question_missing, correct_missing = generate_task(num_sentences=10, num_needles=0)
    agent_test = DeterministicAgent()
    predicted_missing, transcript_missing = agent_test.run_episode(haystack_missing, question_missing, correct_missing)
    print(f"Question: {question_missing}")
    print(f"Correct Answer: {correct_missing}")
    print(f"Predicted Answer: {predicted_missing}")
    test1_passed = predicted_missing == correct_missing or 'not found' in predicted_missing.lower()
    print(f"Test Passed: {test1_passed}\n")
    return test1_passed


def test_multiple_needles():
    """Test Case 2: Multiple needles (num_needles=2)"""
    print("=== Test Case 2: Multiple Needles ===")
    haystack_multi, question_multi, correct_multi = generate_task(num_sentences=10, num_needles=2)
    agent_test = DeterministicAgent()
    predicted_multi, transcript_multi = agent_test.run_episode(haystack_multi, question_multi, correct_multi)
    print(f"Question: {question_multi}")
    print(f"Correct Answer: {correct_multi}")
    print(f"Predicted Answer: {predicted_multi}")
    test2_passed = predicted_multi == correct_multi
    print(f"Test Passed: {test2_passed}\n")
    return test2_passed


def test_long_haystack():
    """Test Case 3: Very long haystack"""
    print("=== Test Case 3: Long Haystack ===")
    haystack_long, question_long, correct_long = generate_task(num_sentences=30, num_needles=1)
    agent_test = DeterministicAgent()
    predicted_long, transcript_long = agent_test.run_episode(haystack_long, question_long, correct_long)
    print(f"Haystack length: {len(haystack_long)} characters")
    print(f"Correct Answer: {correct_long}")
    print(f"Predicted Answer: {predicted_long}")
    test3_passed = predicted_long == correct_long
    print(f"Test Passed: {test3_passed}")
    print(f"REPL Steps: {len(transcript_long)}")
    if transcript_long:
        print(f"Runtime: {transcript_long[0]['exec_result']['runtime_sec']:.4f} sec")
    return test3_passed


def run_all_tests():
    """Run all test cases and display summary."""
    print("="*60)
    print("RUNNING EDGE CASE TESTS")
    print("="*60)
    
    test1_passed = test_missing_needle()
    test2_passed = test_multiple_needles()
    test3_passed = test_long_haystack()
    
    print("="*60)
    print("EDGE CASE TEST SUMMARY")
    print("="*60)
    print(f"Test 1 (Missing Needle): {'PASSED' if test1_passed else 'FAILED'}")
    print(f"Test 2 (Multiple Needles): {'PASSED' if test2_passed else 'FAILED'}")
    print(f"Test 3 (Long Haystack): {'PASSED' if test3_passed else 'FAILED'}")
    print("="*60)
    
    all_passed = test1_passed and test2_passed and test3_passed
    return all_passed


if __name__ == "__main__":
    run_all_tests()
