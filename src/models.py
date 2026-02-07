"""
Agent models for the RLM needle-in-haystack retrieval system.

This module provides:
- Agent: Abstract base class for all RLM agents
- DeterministicAgent: Baseline agent using regex search
- EvaluationFramework: Orchestrates evaluation and collects metrics
"""

from abc import ABC, abstractmethod
from typing import List, Tuple, Dict, Any
import time
import regex as re

from src.utils import safe_execute_code


class Agent(ABC):
    """Abstract base class for all RLM agents."""

    def __init__(self, name: str, max_steps: int = 10):
        """
        Initialize an agent.
        
        Args:
            name: Name identifier for the agent
            max_steps: Maximum number of REPL steps allowed per episode
        """
        self.name = name
        self.max_steps = max_steps
        self.transcript = []  # To store REPL interactions

    @abstractmethod
    def run_episode(self, haystack: str, question: str, correct_answer: str) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Runs a single episode for the agent to find the needle in the haystack.

        Args:
            haystack: The text containing the needle
            question: The question to answer based on the haystack
            correct_answer: The expected correct answer (for evaluation)

        Returns:
            Tuple of (predicted_answer, transcript):
                - predicted_answer: The agent's predicted answer
                - transcript: List of REPL interaction dictionaries
        """
        pass


class DeterministicAgent(Agent):
    """
    A deterministic agent that uses regex to find the needle in the haystack.
    
    Algorithm:
    1. Parse question to extract needle key
    2. Generate regex pattern: {KEY}=([a-zA-Z0-9]+)
    3. Execute regex search in REPL with CONTEXT variable
    4. Extract and return matched value
    """

    def __init__(self, name: str = 'DeterministicAgent', max_steps: int = 1):
        """
        Initialize the deterministic agent.
        
        Args:
            name: Name identifier (default: 'DeterministicAgent')
            max_steps: Maximum REPL steps (default: 1 for single regex search)
        """
        super().__init__(name, max_steps)

    def run_episode(self, haystack: str, question: str, correct_answer: str) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Run a single episode using regex search.
        
        Args:
            haystack: The text containing the needle
            question: The question to answer (format: "What is the value of KEY?")
            correct_answer: Expected answer (for evaluation only)
            
        Returns:
            Tuple of (predicted_answer, transcript)
        """
        self.transcript = []
        predicted_answer = ""

        # Parse the needle_key from the question
        match = re.search(r'What is the value of (.*?)\?', question)
        if not match:
            self.transcript.append({
                'step': 0,
                'action': 'Parse Question',
                'status': 'Failed',
                'output': 'Could not parse needle_key from question.'
            })
            return "N/A", self.transcript

        needle_key = match.group(1)

        # Construct a regex pattern to find the needle_key=VALUE pair
        regex_pattern = r"""%s=([a-zA-Z0-9]+)""" % re.escape(needle_key)

        # Generate Python code to execute in the REPL
        generated_code = f"""import re
search_pattern = r'{regex_pattern}'
match = re.search(search_pattern, CONTEXT)
if match:
    print(match.group(1))
else:
    print("Needle not found")
"""
        # Execute the generated code using safe_execute_code
        exec_result = safe_execute_code(generated_code, custom_globals={'CONTEXT': haystack})

        # Record the executed code and its ExecResult in the transcript
        self.transcript.append({
            'step': 1,
            'action': 'REPL Execution',
            'code': generated_code,
            'exec_result': {
                'ok': exec_result.ok,
                'stdout': exec_result.stdout,
                'stderr': exec_result.stderr,
                'runtime_sec': exec_result.runtime_sec
            }
        })

        # Extract the predicted answer from the stdout
        if exec_result.ok and exec_result.stdout:
            predicted_answer = exec_result.stdout.strip()
        elif not exec_result.ok:
            predicted_answer = f"Error: {exec_result.stderr.strip()}"
        else:
            predicted_answer = "No output from REPL"

        return predicted_answer, self.transcript


class EvaluationFramework:
    """
    Orchestrates evaluation of RLM agents on needle-in-a-haystack tasks.
    
    Implements the reward computation from the objective function:
    - Correctness (C): Binary check if predicted answer matches ground truth
    - Step penalty (∑S_t): Counted as number of REPL steps
    """

    def __init__(self, agents: List[Agent], task_generator: callable, num_episodes: int = 10):
        """
        Initialize the evaluation framework.
        
        Args:
            agents: List of Agent instances to evaluate
            task_generator: Function that returns (haystack, question, correct_answer)
            num_episodes: Number of evaluation episodes to run
        """
        self.agents = agents
        self.task_generator = task_generator
        self.num_episodes = num_episodes
        self.results = []

    def run_evaluation(self):
        """Runs evaluation episodes for all agents and collects metrics."""
        print(f"\n--- Starting Evaluation for {self.num_episodes} Episodes ---")
        for episode_idx in range(self.num_episodes):
            print(f"\n--- Episode {episode_idx + 1}/{self.num_episodes} ---")

            # Generate a new task for each episode
            haystack, question, correct_answer = self.task_generator()

            for agent in self.agents:
                start_time = time.time()
                predicted_answer, transcript = agent.run_episode(haystack, question, correct_answer)
                end_time = time.time()
                runtime = end_time - start_time

                # Determine correctness
                is_correct = (predicted_answer == correct_answer)

                self.results.append({
                    'episode_idx': episode_idx,
                    'agent_name': agent.name,
                    'question': question,
                    'correct_answer': correct_answer,
                    'predicted_answer': predicted_answer,
                    'is_correct': is_correct,
                    'runtime_sec': runtime,
                    'repl_steps': len(transcript),
                    'transcript': transcript
                })
                print(f"Agent: {agent.name}, Correct: {is_correct}, Predicted: '{predicted_answer}', Actual: '{correct_answer}'")
        print("--- Evaluation Completed ---")

    def display_results(self):
        """Displays aggregated results and a sample transcript."""
        if not self.results:
            print("No evaluation results to display. Run run_evaluation() first.")
            return

        print("\n--- Aggregated Results ---")
        agent_metrics = {}
        for res in self.results:
            agent_name = res['agent_name']
            if agent_name not in agent_metrics:
                agent_metrics[agent_name] = {'correct_count': 0, 'total_runtime': 0, 'total_repl_steps': 0, 'episode_count': 0}

            agent_metrics[agent_name]['correct_count'] += 1 if res['is_correct'] else 0
            agent_metrics[agent_name]['total_runtime'] += res['runtime_sec']
            agent_metrics[agent_name]['total_repl_steps'] += res['repl_steps']
            agent_metrics[agent_name]['episode_count'] += 1

        for agent_name, metrics in agent_metrics.items():
            accuracy = (metrics['correct_count'] / metrics['episode_count']) * 100
            avg_runtime = metrics['total_runtime'] / metrics['episode_count']
            avg_repl_steps = metrics['total_repl_steps'] / metrics['episode_count']
            print(f"\nAgent: {agent_name}")
            print(f"  Accuracy: {accuracy:.2f}%")
            print(f"  Avg Runtime: {avg_runtime:.4f} sec")
            print(f"  Avg REPL Steps: {avg_repl_steps:.2f}")

        print("\n--- Example Transcript (First Episode, First Agent) ---")
        if self.results:
            first_episode_transcript = self.results[0]['transcript']
            print(f"Agent: {self.results[0]['agent_name']}, Episode: {self.results[0]['episode_idx'] + 1}")
            print(f"Question: {self.results[0]['question']}")
            print(f"Correct Answer: {self.results[0]['correct_answer']}")
            print(f"Predicted Answer: {self.results[0]['predicted_answer']}")
            print("Transcript:")
            for entry in first_episode_transcript:
                print(f"  Step {entry['step']}: {entry['action']}")
                if 'code' in entry:
                    print(f"    Code: {entry['code'].strip().splitlines()[0]}...")
                if 'exec_result' in entry:
                    er = entry['exec_result']
                    print(f"    Exec OK: {er['ok']}")
                    if er['stdout']:
                        print(f"    Stdout: {er['stdout'].strip()}")
                    if er['stderr']:
                        print(f"    Stderr: {er['stderr'].strip()}")
                    print(f"    Runtime: {er['runtime_sec']:.4f} sec")
