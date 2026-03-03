"""
Agent models for the RLM needle-in-haystack retrieval system.

This module provides:
- Agent: Abstract base class for all RLM agents
- DeterministicAgent: Baseline agent using regex search (Week 4)
- HeuristicMultiStepAgent: Multi-step agent using template strategies (Week 6)
- LLMAgent: LLM-based agent using HuggingFace Inference API (Week 8)
- EvaluationFramework: Orchestrates evaluation and collects metrics
- TrainingLoop: Skeleton for imitation-learning / reward-weighted training (Week 7)
"""

from abc import ABC, abstractmethod
from typing import List, Tuple, Dict, Any, Optional
import time
import re
import random
import os

from src.utils import safe_execute_code, compute_reward


class Agent(ABC):
    """Abstract base class for all RLM agents."""

    def __init__(self, name: str, max_steps: int = 10):
        self.name = name
        self.max_steps = max_steps
        self.transcript: List[Dict[str, Any]] = []

    @abstractmethod
    def run_episode(
        self, haystack: str, question: str, correct_answer: str
    ) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Run a single episode.

        Returns:
            (predicted_answer, transcript)
        """
        pass


# ---------------------------------------------------------------------------
# Week 4 – Deterministic baseline
# ---------------------------------------------------------------------------

class DeterministicAgent(Agent):
    """Deterministic agent using regex to find KEY=VALUE needles."""

    def __init__(self, name: str = "DeterministicAgent", max_steps: int = 1):
        super().__init__(name, max_steps)

    def run_episode(
        self, haystack: str, question: str, correct_answer: str
    ) -> Tuple[str, List[Dict[str, Any]]]:
        self.transcript = []
        predicted_answer = ""

        match = re.search(r"What is the value of (.*?)\?", question)
        if not match:
            self.transcript.append(
                {"step": 0, "action": "Parse Question", "status": "Failed",
                 "output": "Could not parse needle_key from question."}
            )
            return "N/A", self.transcript

        needle_key = match.group(1)
        regex_pattern = r"""%s=([a-zA-Z0-9]+)""" % re.escape(needle_key)

        generated_code = (
            f"import re\n"
            f"search_pattern = r'{regex_pattern}'\n"
            f"match = re.search(search_pattern, CONTEXT)\n"
            f"if match:\n"
            f"    print(match.group(1))\n"
            f"else:\n"
            f'    print("Needle not found")\n'
        )

        exec_result = safe_execute_code(generated_code, custom_globals={"CONTEXT": haystack})

        self.transcript.append({
            "step": 1,
            "action": "REPL Execution",
            "code": generated_code,
            "exec_result": {
                "ok": exec_result.ok,
                "stdout": exec_result.stdout,
                "stderr": exec_result.stderr,
                "runtime_sec": exec_result.runtime_sec,
            },
        })

        if exec_result.ok and exec_result.stdout:
            predicted_answer = exec_result.stdout.strip()
        elif not exec_result.ok:
            predicted_answer = f"Error: {exec_result.stderr.strip()}"
        else:
            predicted_answer = "No output from REPL"

        return predicted_answer, self.transcript


# ---------------------------------------------------------------------------
# Week 6 – Heuristic multi-step agent
# ---------------------------------------------------------------------------

class HeuristicMultiStepAgent(Agent):
    """
    Agent that selects a strategy based on the question type, potentially
    using multiple REPL steps.

    Supported question patterns:
    - Simple needle: "What is the value of KEY?" -> single regex
    - KV extraction: "find the first entry where ..." -> filter + extract
    - Aggregation: "Find all METRIC_* keys ... return their sum" -> findall + sum
    """

    def __init__(self, name: str = "HeuristicMultiStepAgent", max_steps: int = 5):
        super().__init__(name, max_steps)

    def _exec_step(
        self, step_num: int, code: str, context: str
    ) -> Tuple[Optional[str], Dict[str, Any]]:
        """Execute one REPL step and return (stdout, transcript_entry)."""
        exec_result = safe_execute_code(code, custom_globals={"CONTEXT": context})
        entry = {
            "step": step_num,
            "action": "REPL Execution",
            "code": code,
            "exec_result": {
                "ok": exec_result.ok,
                "stdout": exec_result.stdout,
                "stderr": exec_result.stderr,
                "runtime_sec": exec_result.runtime_sec,
            },
        }
        stdout = exec_result.stdout.strip() if exec_result.ok and exec_result.stdout else None
        return stdout, entry

    def run_episode(
        self, haystack: str, question: str, correct_answer: str
    ) -> Tuple[str, List[Dict[str, Any]]]:
        self.transcript = []
        q_lower = question.lower()

        # ---- Strategy selection ----
        if "find all metric" in q_lower and "sum" in q_lower:
            return self._strategy_aggregate(haystack, question)
        elif "find the first entry" in q_lower:
            return self._strategy_kv_filter(haystack, question)
        else:
            return self._strategy_simple_needle(haystack, question)

    # -- Strategy: simple needle (1 step) --
    def _strategy_simple_needle(
        self, haystack: str, question: str
    ) -> Tuple[str, List[Dict[str, Any]]]:
        match = re.search(r"What is the value of (.*?)\?", question)
        if not match:
            self.transcript.append({"step": 0, "action": "Parse Question", "status": "Failed"})
            return "N/A", self.transcript

        key = match.group(1)
        code = (
            f"import re\n"
            f"m = re.search(r'{re.escape(key)}=([a-zA-Z0-9]+)', CONTEXT)\n"
            f"print(m.group(1) if m else 'Needle not found')\n"
        )
        stdout, entry = self._exec_step(1, code, haystack)
        self.transcript.append(entry)
        return (stdout or "Needle not found"), self.transcript

    # -- Strategy: filter + extract (2 steps) --
    def _strategy_kv_filter(
        self, haystack: str, question: str
    ) -> Tuple[str, List[Dict[str, Any]]]:
        # Parse filter and target from question
        filter_match = re.search(r"where (\w+)=(\w+)", question)
        target_match = re.search(r"return its (\w+) value", question)

        if not filter_match or not target_match:
            self.transcript.append({"step": 0, "action": "Parse Question", "status": "Failed"})
            return "N/A", self.transcript

        filter_key = filter_match.group(1)
        filter_val = filter_match.group(2)
        target_field = target_match.group(1)

        # Step 1: Find matching lines
        code_step1 = (
            f"lines = CONTEXT.split('\\n')\n"
            f"matches = [l for l in lines if '{filter_key}={filter_val}' in l]\n"
            f"print(len(matches))\n"
            f"if matches:\n"
            f"    print(matches[0])\n"
        )
        stdout1, entry1 = self._exec_step(1, code_step1, haystack)
        self.transcript.append(entry1)

        if not stdout1 or stdout1 == "0":
            return "not_found", self.transcript

        # Extract the first matching line from stdout
        stdout_lines = stdout1.strip().split("\n")
        first_match_line = stdout_lines[1] if len(stdout_lines) > 1 else ""

        # Step 2: Extract target field from the matched line
        code_step2 = (
            f"import re\n"
            f"line = '''{first_match_line}'''\n"
            f"m = re.search(r'{target_field}=(\\w+)', line)\n"
            f"print(m.group(1) if m else 'not_found')\n"
        )
        stdout2, entry2 = self._exec_step(2, code_step2, haystack)
        self.transcript.append(entry2)
        return (stdout2 or "not_found"), self.transcript

    # -- Strategy: aggregate METRIC_* values (2 steps) --
    def _strategy_aggregate(
        self, haystack: str, question: str
    ) -> Tuple[str, List[Dict[str, Any]]]:
        # Step 1: Find all METRIC_X=<number> pairs
        code_step1 = (
            "import re\n"
            "pairs = re.findall(r'METRIC_[A-Z]=(\\d+)', CONTEXT)\n"
            "print(','.join(pairs))\n"
        )
        stdout1, entry1 = self._exec_step(1, code_step1, haystack)
        self.transcript.append(entry1)

        if not stdout1:
            return "0", self.transcript

        # Step 2: Sum the extracted values
        code_step2 = (
            f"values = [{stdout1}]\n"
            f"print(sum(values))\n"
        )
        stdout2, entry2 = self._exec_step(2, code_step2, haystack)
        self.transcript.append(entry2)
        return (stdout2 or "0"), self.transcript


# ---------------------------------------------------------------------------
# Week 8 – LLM-based agent via HuggingFace Inference API
# ---------------------------------------------------------------------------

# System prompt instructs the LLM how to generate REPL code
_LLM_SYSTEM_PROMPT = (
    "You are a Python code generator for a sandboxed REPL environment.\n"
    "You will be given a question about a text stored in the variable CONTEXT.\n"
    "Write Python code that processes CONTEXT and prints the answer.\n\n"
    "Rules:\n"
    "- The variable CONTEXT is already defined and contains the full text.\n"
    "- You may only import the `re` module. No other imports are allowed.\n"
    "- You MUST call print() with your final answer.\n"
    "- Output ONLY a Python code block. No explanation outside the code block.\n"
    "- Keep your code concise and correct.\n"
)


def extract_code_from_response(response_text: str) -> Optional[str]:
    """Extract Python code from an LLM response containing markdown code blocks."""
    # Try ```python ... ``` first
    match = re.search(r"```python\s*\n(.*?)```", response_text, re.DOTALL)
    if match:
        return match.group(1).strip()
    # Fall back to ``` ... ```
    match = re.search(r"```\s*\n(.*?)```", response_text, re.DOTALL)
    if match:
        return match.group(1).strip()
    # If the whole response looks like code (no markdown), use it directly
    lines = response_text.strip().splitlines()
    if lines and not any(line.startswith("#") and "```" in line for line in lines):
        # Check if it looks like Python code
        code_indicators = ("import ", "print(", "re.", "CONTEXT", "=", "for ", "if ")
        if any(response_text.strip().startswith(ind) for ind in code_indicators):
            return response_text.strip()
    return None


class LLMAgent(Agent):
    """
    LLM-based agent that calls a HuggingFace model to generate Python code
    for REPL execution.

    Uses huggingface_hub.InferenceClient with chat_completion. The client
    is lazy-initialized on first use and requires the HF_TOKEN environment
    variable to be set.
    """

    def __init__(
        self,
        name: str = "LLMAgent",
        max_steps: int = 5,
        model_id: str = "Qwen/Qwen2.5-Coder-7B-Instruct",
        max_tokens: int = 512,
        temperature: float = 0.2,
    ):
        super().__init__(name, max_steps)
        self.model_id = model_id
        self.max_tokens = max_tokens
        self.temperature = temperature
        self._client = None

    def _get_client(self):
        """Lazy-initialize the HuggingFace InferenceClient."""
        if self._client is None:
            from huggingface_hub import InferenceClient
            token = os.environ.get("HF_TOKEN")
            if not token:
                raise EnvironmentError(
                    "HF_TOKEN environment variable is required for LLMAgent. "
                    "Get a free token at https://huggingface.co/settings/tokens"
                )
            self._client = InferenceClient(model=self.model_id, token=token)
        return self._client

    def _call_llm(self, messages: List[Dict[str, str]]) -> str:
        """Call the LLM with exponential backoff on rate limits."""
        client = self._get_client()
        backoff_times = [5, 10, 20]
        for attempt in range(len(backoff_times) + 1):
            try:
                response = client.chat_completion(
                    messages=messages,
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                )
                return response.choices[0].message.content
            except Exception as e:
                error_str = str(e)
                if "429" in error_str or "rate" in error_str.lower():
                    if attempt < len(backoff_times):
                        wait = backoff_times[attempt]
                        time.sleep(wait)
                        continue
                raise
        return ""

    def run_episode(
        self, haystack: str, question: str, correct_answer: str
    ) -> Tuple[str, List[Dict[str, Any]]]:
        self.transcript = []
        predicted_answer = ""

        # Build context preview for the prompt (first 500 chars)
        context_preview = haystack[:500]
        if len(haystack) > 500:
            context_preview += f"\n... ({len(haystack)} total characters)"

        # Initialize conversation with system prompt and user request
        messages = [
            {"role": "system", "content": _LLM_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"Context preview (the full text is available as the variable CONTEXT):\n"
                    f"---\n{context_preview}\n---\n\n"
                    f"Question: {question}\n\n"
                    f"Write Python code to answer this question. "
                    f"The full context is in the variable CONTEXT. "
                    f"Print only the final answer."
                ),
            },
        ]

        for step in range(1, self.max_steps + 1):
            # Step A: Call LLM
            try:
                llm_response = self._call_llm(messages)
            except Exception as e:
                self.transcript.append({
                    "step": step,
                    "action": "LLM Call Failed",
                    "error": str(e),
                })
                predicted_answer = f"LLM Error: {e}"
                break

            # Step B: Extract code
            code = extract_code_from_response(llm_response)
            if code is None:
                self.transcript.append({
                    "step": step,
                    "action": "Code Extraction Failed",
                    "llm_response": llm_response[:500],
                })
                # Ask LLM to provide proper code block
                messages.append({"role": "assistant", "content": llm_response})
                messages.append({
                    "role": "user",
                    "content": (
                        "I could not extract Python code from your response. "
                        "Please provide your code inside a ```python code block. "
                        "Remember to print() the final answer."
                    ),
                })
                continue

            # Step C: Execute code in sandbox
            exec_result = safe_execute_code(code, custom_globals={"CONTEXT": haystack})

            self.transcript.append({
                "step": step,
                "action": "REPL Execution",
                "code": code,
                "llm_response": llm_response[:500],
                "exec_result": {
                    "ok": exec_result.ok,
                    "stdout": exec_result.stdout,
                    "stderr": exec_result.stderr,
                    "runtime_sec": exec_result.runtime_sec,
                },
            })

            # Step D: Check result
            if exec_result.ok and exec_result.stdout and exec_result.stdout.strip():
                # Take last non-empty line as the answer
                output_lines = exec_result.stdout.strip().splitlines()
                predicted_answer = output_lines[-1].strip()
                break
            elif not exec_result.ok:
                # Error: send it back to LLM for retry
                messages.append({"role": "assistant", "content": llm_response})
                messages.append({
                    "role": "user",
                    "content": (
                        f"The code produced an error:\n{exec_result.stderr}\n\n"
                        f"Please fix the code and try again. "
                        f"Remember: only `re` can be imported, "
                        f"CONTEXT holds the full text, and you must print() the answer."
                    ),
                })
            else:
                # No output: ask LLM to add print()
                messages.append({"role": "assistant", "content": llm_response})
                messages.append({
                    "role": "user",
                    "content": (
                        "The code ran successfully but produced no output. "
                        "Make sure to call print() with the final answer."
                    ),
                })

        if not predicted_answer:
            predicted_answer = "No answer produced"

        return predicted_answer, self.transcript


class LocalLLMAgent(LLMAgent):
    """
    Agent using a locally-loaded transformers model (e.g., fine-tuned with LoRA).

    Inherits the multi-step retry logic from LLMAgent but replaces
    API calls with local model.generate().
    """

    def __init__(
        self,
        model,
        tokenizer,
        name: str = "LocalLLMAgent",
        max_steps: int = 5,
        max_new_tokens: int = 512,
        temperature: float = 0.2,
    ):
        # Call Agent.__init__ directly (skip LLMAgent's API setup)
        Agent.__init__(self, name, max_steps)
        self.model = model
        self.tokenizer = tokenizer
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature

    def _call_llm(self, messages: List[Dict[str, str]]) -> str:
        """Generate a response using the local model."""
        import torch

        input_text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(
            input_text, return_tensors="pt", truncation=True, max_length=2048
        ).to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=self.temperature > 0,
                temperature=max(self.temperature, 0.01),
                top_p=0.9,
                pad_token_id=self.tokenizer.pad_token_id,
            )

        generated = outputs[0, inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(generated, skip_special_tokens=True)


# ---------------------------------------------------------------------------
# Evaluation framework
# ---------------------------------------------------------------------------

class EvaluationFramework:
    """
    Orchestrates evaluation of RLM agents and collects metrics.

    Computes reward per episode:
        R = C - lambda_s * (steps / max_steps) - lambda_t * tokens
    """

    def __init__(
        self,
        agents: List[Agent],
        task_generator: callable,
        num_episodes: int = 10,
        step_penalty: float = 0.05,
    ):
        self.agents = agents
        self.task_generator = task_generator
        self.num_episodes = num_episodes
        self.step_penalty = step_penalty
        self.results: List[Dict[str, Any]] = []

    def run_evaluation(self) -> List[Dict[str, Any]]:
        """Run evaluation episodes for all agents and collect metrics."""
        print(f"\n--- Starting Evaluation for {self.num_episodes} Episodes ---")
        for episode_idx in range(self.num_episodes):
            print(f"\n--- Episode {episode_idx + 1}/{self.num_episodes} ---")

            haystack, question, correct_answer = self.task_generator()

            for agent in self.agents:
                start_time = time.time()
                predicted_answer, transcript = agent.run_episode(
                    haystack, question, correct_answer
                )
                runtime = time.time() - start_time

                is_correct = predicted_answer == correct_answer
                reward = compute_reward(
                    is_correct=is_correct,
                    num_steps=len(transcript),
                    max_steps=agent.max_steps,
                    step_penalty=self.step_penalty,
                )

                self.results.append({
                    "episode_idx": episode_idx,
                    "agent_name": agent.name,
                    "question": question,
                    "correct_answer": correct_answer,
                    "predicted_answer": predicted_answer,
                    "is_correct": is_correct,
                    "reward": reward,
                    "runtime_sec": runtime,
                    "repl_steps": len(transcript),
                    "transcript": transcript,
                })
                status = "correct" if is_correct else "WRONG"
                print(
                    f"  {agent.name}: {status} | "
                    f"reward={reward:.3f} | steps={len(transcript)} | "
                    f"time={runtime:.4f}s"
                )

        print("\n--- Evaluation Completed ---")
        return self.results

    def display_results(self):
        """Display aggregated results per agent."""
        if not self.results:
            print("No results. Run run_evaluation() first.")
            return

        print("\n--- Aggregated Results ---")
        agent_metrics: Dict[str, Dict] = {}
        for res in self.results:
            name = res["agent_name"]
            if name not in agent_metrics:
                agent_metrics[name] = {
                    "correct": 0, "total_runtime": 0.0,
                    "total_steps": 0, "total_reward": 0.0, "count": 0,
                }
            m = agent_metrics[name]
            m["correct"] += int(res["is_correct"])
            m["total_runtime"] += res["runtime_sec"]
            m["total_steps"] += res["repl_steps"]
            m["total_reward"] += res["reward"]
            m["count"] += 1

        for name, m in agent_metrics.items():
            n = m["count"]
            print(f"\nAgent: {name}")
            print(f"  Accuracy:    {m['correct'] / n * 100:.1f}%")
            print(f"  Avg Reward:  {m['total_reward'] / n:.3f}")
            print(f"  Avg Steps:   {m['total_steps'] / n:.2f}")
            print(f"  Avg Runtime: {m['total_runtime'] / n:.4f}s")

        # Show a sample transcript
        if self.results:
            r = self.results[0]
            print(f"\n--- Sample Transcript (Episode 1, {r['agent_name']}) ---")
            print(f"  Q: {r['question']}")
            print(f"  Expected: {r['correct_answer']}")
            print(f"  Predicted: {r['predicted_answer']}")
            for entry in r["transcript"]:
                print(f"  Step {entry['step']}: {entry['action']}")
                if "code" in entry:
                    first_line = entry["code"].strip().splitlines()[0]
                    print(f"    Code: {first_line}...")
                if "exec_result" in entry:
                    er = entry["exec_result"]
                    print(f"    OK={er['ok']}  time={er['runtime_sec']:.4f}s")
                    if er["stdout"]:
                        print(f"    stdout: {er['stdout'].strip()[:80]}")


# ---------------------------------------------------------------------------
# Week 7 – Training loop skeleton
# ---------------------------------------------------------------------------

class Trajectory:
    """A single episode trajectory for training."""

    def __init__(
        self,
        context: str,
        question: str,
        correct_answer: str,
        predicted_answer: str,
        actions: List[str],
        reward: float,
    ):
        self.context = context
        self.question = question
        self.correct_answer = correct_answer
        self.predicted_answer = predicted_answer
        self.actions = actions
        self.reward = reward

    @property
    def is_success(self) -> bool:
        return self.predicted_answer == self.correct_answer


class TrainingLoop:
    """
    Skeleton for reward-weighted imitation learning.

    Phase 1 (current): Collect trajectories and filter successful ones.
    Phase 2 (future): Use successful trajectories for behavior cloning.
    Phase 3 (future): Apply reward-weighted policy gradient updates.
    """

    def __init__(
        self,
        agent: Agent,
        task_generator: callable,
        batch_size: int = 16,
        success_threshold: float = 0.5,
    ):
        self.agent = agent
        self.task_generator = task_generator
        self.batch_size = batch_size
        self.success_threshold = success_threshold
        self.trajectory_buffer: List[Trajectory] = []
        self.training_history: List[Dict[str, float]] = []

    def collect_batch(self) -> List[Trajectory]:
        """Collect a batch of trajectories by running the agent."""
        batch: List[Trajectory] = []
        for _ in range(self.batch_size):
            context, question, correct_answer = self.task_generator()
            predicted, transcript = self.agent.run_episode(context, question, correct_answer)

            actions = [
                entry.get("code", "") for entry in transcript if "code" in entry
            ]
            reward = compute_reward(
                is_correct=(predicted == correct_answer),
                num_steps=len(transcript),
                max_steps=self.agent.max_steps,
            )
            traj = Trajectory(
                context=context,
                question=question,
                correct_answer=correct_answer,
                predicted_answer=predicted,
                actions=actions,
                reward=reward,
            )
            batch.append(traj)

        self.trajectory_buffer.extend(batch)
        return batch

    def filter_successes(self) -> List[Trajectory]:
        """Return trajectories with reward above threshold (for imitation learning)."""
        return [t for t in self.trajectory_buffer if t.reward >= self.success_threshold]

    def train_step(self) -> Dict[str, float]:
        """
        One training iteration:
        1. Collect a batch
        2. Filter successful trajectories
        3. Log statistics (actual weight updates are future work)
        """
        batch = self.collect_batch()
        successes = self.filter_successes()

        total_reward = sum(t.reward for t in batch)
        accuracy = sum(1 for t in batch if t.is_success) / len(batch)
        avg_steps = sum(len(t.actions) for t in batch) / len(batch)

        stats = {
            "batch_size": len(batch),
            "num_successes": len(successes),
            "accuracy": accuracy,
            "avg_reward": total_reward / len(batch),
            "avg_steps": avg_steps,
            "buffer_size": len(self.trajectory_buffer),
        }
        self.training_history.append(stats)
        return stats

    def run(self, num_iterations: int = 5) -> List[Dict[str, float]]:
        """Run multiple training iterations and return history."""
        print(f"\n--- Training Loop: {num_iterations} iterations, batch={self.batch_size} ---")
        for i in range(num_iterations):
            stats = self.train_step()
            print(
                f"  Iter {i+1}/{num_iterations}: "
                f"acc={stats['accuracy']:.1%} | "
                f"reward={stats['avg_reward']:.3f} | "
                f"steps={stats['avg_steps']:.1f} | "
                f"buffer={stats['buffer_size']}"
            )
        print("--- Training Complete ---")
        return self.training_history
