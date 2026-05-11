"""
Poker-specific training pipeline.

Adapts the generic training infrastructure (src/training.py) for poker:
- Poker prompt formatting (full context, poker system prompt)
- Poker trajectory collection with reasoning traces
- PokerBCTrainer: behavior cloning on heuristic trajectories
- PokerReinforceTrainer: RL with EV-based rewards
"""

from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple, Callable
import csv
import json
import os
import random
import re

_torch_available = False
try:
    import torch
    import torch.nn.functional as F
    _torch_available = True
except ImportError:
    pass

from src.models import Agent
from src.utils import safe_execute_code
from src.poker.environment import GameState
from src.poker.tasks import (
    generate_poker_task,
    generate_poker_task_with_trace,
    generate_preflop_task,
    generate_postflop_task,
    POKER_SYSTEM_PROMPT,
)
from src.poker.rewards import compute_poker_reward, compute_poker_reward_simple, parse_action
from src.poker.heuristic import HeuristicPokerBot


# ── Prompt formatting ─────────────────────────────────────────────────

def format_poker_prompt(question: str, context: str) -> List[Dict[str, str]]:
    """Format poker task as chat messages. Sends full context (history matters)."""
    return [
        {"role": "system", "content": POKER_SYSTEM_PROMPT},
        {"role": "user", "content": f"{context}\n\n{question}"},
    ]


# ── Trajectory data ───────────────────────────────────────────────────

@dataclass
class PokerTrajectory:
    """A poker trajectory with all info needed for training."""
    context: str
    question: str
    correct_answer: str
    predicted_answer: str
    code: str
    reward: float
    is_correct: bool
    num_steps: int
    messages: List[Dict[str, str]]
    reasoning_trace: str = ""
    action_type: str = ""
    action_amount: float = 0.0
    parsed_stats: bool = False


# ── Trajectory collection ─────────────────────────────────────────────

def collect_poker_trajectories(
    agent: Agent,
    num_episodes: int = 500,
    task_generators: Optional[List[Callable]] = None,
) -> List[PokerTrajectory]:
    """
    Collect poker trajectories using any agent.

    Uses generate_poker_task by default, returns PokerTrajectory objects
    compatible with the training pipeline.
    """
    if task_generators is None:
        task_generators = [generate_poker_task]

    trajectories = []

    for i in range(num_episodes):
        gen = random.choice(task_generators)
        context, question, correct_answer = gen()

        predicted, transcript = agent.run_episode(context, question, correct_answer)

        # Check action type match
        pred_type, pred_amt = parse_action(predicted)
        corr_type, corr_amt = parse_action(correct_answer)
        is_correct = (pred_type == corr_type)
        reward = compute_poker_reward_simple(predicted, correct_answer)

        # Extract the last code from transcript
        codes = [e.get("code", "") for e in transcript if "code" in e]
        code = codes[-1] if codes else ""

        # Check if code parsed opponent stats
        all_stdout = " ".join(
            e.get("exec_result", {}).get("stdout", "")
            for e in transcript
        )
        parsed_stats = "VPIP" in all_stdout or "vpip" in all_stdout

        messages = format_poker_prompt(question, context)

        trajectories.append(PokerTrajectory(
            context=context,
            question=question,
            correct_answer=correct_answer,
            predicted_answer=predicted,
            code=code,
            reward=reward,
            is_correct=is_correct,
            num_steps=len(transcript),
            messages=messages,
            action_type=pred_type,
            action_amount=pred_amt,
            parsed_stats=parsed_stats,
        ))

        if (i + 1) % 100 == 0:
            successes = sum(1 for t in trajectories if t.is_correct)
            print(f"  Collected {i+1}/{num_episodes} trajectories ({successes} correct)")

    return trajectories


def collect_poker_trajectories_with_traces(
    num_episodes: int = 500,
) -> List[PokerTrajectory]:
    """
    Collect trajectories directly from the heuristic bot with reasoning traces.

    This is the most efficient way to generate BC training data — it
    bypasses the Agent interface and directly captures the heuristic's
    3-step reasoning as the code target.
    """
    bot = HeuristicPokerBot()
    trajectories = []

    for i in range(num_episodes):
        context, question, answer, trace = generate_poker_task_with_trace()

        messages = format_poker_prompt(question, context)

        # The "code" for BC is a combined script that does retrieve + compute + decide
        code = _trace_to_code(context, answer)

        trajectories.append(PokerTrajectory(
            context=context,
            question=question,
            correct_answer=answer,
            predicted_answer=answer,
            code=code,
            reward=1.0,
            is_correct=True,
            num_steps=3,
            messages=messages,
            reasoning_trace=trace,
            action_type=parse_action(answer)[0],
            action_amount=parse_action(answer)[1],
            parsed_stats=True,
        ))

        if (i + 1) % 100 == 0:
            print(f"  Collected {i+1}/{num_episodes} trajectories with traces")

    return trajectories


def _trace_to_code(context: str, answer: str) -> str:
    """Convert a heuristic reasoning trace into a single executable code block."""
    return f'''import re

# STEP 1: RETRIEVE - Parse opponent stats from hand history
lines = CONTEXT.split('\\n')
history_start = -1
for i, line in enumerate(lines):
    if '=== PREVIOUS HANDS' in line:
        history_start = i
        break

stats = {{}}
if history_start >= 0:
    first_raiser = None
    in_preflop = False
    in_postflop = False
    for line in lines[history_start:]:
        line = line.strip()
        if line.startswith('Hand #'):
            first_raiser = None
            in_preflop = False
            in_postflop = False
        if 'Preflop:' in line:
            in_preflop = True
            in_postflop = False
        elif any(s in line for s in ['Flop', 'Turn', 'River']):
            in_preflop = False
            in_postflop = True
        for m in re.finditer(r'(\\w+)\\s+(raises|calls|bets|folds|checks)', line):
            pos, action = m.group(1), m.group(2)
            if pos not in stats:
                stats[pos] = {{'hands':0,'vpip':0,'pfr':0,'bets':0,'calls':0}}
            if in_preflop and action in ('raises','calls','bets'):
                stats[pos]['vpip'] += 1
            if in_preflop and action == 'raises':
                stats[pos]['pfr'] += 1
            if in_postflop and action in ('raises','bets'):
                stats[pos]['bets'] += 1
            if in_postflop and action == 'calls':
                stats[pos]['calls'] += 1
        if line.startswith('Result:'):
            for pos in stats:
                stats[pos]['hands'] += 1

# STEP 2: COMPUTE - Parse hand and evaluate
hand_match = re.search(r'Your Hand:\\s*(\\w+)\\s+(\\w+)', CONTEXT)
pot_match = re.search(r'Pot:\\s*\\$(\\d+)', CONTEXT)
call_match = re.search(r'To Call:\\s*\\$(\\d+)', CONTEXT)
pos_match = re.search(r'Your Position:\\s*(\\w+)', CONTEXT)
board_match = re.search(r'Community Cards:\\s*(.+?)\\s*\\(', CONTEXT)

pot = int(pot_match.group(1)) if pot_match else 0
to_call = int(call_match.group(1)) if call_match else 0
pot_odds = to_call / (pot + to_call) if (pot + to_call) > 0 and to_call > 0 else 0

rank_values = {{'2':2,'3':3,'4':4,'5':5,'6':6,'7':7,'8':8,'9':9,'T':10,'J':11,'Q':12,'K':13,'A':14}}
hero_cards = []
if hand_match:
    for c in [hand_match.group(1), hand_match.group(2)]:
        if c[0] in rank_values:
            hero_cards.append(rank_values[c[0]])

board_cards = []
if board_match:
    for c in board_match.group(1).strip().split():
        if c and c[0] in rank_values:
            board_cards.append(rank_values[c[0]])

all_ranks = hero_cards + board_cards
rank_counts = {{}}
for r in all_ranks:
    rank_counts[r] = rank_counts.get(r, 0) + 1
has_pair = any(v >= 2 for v in rank_counts.values())

# STEP 3: DECIDE - Combine analysis with opponent profile
# Check villain tendencies
betting_lines = [l for l in lines if 'bets' in l or 'raises' in l]
villain_pos = None
for line in reversed(betting_lines):
    m = re.search(r'(\\w+)\\s+(bets|raises)', line)
    if m:
        villain_pos = m.group(1)
        break

adjustment = "none"
if villain_pos and villain_pos in stats:
    s = stats[villain_pos]
    h = max(s['hands'], 1)
    agg = s['bets'] / max(s['calls'], 1)
    vpip_pct = s['vpip'] / h
    if agg > 2.0:
        adjustment = "aggressive"
    elif agg < 1.0 and to_call > 0:
        adjustment = "passive_betting"
    if vpip_pct > 0.4:
        adjustment += "_loose"

print("{answer}")
'''


# ── Poker Behavior Cloning ────────────────────────────────────────────

def format_poker_sft_example(
    messages: List[Dict[str, str]], code: str, tokenizer
) -> str:
    """Format a (prompt, code) pair as a chat string for SFT."""
    full_messages = messages + [
        {"role": "assistant", "content": f"```python\n{code}\n```"},
    ]
    return tokenizer.apply_chat_template(full_messages, tokenize=False)


class PokerBCTrainer:
    """
    Behavior cloning on heuristic poker trajectories.

    Fine-tunes a small LLM (Qwen 1.5B + LoRA) to replicate the
    heuristic's 3-step retrieve → compute → decide reasoning.
    """

    def __init__(
        self,
        model,
        tokenizer,
        output_dir: str = "./checkpoints/poker_bc",
        num_epochs: int = 3,
        batch_size: int = 4,
        learning_rate: float = 2e-4,
        max_length: int = 4096,
        gradient_accumulation_steps: int = 2,
        weight_decay: float = 0.0,
        seed: Optional[int] = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.output_dir = output_dir
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.max_length = max_length
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.weight_decay = weight_decay
        self.seed = seed

    def train(self, trajectories: List[PokerTrajectory]) -> Dict[str, Any]:
        """Run behavior cloning on successful trajectories."""
        from trl import SFTConfig, SFTTrainer
        from datasets import Dataset as HFDataset

        # Filter to correct trajectories with code
        successful = [t for t in trajectories if t.is_correct and t.code]
        print(f"Training on {len(successful)}/{len(trajectories)} successful trajectories")

        if not successful:
            raise ValueError("No successful trajectories to train on")

        texts = []
        for traj in successful:
            text = format_poker_sft_example(traj.messages, traj.code, self.tokenizer)
            texts.append(text)

        dataset = HFDataset.from_dict({"text": texts})

        # Detect GPU capability: use bf16 on Ampere+, fp16 on older (T4, etc.)
        use_bf16 = False
        use_fp16 = False
        if _torch_available and torch.cuda.is_available():
            cap = torch.cuda.get_device_capability()
            if cap[0] >= 8:  # Ampere (A100, A10G, etc.)
                use_bf16 = True
            else:  # Turing (T4), Volta (V100), etc.
                use_fp16 = True

        seed_kwargs: Dict[str, Any] = {}
        if self.seed is not None:
            seed_kwargs["seed"] = self.seed
            seed_kwargs["data_seed"] = self.seed

        training_args = SFTConfig(
            output_dir=self.output_dir,
            num_train_epochs=self.num_epochs,
            per_device_train_batch_size=self.batch_size,
            learning_rate=self.learning_rate,
            max_length=self.max_length,
            logging_steps=10,
            save_steps=100,
            save_total_limit=2,
            bf16=use_bf16,
            fp16=use_fp16,
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            warmup_ratio=0.1,
            lr_scheduler_type="cosine",
            dataset_text_field="text",
            gradient_checkpointing=True,
            optim="adamw_torch",
            weight_decay=self.weight_decay,
            **seed_kwargs,
        )

        trainer = SFTTrainer(
            model=self.model,
            args=training_args,
            train_dataset=dataset,
            processing_class=self.tokenizer,
        )

        result = trainer.train()
        trainer.save_model(self.output_dir)
        self.tokenizer.save_pretrained(self.output_dir)

        return {
            "train_loss": result.training_loss,
            "num_examples": len(successful),
            "epochs": self.num_epochs,
            "output_dir": self.output_dir,
        }


# ── Poker REINFORCE ───────────────────────────────────────────────────

class PokerReinforceTrainer:
    """
    REINFORCE with EV-based rewards for poker.

    Each iteration:
    1. Generate poker tasks
    2. Model generates code, execute in sandbox, parse action
    3. Compute poker reward (action quality + EV + reasoning)
    4. Policy gradient update
    """

    def __init__(
        self,
        model,
        tokenizer,
        output_dir: str = "./checkpoints/poker_rl",
        batch_size: int = 8,
        learning_rate: float = 5e-6,
        max_new_tokens: int = 1024,
        max_length: int = 2048,
        baseline_ema: float = 0.9,
        max_grad_norm: float = 1.0,
        advantage_clip: float = 2.0,
        task_generator: Callable = generate_poker_task,
        ema_gamma: float = 0.9,
        sample_temperature: float = 0.15,
        sample_top_p: float = 0.9,
        action_space: str = "simple",
        eval_every: int = 5,
        eval_episodes: int = 40,
        eval_seed: int = 1234,
    ):
        if not _torch_available:
            raise ImportError("PyTorch required for RL training")

        self.model = model
        self.tokenizer = tokenizer
        self.output_dir = output_dir
        self.batch_size = batch_size
        self.max_new_tokens = max_new_tokens
        self.max_length = max_length
        self.baseline_ema = baseline_ema
        self.max_grad_norm = max_grad_norm
        self.advantage_clip = advantage_clip
        self.task_generator = task_generator
        self.ema_gamma = ema_gamma
        self.sample_temperature = sample_temperature
        self.sample_top_p = sample_top_p
        self.action_space = action_space
        self.eval_every = eval_every
        self.eval_episodes = max(0, eval_episodes)
        self.eval_seed = eval_seed

        trainable_params = [p for p in model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(trainable_params, lr=learning_rate)

        self.reward_baseline = 0.0
        self.reward_ema = None  # EMA of reward: avg_{i+1} = (1-gamma)*r_i + gamma*avg_i
        self.accuracy_ema = None
        self.history: List[Dict[str, float]] = []
        self.eval_history: List[Dict[str, float]] = []
        self.best_eval_accuracy: float = -1.0
        self.best_eval_reward: float = -1.0
        self.best_eval_iteration: int = 0
        self.eval_tasks: List[Tuple[str, str, str]] = self._build_eval_tasks(
            self.eval_episodes, self.eval_seed
        )

    def _generate_code(
        self, messages: List[Dict[str, str]]
    ) -> Tuple[str, "torch.Tensor"]:
        """Generate code from prompt. Returns (response_text, generated_token_ids)."""
        from src.models import extract_code_from_response

        with torch.no_grad():
            input_text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = self.tokenizer(
                input_text, return_tensors="pt", truncation=True, max_length=self.max_length
            ).to(self.model.device)
            input_length = inputs["input_ids"].shape[1]

            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=True,
                temperature=self.sample_temperature,
                top_p=self.sample_top_p,
                pad_token_id=self.tokenizer.pad_token_id,
            )

            generated_ids = outputs[0, input_length:]
            generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)

        return generated_text, generated_ids

    @staticmethod
    def _action_to_text(action_type: str, amount: float) -> str:
        """Convert parsed action tuple into canonical output text."""
        if action_type in ("fold", "check"):
            return action_type
        if action_type in ("call", "raise"):
            if amount > 0:
                return f"{action_type} ${int(round(amount))}"
            return action_type
        return "fold"

    def _canonicalize_action_text(self, text: str) -> str:
        """Normalize an action string according to selected action space."""
        action_type, amount = parse_action(text)
        if self.action_space == "simple":
            return action_type
        return self._action_to_text(action_type, amount)

    @staticmethod
    def _extract_code_fallback(response_text: str) -> Optional[str]:
        """
        Fallback code extraction when markdown fences are missing.

        Heuristic: if response contains python-like lines, slice from first
        likely code line through the end.
        """
        from src.models import extract_code_from_response

        code = extract_code_from_response(response_text)
        if code is not None:
            return code

        lines = response_text.splitlines()
        if not lines:
            return None

        start_idx = None
        for i, line in enumerate(lines):
            stripped = line.strip()
            if (
                stripped.startswith("import ")
                or stripped.startswith("print(")
                or "CONTEXT" in stripped
                or stripped.startswith("for ")
                or stripped.startswith("if ")
                or stripped.startswith("lines =")
            ):
                start_idx = i
                break

        if start_idx is None:
            return None

        candidate = "\n".join(lines[start_idx:]).strip()
        if "print(" in candidate:
            return candidate
        return None

    def _build_eval_tasks(self, num_tasks: int, seed: int) -> List[Tuple[str, str, str]]:
        """Create deterministic held-out tasks without perturbing global RNG."""
        if num_tasks <= 0:
            return []
        old_state = random.getstate()
        random.seed(seed)
        tasks = [self.task_generator() for _ in range(num_tasks)]
        random.setstate(old_state)
        return tasks

    @staticmethod
    def _looks_like_action_text(text: str) -> bool:
        """Quick check for action-like text in model output."""
        return bool(re.search(r"\b(fold|check|call|raise)\b", text.lower()))

    def _predict_action_for_task(self, context: str, question: str) -> str:
        """Generate a single action prediction for held-out eval task."""
        base_messages = format_poker_prompt(question, context)
        if self.action_space == "simple":
            action_format = "(fold/check/call/raise)"
        else:
            action_format = "(fold/check/call $X/raise $X)"
        base_messages[-1]["content"] += (
            "\n\nImportant output format:\n"
            "- Return ONLY executable Python code.\n"
            "- No markdown fences.\n"
            "- Last printed line must be exactly one poker action "
            f"{action_format}.\n"
        )

        messages = base_messages
        response_text = ""
        predicted = ""
        for attempt_idx in range(2):
            response_text, _ = self._generate_code(messages)
            code = self._extract_code_fallback(response_text)
            if code is not None:
                exec_result = safe_execute_code(code, custom_globals={"CONTEXT": context})
                if exec_result.ok and exec_result.stdout and exec_result.stdout.strip():
                    predicted = exec_result.stdout.strip().splitlines()[-1].strip()
                    break
            if attempt_idx < 1:
                messages = messages + [{
                    "role": "assistant",
                    "content": response_text,
                }, {
                    "role": "user",
                    "content": (
                        "Return only runnable Python code and print exactly one final action "
                        f"{action_format}."
                    ),
                }]

        if not predicted:
            pred_type, pred_amt = parse_action(response_text)
            predicted = self._action_to_text(pred_type, pred_amt)
        return self._canonicalize_action_text(predicted)

    def _evaluate_policy(self) -> Dict[str, float]:
        """Evaluate current policy on fixed held-out tasks."""
        if not self.eval_tasks:
            return {"eval_accuracy": 0.0, "eval_avg_reward": 0.0, "eval_episodes": 0}
        self.model.eval()
        correct = 0
        total_reward = 0.0
        for context, question, correct_action_raw in self.eval_tasks:
            pred = self._predict_action_for_task(context, question)
            correct_action = self._canonicalize_action_text(correct_action_raw)
            reward = compute_poker_reward_simple(pred, correct_action)
            total_reward += reward
            if parse_action(pred)[0] == parse_action(correct_action)[0]:
                correct += 1
        n = len(self.eval_tasks)
        return {
            "eval_accuracy": correct / n,
            "eval_avg_reward": total_reward / n,
            "eval_episodes": n,
        }

    def _compute_log_probs(
        self, messages: List[Dict[str, str]], generated_ids: "torch.Tensor"
    ) -> "torch.Tensor":
        """Compute log probability of generated tokens under current policy."""
        input_text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(
            input_text, return_tensors="pt", truncation=True, max_length=self.max_length
        ).to(self.model.device)
        prompt_length = inputs["input_ids"].shape[1]

        full_ids = torch.cat(
            [inputs["input_ids"], generated_ids.unsqueeze(0)], dim=1
        )

        outputs = self.model(input_ids=full_ids)
        logits = outputs.logits

        gen_logits = logits[:, prompt_length - 1:-1, :]
        log_probs = F.log_softmax(gen_logits, dim=-1)

        token_log_probs = log_probs.gather(
            -1, generated_ids.unsqueeze(0).unsqueeze(-1)
        ).squeeze(-1)

        return token_log_probs.sum()

    def train_step(self) -> Dict[str, float]:
        """One REINFORCE iteration."""
        self.model.train()
        trajectories = []
        rollout_data = []
        attempted = 0
        extracted = 0
        exec_ok_count = 0
        nonzero_reward_count = 0
        no_code_count = 0
        exec_error_count = 0
        stdout_empty_count = 0
        fallback_used_count = 0
        parse_failed_count = 0
        wrapped_action_code_count = 0

        for _ in range(self.batch_size):
            attempted += 1
            context, question, correct_answer = self.task_generator()
            base_messages = format_poker_prompt(question, context)
            if self.action_space == "simple":
                action_format = "(fold/check/call/raise)"
            else:
                action_format = "(fold/check/call $X/raise $X)"
            base_messages[-1]["content"] += (
                "\n\nImportant output format:\n"
                "- Return ONLY executable Python code.\n"
                "- No markdown fences.\n"
                "- Last printed line must be exactly one poker action "
                f"{action_format}.\n"
            )

            messages = base_messages
            response_text = ""
            generated_ids = None
            code = None
            predicted = ""
            parsed_stats = False

            # Retry generation to recover executable code before fallback.
            for attempt_idx in range(3):
                response_text, generated_ids = self._generate_code(messages)
                code = self._extract_code_fallback(response_text)

                if code is None:
                    # If final retry still has no code, convert response into
                    # canonical action and execute via minimal Python.
                    if attempt_idx == 2:
                        wrapped_action_code_count += 1
                        raw_type, raw_amt = parse_action(response_text)
                        action_text = self._canonicalize_action_text(
                            self._action_to_text(raw_type, raw_amt)
                        )
                        code = f'print("{action_text}")'
                    else:
                        no_code_count += 1
                        if attempt_idx < 2:
                            messages = messages + [{
                                "role": "assistant",
                                "content": response_text,
                            }, {
                                "role": "user",
                                "content": (
                                    "Your previous response was not executable Python code.\n"
                                    "Return only runnable Python now, ending with exactly one "
                                    f"print action line {action_format}."
                                ),
                            }]
                        continue

                if code is None:
                    if attempt_idx < 2:
                        messages = messages + [{
                            "role": "assistant",
                            "content": response_text,
                        }, {
                            "role": "user",
                            "content": (
                                "Your previous response was not executable Python code.\n"
                                "Return only runnable Python now, ending with exactly one "
                                f"print action line {action_format}."
                            ),
                        }]
                    continue

                extracted += 1
                exec_result = safe_execute_code(code, custom_globals={"CONTEXT": context})

                if not exec_result.ok:
                    exec_error_count += 1
                    if attempt_idx < 2:
                        err = (exec_result.stderr or "execution failed").strip()[:300]
                        messages = messages + [{
                            "role": "assistant",
                            "content": response_text,
                        }, {
                            "role": "user",
                            "content": (
                                "Your code failed to execute. Error:\n"
                                f"{err}\n"
                                "Fix and return only runnable Python ending with one action print."
                            ),
                        }]
                    continue

                exec_ok_count += 1
                if exec_result.stdout and exec_result.stdout.strip():
                    predicted = exec_result.stdout.strip().splitlines()[-1].strip()
                    predicted = self._canonicalize_action_text(predicted)
                    parsed_stats = "vpip" in exec_result.stdout.lower()
                    break

                stdout_empty_count += 1
                if attempt_idx < 2:
                    messages = messages + [{
                        "role": "assistant",
                        "content": response_text,
                    }, {
                        "role": "user",
                        "content": (
                            "Your code ran but did not print an action.\n"
                            "Return only runnable Python and ensure the final printed line is "
                            f"exactly one action: {action_format}."
                        ),
                    }]

            # Fallback: parse direct action text when code path still fails.
            if not predicted:
                fallback_used_count += 1
                pred_type, pred_amt = parse_action(response_text)
                if pred_type == "fold" and not self._looks_like_action_text(response_text):
                    parse_failed_count += 1
                predicted = self._canonicalize_action_text(
                    self._action_to_text(pred_type, pred_amt)
                )

            correct_action = self._canonicalize_action_text(correct_answer)
            reward = compute_poker_reward_simple(predicted, correct_action)
            if reward > 0:
                nonzero_reward_count += 1

            pred_type, pred_amt = parse_action(predicted)
            corr_type, _ = parse_action(correct_action)
            trajectories.append(PokerTrajectory(
                context=context,
                question=question,
                correct_answer=correct_action,
                predicted_answer=predicted,
                code=code,
                reward=reward,
                is_correct=(pred_type == corr_type),
                num_steps=1,
                messages=messages,
                action_type=pred_type,
                action_amount=pred_amt,
                parsed_stats=parsed_stats,
            ))
            if generated_ids is None:
                # Should not occur in practice, but protects log-prob computation path.
                continue
            rollout_data.append((messages, generated_ids))

        if not trajectories or not rollout_data:
            return {
                "accuracy": 0.0, "avg_reward": 0.0,
                "loss": 0.0, "baseline": self.reward_baseline,
                "batch_size": 0,
                "attempted": attempted,
                "code_extracted": extracted,
                "exec_ok": exec_ok_count,
                "nonzero_reward": nonzero_reward_count,
                "no_code": no_code_count,
                "exec_error": exec_error_count,
                "stdout_empty": stdout_empty_count,
                "fallback_used": fallback_used_count,
                "parse_failed": parse_failed_count,
                "wrapped_action_code": wrapped_action_code_count,
            }

        # Compute advantages
        rewards = [t.reward for t in trajectories]
        avg_reward = sum(rewards) / len(rewards)
        self.reward_baseline = (
            self.baseline_ema * self.reward_baseline
            + (1 - self.baseline_ema) * avg_reward
        )
        advantages = [r - self.reward_baseline for r in rewards]
        # Normalize and clip advantages to reduce REINFORCE update variance.
        adv_tensor = torch.tensor(advantages, device=self.model.device, dtype=torch.float32)
        adv_std = adv_tensor.std(unbiased=False).clamp_min(1e-6)
        adv_tensor = (adv_tensor - adv_tensor.mean()) / adv_std
        adv_tensor = torch.clamp(adv_tensor, -self.advantage_clip, self.advantage_clip)
        advantages = adv_tensor.tolist()

        # Policy gradient
        self.optimizer.zero_grad()
        total_loss = 0.0

        for (messages, gen_ids), advantage in zip(rollout_data, advantages):
            log_prob = self._compute_log_probs(messages, gen_ids)
            loss = -advantage * log_prob
            (loss / len(trajectories)).backward()
            total_loss += loss.item()

        torch.nn.utils.clip_grad_norm_(
            [p for p in self.model.parameters() if p.requires_grad],
            self.max_grad_norm,
        )
        self.optimizer.step()

        accuracy = sum(1 for t in trajectories if t.is_correct) / len(trajectories)

        # Update EMA: avg_{i+1} = (1 - gamma) * r_i + gamma * avg_i
        if self.reward_ema is None:
            self.reward_ema = avg_reward
            self.accuracy_ema = accuracy
        else:
            self.reward_ema = (1 - self.ema_gamma) * avg_reward + self.ema_gamma * self.reward_ema
            self.accuracy_ema = (1 - self.ema_gamma) * accuracy + self.ema_gamma * self.accuracy_ema

        stats = {
            "accuracy": accuracy,
            "avg_reward": avg_reward,
            "reward_ema": self.reward_ema,
            "accuracy_ema": self.accuracy_ema,
            "loss": total_loss / len(trajectories),
            "baseline": self.reward_baseline,
            "batch_size": len(trajectories),
            "attempted": attempted,
            "code_extracted": extracted,
            "exec_ok": exec_ok_count,
            "nonzero_reward": nonzero_reward_count,
            "no_code": no_code_count,
            "exec_error": exec_error_count,
            "stdout_empty": stdout_empty_count,
            "fallback_used": fallback_used_count,
            "parse_failed": parse_failed_count,
            "wrapped_action_code": wrapped_action_code_count,
        }
        self.history.append(stats)
        return stats

    def train(self, num_iterations: int = 20) -> List[Dict[str, float]]:
        """Run multiple REINFORCE iterations."""
        print(f"\n--- Poker REINFORCE: {num_iterations} iters, batch={self.batch_size}, ema_gamma={self.ema_gamma} ---")
        if self.eval_tasks:
            print(
                f"--- Held-out eval enabled: episodes={len(self.eval_tasks)}, "
                f"eval_every={self.eval_every}, seed={self.eval_seed} ---"
            )

        for i in range(num_iterations):
            stats = self.train_step()
            real_code = stats['code_extracted'] - stats.get('wrapped_action_code', 0)
            print(
                f"  Iter {i+1}/{num_iterations}: "
                f"acc={stats['accuracy']:.1%} (ema={stats['accuracy_ema']:.1%}) | "
                f"reward={stats['avg_reward']:.3f} (ema={stats['reward_ema']:.3f}) | "
                f"loss={stats['loss']:.4f} | "
                f"real_code={real_code}/{stats['attempted']} | "
                f"wrapped={stats.get('wrapped_action_code', 0)}"
            )

            if (i + 1) % 5 == 0:
                save_dir = os.path.join(self.output_dir, f"iter_{i+1}")
                os.makedirs(save_dir, exist_ok=True)
                self.model.save_pretrained(save_dir)
                self.tokenizer.save_pretrained(save_dir)
                print(f"    Checkpoint saved to {save_dir}")

            if self.eval_tasks and self.eval_every > 0 and (i + 1) % self.eval_every == 0:
                eval_stats = self._evaluate_policy()
                eval_row = {
                    "iteration": i + 1,
                    **eval_stats,
                    "train_accuracy_ema": stats.get("accuracy_ema", 0.0),
                    "train_reward_ema": stats.get("reward_ema", 0.0),
                }
                self.eval_history.append(eval_row)
                print(
                    f"    Eval @iter {i+1}: "
                    f"acc={eval_stats['eval_accuracy']:.1%} | "
                    f"reward={eval_stats['eval_avg_reward']:.3f} | "
                    f"episodes={int(eval_stats['eval_episodes'])}"
                )

                improved = (
                    eval_stats["eval_accuracy"] > self.best_eval_accuracy
                    or (
                        eval_stats["eval_accuracy"] == self.best_eval_accuracy
                        and eval_stats["eval_avg_reward"] > self.best_eval_reward
                    )
                )
                if improved:
                    self.best_eval_accuracy = eval_stats["eval_accuracy"]
                    self.best_eval_reward = eval_stats["eval_avg_reward"]
                    self.best_eval_iteration = i + 1
                    best_dir = os.path.join(self.output_dir, "best_by_eval")
                    os.makedirs(best_dir, exist_ok=True)
                    self.model.save_pretrained(best_dir)
                    self.tokenizer.save_pretrained(best_dir)
                    print(
                        f"    New best checkpoint @iter {i+1} "
                        f"(eval_acc={self.best_eval_accuracy:.1%}, eval_reward={self.best_eval_reward:.3f})"
                    )

        os.makedirs(self.output_dir, exist_ok=True)
        self.model.save_pretrained(self.output_dir)
        self.tokenizer.save_pretrained(self.output_dir)
        print(f"--- Training complete. Model saved to {self.output_dir} ---")

        return self.history

    def plot_training(self, save_path: Optional[str] = None):
        """Plot reward/accuracy/loss and execution-path diagnostics."""
        if not self.history:
            print("No training history to plot.")
            return

        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
        except ImportError:
            print("matplotlib not available — skipping plot")
            return

        iters = list(range(1, len(self.history) + 1))
        ema_rewards = [h['reward_ema'] for h in self.history]
        raw_accs = [h['accuracy'] for h in self.history]
        ema_accs = [h['accuracy_ema'] for h in self.history]
        raw_losses = [h['loss'] for h in self.history]
        real_code_ratio = [
            max(0.0, (h.get('code_extracted', 0) - h.get('wrapped_action_code', 0)) / max(h.get('attempted', 1), 1))
            for h in self.history
        ]
        wrapped_ratio = [
            h.get('wrapped_action_code', 0) / max(h.get('attempted', 1), 1)
            for h in self.history
        ]
        fallback_ratio = [
            h.get('fallback_used', 0) / max(h.get('attempted', 1), 1)
            for h in self.history
        ]

        fig, axes = plt.subplots(2, 2, figsize=(13, 9))
        axes = axes.flatten()

        # Reward plot (EMA-focused to reduce per-batch noise)
        axes[0].plot(iters, ema_rewards, '-', color='#2c3e50', linewidth=2, label=f'EMA (gamma={self.ema_gamma})')
        axes[0].set_xlabel('Iteration')
        axes[0].set_ylabel('Reward')
        axes[0].set_title('Reward EMA')
        axes[0].legend()
        axes[0].grid(alpha=0.3)

        # Accuracy plot
        axes[1].plot(iters, raw_accs, 'o-', alpha=0.3, color='#e74c3c', markersize=3, label='Raw (per batch)')
        axes[1].plot(iters, ema_accs, '-', color='#2c3e50', linewidth=2, label=f'EMA (gamma={self.ema_gamma})')
        if self.eval_history:
            eval_iters = [int(h["iteration"]) for h in self.eval_history]
            eval_accs = [h["eval_accuracy"] for h in self.eval_history]
            axes[1].plot(eval_iters, eval_accs, 's-', color='#16a085', linewidth=1.5, markersize=4, label='Held-out eval')
        axes[1].set_xlabel('Iteration')
        axes[1].set_ylabel('Accuracy')
        axes[1].set_title('Accuracy per Batch + EMA')
        axes[1].set_ylim(-0.05, 1.05)
        axes[1].legend()
        axes[1].grid(alpha=0.3)

        # Loss plot
        axes[2].plot(iters, raw_losses, 'o-', color='#8e44ad', alpha=0.8, markersize=3)
        axes[2].axhline(0.0, color='#2c3e50', linewidth=1, alpha=0.5)
        axes[2].set_xlabel('Iteration')
        axes[2].set_ylabel('Loss')
        axes[2].set_title('Policy Gradient Loss')
        axes[2].grid(alpha=0.3)

        # Execution/fallback path diagnostics
        axes[3].plot(iters, real_code_ratio, 'o-', color='#27ae60', markersize=3, label='Real code ratio')
        axes[3].plot(iters, wrapped_ratio, 'o-', color='#f39c12', markersize=3, label='Wrapped ratio')
        axes[3].plot(iters, fallback_ratio, 'o-', color='#c0392b', markersize=3, label='Fallback ratio')
        axes[3].set_xlabel('Iteration')
        axes[3].set_ylabel('Fraction of batch')
        axes[3].set_title('Execution Path Diagnostics')
        axes[3].set_ylim(-0.05, 1.05)
        axes[3].legend()
        axes[3].grid(alpha=0.3)

        plt.tight_layout()
        if save_path is None:
            save_path = os.path.join(self.output_dir, 'training_curves.png')
        os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"Training curves saved to {save_path}")

    def save_training_analysis(self, save_dir: Optional[str] = None) -> Dict[str, Any]:
        """Persist per-iteration history and compact summary JSON/CSV."""
        if not self.history:
            print("No training history to save.")
            return {}

        if save_dir is None:
            save_dir = self.output_dir
        os.makedirs(save_dir, exist_ok=True)

        history_json = os.path.join(save_dir, "training_history.json")
        summary_json = os.path.join(save_dir, "training_summary.json")
        history_csv = os.path.join(save_dir, "training_history.csv")

        with open(history_json, "w", encoding="utf-8") as f:
            json.dump(self.history, f, indent=2)

        fields = list(self.history[0].keys())
        with open(history_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(self.history)

        first = self.history[0]
        last = self.history[-1]
        summary: Dict[str, Any] = {
            "num_iterations": len(self.history),
            "first": {
                "accuracy": first.get("accuracy"),
                "avg_reward": first.get("avg_reward"),
                "loss": first.get("loss"),
            },
            "last": {
                "accuracy": last.get("accuracy"),
                "avg_reward": last.get("avg_reward"),
                "loss": last.get("loss"),
            },
            "best_accuracy": max(h.get("accuracy", 0.0) for h in self.history),
            "best_reward": max(h.get("avg_reward", 0.0) for h in self.history),
            "best_eval": {
                "iteration": self.best_eval_iteration,
                "eval_accuracy": self.best_eval_accuracy if self.best_eval_accuracy >= 0 else None,
                "eval_avg_reward": self.best_eval_reward if self.best_eval_reward >= 0 else None,
            },
            "paths": {
                "history_json": history_json,
                "history_csv": history_csv,
            },
        }

        with open(summary_json, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        self.save_eval_leaderboard(save_dir)
        print(f"Training analysis saved to {summary_json} and {history_csv}")
        return summary

    def save_eval_leaderboard(self, save_dir: Optional[str] = None) -> Dict[str, Any]:
        """Persist held-out evaluation leaderboard."""
        if save_dir is None:
            save_dir = self.output_dir
        os.makedirs(save_dir, exist_ok=True)

        leaderboard_json = os.path.join(save_dir, "eval_leaderboard.json")
        leaderboard_csv = os.path.join(save_dir, "eval_leaderboard.csv")
        best_json = os.path.join(save_dir, "best_checkpoint.json")

        if self.eval_history:
            fields = list(self.eval_history[0].keys())
            with open(leaderboard_csv, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fields)
                writer.writeheader()
                writer.writerows(self.eval_history)
            with open(leaderboard_json, "w", encoding="utf-8") as f:
                json.dump(self.eval_history, f, indent=2)

        best_payload = {
            "best_iteration": self.best_eval_iteration,
            "best_eval_accuracy": self.best_eval_accuracy if self.best_eval_accuracy >= 0 else None,
            "best_eval_avg_reward": self.best_eval_reward if self.best_eval_reward >= 0 else None,
            "best_checkpoint_dir": (
                os.path.join(self.output_dir, "best_by_eval")
                if self.best_eval_iteration > 0 else None
            ),
            "eval_episodes": len(self.eval_tasks),
            "eval_every": self.eval_every,
        }
        with open(best_json, "w", encoding="utf-8") as f:
            json.dump(best_payload, f, indent=2)
        return best_payload
