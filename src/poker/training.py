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
        max_length: int = 2048,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.output_dir = output_dir
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.max_length = max_length

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

        training_args = SFTConfig(
            output_dir=self.output_dir,
            num_train_epochs=self.num_epochs,
            per_device_train_batch_size=self.batch_size,
            learning_rate=self.learning_rate,
            max_length=self.max_length,
            logging_steps=10,
            save_steps=100,
            save_total_limit=2,
            bf16=True,
            gradient_accumulation_steps=2,
            warmup_ratio=0.1,
            lr_scheduler_type="cosine",
            dataset_text_field="text",
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
        learning_rate: float = 1e-5,
        max_new_tokens: int = 1024,
        max_length: int = 2048,
        baseline_ema: float = 0.9,
        max_grad_norm: float = 1.0,
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

        trainable_params = [p for p in model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(trainable_params, lr=learning_rate)

        self.reward_baseline = 0.0
        self.history: List[Dict[str, float]] = []

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
                temperature=0.2,
                top_p=0.95,
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

    @staticmethod
    def _looks_like_action_text(text: str) -> bool:
        """Quick check for action-like text in model output."""
        return bool(re.search(r"\b(fold|check|call|raise)\b", text.lower()))

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
            context, question, correct_answer = generate_poker_task()
            base_messages = format_poker_prompt(question, context)
            base_messages[-1]["content"] += (
                "\n\nImportant output format:\n"
                "- Return ONLY executable Python code.\n"
                "- No markdown fences.\n"
                "- Last printed line must be exactly one poker action "
                "(fold/check/call $X/raise $X).\n"
            )

            messages = base_messages
            response_text = ""
            generated_ids = None
            code = None
            predicted = ""
            parsed_stats = False
            used_fallback = False
            had_code = False
            had_exec_ok = False

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
                        action_text = self._action_to_text(raw_type, raw_amt)
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
                                    "print action line (fold/check/call $X/raise $X)."
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
                                "print action line (fold/check/call $X/raise $X)."
                            ),
                        }]
                    continue

                had_code = True
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

                had_exec_ok = True
                exec_ok_count += 1
                if exec_result.stdout and exec_result.stdout.strip():
                    predicted = exec_result.stdout.strip().splitlines()[-1].strip()
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
                            "exactly one action: fold/check/call $X/raise $X."
                        ),
                    }]

            # Fallback: parse direct action text when code path still fails.
            if not predicted:
                used_fallback = True
                fallback_used_count += 1
                pred_type, pred_amt = parse_action(response_text)
                if pred_type == "fold" and not self._looks_like_action_text(response_text):
                    parse_failed_count += 1
                predicted = self._action_to_text(pred_type, pred_amt)

            reward = compute_poker_reward_simple(predicted, correct_answer)
            # Small shaping bonuses to encourage executable behavior.
            if had_code:
                reward += 0.05
            if had_exec_ok:
                reward += 0.10
            reward = min(reward, 1.0)
            if reward > 0:
                nonzero_reward_count += 1

            pred_type, pred_amt = parse_action(predicted)
            trajectories.append(PokerTrajectory(
                context=context,
                question=question,
                correct_answer=correct_answer,
                predicted_answer=predicted,
                code=code,
                reward=reward,
                is_correct=(pred_type == parse_action(correct_answer)[0]),
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
        stats = {
            "accuracy": accuracy,
            "avg_reward": avg_reward,
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
        print(f"\n--- Poker REINFORCE: {num_iterations} iters, batch={self.batch_size} ---")

        for i in range(num_iterations):
            stats = self.train_step()
            print(
                f"  Iter {i+1}/{num_iterations}: "
                f"acc={stats['accuracy']:.1%} | "
                f"reward={stats['avg_reward']:.3f} | "
                f"loss={stats['loss']:.4f} | "
                f"baseline={stats['baseline']:.3f} | "
                f"code={stats['code_extracted']}/{stats['attempted']} | "
                f"exec_ok={stats['exec_ok']} | "
                f"nz_reward={stats['nonzero_reward']} | "
                f"fb={stats['fallback_used']} | "
                f"no_code={stats['no_code']} | "
                f"exec_err={stats['exec_error']} | "
                f"stdout_empty={stats['stdout_empty']} | "
                f"wrapped={stats['wrapped_action_code']}"
            )

            if (i + 1) % 5 == 0:
                save_dir = os.path.join(self.output_dir, f"iter_{i+1}")
                os.makedirs(save_dir, exist_ok=True)
                self.model.save_pretrained(save_dir)
                self.tokenizer.save_pretrained(save_dir)
                print(f"    Checkpoint saved to {save_dir}")

        os.makedirs(self.output_dir, exist_ok=True)
        self.model.save_pretrained(self.output_dir)
        self.tokenizer.save_pretrained(self.output_dir)
        print(f"--- Training complete. Model saved to {self.output_dir} ---")

        return self.history
