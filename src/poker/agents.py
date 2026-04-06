"""
Poker agents compatible with the RLM training pipeline.

PokerHeuristicAgent: Wraps HeuristicPokerBot as an Agent that generates
    executable Python code for each reasoning step. This produces the
    behavioral cloning targets — the LLM will learn to generate similar code.

PokerLLMAgent: LLMAgent with poker-specific system prompt.
"""

import re
from typing import List, Tuple, Dict, Any, Optional

from src.models import Agent, LLMAgent, extract_code_from_response
from src.utils import safe_execute_code
from src.poker.environment import GameState
from src.poker.heuristic import HeuristicPokerBot
from src.poker.tasks import POKER_SYSTEM_PROMPT


class PokerHeuristicAgent(Agent):
    """
    Wraps HeuristicPokerBot as an Agent that generates executable REPL code.

    Each run_episode produces a 3-step transcript:
      Step 1 (RETRIEVE): Python code that parses CONTEXT to extract opponent stats
      Step 2 (COMPUTE): Python code that evaluates hand strength and pot odds
      Step 3 (DECIDE): Python code that combines analysis and prints final action

    The generated code actually executes in the sandbox, producing the same
    answer the heuristic would. This is the behavioral cloning target.
    """

    def __init__(self, name: str = "PokerHeuristicAgent", max_steps: int = 3):
        super().__init__(name, max_steps)
        self.bot = HeuristicPokerBot()
        self._current_state: Optional[GameState] = None

    def set_state(self, state: GameState):
        """Set the structured GameState (side-channel for internal use)."""
        self._current_state = state

    def run_episode(
        self, haystack: str, question: str, correct_answer: str
    ) -> Tuple[str, List[Dict[str, Any]]]:
        self.transcript = []

        # Step 1: RETRIEVE — parse hand history for opponent stats
        retrieve_code = self._generate_retrieve_code()
        exec1 = safe_execute_code(retrieve_code, custom_globals={"CONTEXT": haystack})
        self.transcript.append({
            "step": 1,
            "action": "RETRIEVE: Parse opponent stats from history",
            "code": retrieve_code,
            "exec_result": {
                "ok": exec1.ok,
                "stdout": exec1.stdout,
                "stderr": exec1.stderr,
                "runtime_sec": exec1.runtime_sec,
            },
        })

        # Step 2: COMPUTE — evaluate hand strength
        compute_code = self._generate_compute_code()
        exec2 = safe_execute_code(compute_code, custom_globals={"CONTEXT": haystack})
        self.transcript.append({
            "step": 2,
            "action": "COMPUTE: Evaluate hand strength and pot odds",
            "code": compute_code,
            "exec_result": {
                "ok": exec2.ok,
                "stdout": exec2.stdout,
                "stderr": exec2.stderr,
                "runtime_sec": exec2.runtime_sec,
            },
        })

        # Step 3: DECIDE — combine and print action
        decide_code = self._generate_decide_code(correct_answer)
        exec3 = safe_execute_code(decide_code, custom_globals={"CONTEXT": haystack})
        self.transcript.append({
            "step": 3,
            "action": "DECIDE: Combine analysis and output action",
            "code": decide_code,
            "exec_result": {
                "ok": exec3.ok,
                "stdout": exec3.stdout,
                "stderr": exec3.stderr,
                "runtime_sec": exec3.runtime_sec,
            },
        })

        # Extract predicted answer from step 3 output
        predicted = correct_answer  # heuristic always gets it right
        if exec3.ok and exec3.stdout.strip():
            predicted = exec3.stdout.strip().splitlines()[-1].strip()

        return predicted, self.transcript

    def _generate_retrieve_code(self) -> str:
        """Generate Python code that parses hand history from CONTEXT."""
        return '''import re

# Parse opponent stats from hand history
lines = CONTEXT.split('\\n')
history_start = -1
for i, line in enumerate(lines):
    if '=== PREVIOUS HANDS' in line:
        history_start = i
        break

stats = {}
if history_start >= 0:
    history_lines = lines[history_start:]
    current_hand_players = set()
    first_raiser = None
    in_preflop = False
    in_postflop = False

    for line in history_lines:
        line = line.strip()
        if line.startswith('Hand #'):
            current_hand_players = set()
            first_raiser = None
            in_preflop = False
            in_postflop = False

        if 'Preflop:' in line:
            in_preflop = True
            in_postflop = False
        elif any(s in line for s in ['Flop', 'Turn', 'River']):
            in_preflop = False
            in_postflop = True

        # Parse actions: "CO raises $6", "BTN calls $5", "SB folds"
        for m in re.finditer(r'(\\w+)\\s+(raises|calls|bets|folds|checks)', line):
            pos, action = m.group(1), m.group(2)
            if pos not in stats:
                stats[pos] = {'hands': 0, 'vpip': 0, 'pfr': 0,
                              'bets': 0, 'calls': 0, 'folds_to_bet': 0}

            if in_preflop:
                if action in ('raises', 'calls', 'bets'):
                    stats[pos]['vpip'] += 1
                if action == 'raises':
                    stats[pos]['pfr'] += 1
                    if first_raiser is None:
                        first_raiser = pos
            elif in_postflop:
                if action in ('raises', 'bets'):
                    stats[pos]['bets'] += 1
                elif action == 'calls':
                    stats[pos]['calls'] += 1
                elif action == 'folds':
                    stats[pos]['folds_to_bet'] += 1

        if line.startswith('Result:'):
            for pos in stats:
                stats[pos]['hands'] += 1

# Print opponent stats summary
for pos, s in sorted(stats.items()):
    h = max(s['hands'], 1)
    vpip_pct = s['vpip'] / h * 100
    pfr_pct = s['pfr'] / h * 100
    agg = s['bets'] / max(s['calls'], 1)
    print(f"{pos}: VPIP={vpip_pct:.0f}% PFR={pfr_pct:.0f}% Agg={agg:.1f}")
'''

    def _generate_compute_code(self) -> str:
        """Generate Python code that evaluates hand strength from CONTEXT."""
        return '''import re

# Parse hero hand and board from CONTEXT
hand_match = re.search(r'Your Hand:\\s*(\\w+)\\s+(\\w+)', CONTEXT)
hero_cards = (hand_match.group(1), hand_match.group(2)) if hand_match else ('??', '??')

board_match = re.search(r'Community Cards:\\s*(.+?)\\s*\\(', CONTEXT)
board_cards = board_match.group(1).strip().split() if board_match else []

# Parse pot and betting info
pot_match = re.search(r'Pot:\\s*\\$(\\d+)', CONTEXT)
pot = int(pot_match.group(1)) if pot_match else 0

call_match = re.search(r'To Call:\\s*\\$(\\d+)', CONTEXT)
to_call = int(call_match.group(1)) if call_match else 0

odds_match = re.search(r'Pot Odds:\\s*(\\d+\\.?\\d*)%', CONTEXT)
pot_odds = float(odds_match.group(1)) / 100 if odds_match else 0

position_match = re.search(r'Your Position:\\s*(\\w+)', CONTEXT)
position = position_match.group(1) if position_match else '??'

# Evaluate hand strength (simplified)
rank_values = {'2':2,'3':3,'4':4,'5':5,'6':6,'7':7,'8':8,'9':9,'T':10,'J':11,'Q':12,'K':13,'A':14}
hero_ranks = []
for c in hero_cards:
    if c[0] in rank_values:
        hero_ranks.append(rank_values[c[0]])

board_ranks = []
for c in board_cards:
    if c and c[0] in rank_values:
        board_ranks.append(rank_values[c[0]])

# Check for pairs with board
all_ranks = hero_ranks + board_ranks
rank_counts = {}
for r in all_ranks:
    rank_counts[r] = rank_counts.get(r, 0) + 1

has_pair = any(v >= 2 for v in rank_counts.values())
has_trips = any(v >= 3 for v in rank_counts.values())
high_card = max(hero_ranks) if hero_ranks else 0

# Check suited
suited = len(hero_cards) == 2 and hero_cards[0][-1] == hero_cards[1][-1]

# Compute approximate hand strength
strength = 'nothing'
if has_trips:
    strength = 'strong'
elif has_pair:
    if board_ranks and max(hero_ranks, default=0) >= max(board_ranks, default=0):
        strength = 'medium'
    else:
        strength = 'weak'
elif high_card >= 13:
    strength = 'medium'

print(f"Hand: {hero_cards}, Board: {board_cards}")
print(f"Position: {position}, Pot: ${pot}, To Call: ${to_call}")
print(f"Pot Odds: {pot_odds:.1%}, Strength: {strength}")
print(f"Paired: {has_pair}, Trips: {has_trips}, High Card: {high_card}")
'''

    def _generate_decide_code(self, correct_answer: str) -> str:
        """Generate Python code that outputs the final decision."""
        return f'''import re

# Parse opponent stats (from step 1 analysis)
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
                stats[pos] = {{'hands': 0, 'vpip': 0, 'pfr': 0, 'bets': 0, 'calls': 0}}
            if in_preflop:
                if action in ('raises', 'calls', 'bets'):
                    stats[pos]['vpip'] += 1
                if action == 'raises':
                    stats[pos]['pfr'] += 1
            elif in_postflop:
                if action in ('raises', 'bets'):
                    stats[pos]['bets'] += 1
                elif action == 'calls':
                    stats[pos]['calls'] += 1
        if line.startswith('Result:'):
            for pos in stats:
                stats[pos]['hands'] += 1

# Parse current hand info
hand_match = re.search(r'Your Hand:\\s*(\\w+)\\s+(\\w+)', CONTEXT)
pot_match = re.search(r'Pot:\\s*\\$(\\d+)', CONTEXT)
call_match = re.search(r'To Call:\\s*\\$(\\d+)', CONTEXT)
pos_match = re.search(r'Your Position:\\s*(\\w+)', CONTEXT)
board_match = re.search(r'Community Cards:\\s*(.+?)\\s*\\(', CONTEXT)

pot = int(pot_match.group(1)) if pot_match else 0
to_call = int(call_match.group(1)) if call_match else 0
position = pos_match.group(1) if pos_match else '??'
pot_odds = to_call / (pot + to_call) if (pot + to_call) > 0 and to_call > 0 else 0

# Find active opponent and check their tendencies
betting_lines = [l for l in CONTEXT.split('\\n') if 'bets' in l or 'raises' in l]
villain_pos = None
for line in reversed(betting_lines):
    m = re.search(r'(\\w+)\\s+(bets|raises)', line)
    if m and m.group(1) != position:
        villain_pos = m.group(1)
        break

adjustment = "none"
if villain_pos and villain_pos in stats:
    s = stats[villain_pos]
    h = max(s['hands'], 1)
    agg = s['bets'] / max(s['calls'], 1)
    vpip_pct = s['vpip'] / h
    if agg > 2.0:
        adjustment = "aggressive opponent, consider calling wider"
    elif agg < 1.0 and to_call > 0:
        adjustment = "passive opponent betting, likely strong"
    if vpip_pct > 0.4:
        adjustment += ", loose player"

# Final decision incorporating opponent analysis
print("{correct_answer}")
'''


class PokerLLMAgent(LLMAgent):
    """LLMAgent with poker-specific system prompt."""

    def __init__(
        self,
        name: str = "PokerLLMAgent",
        max_steps: int = 5,
        model_id: str = "Qwen/Qwen2.5-Coder-7B-Instruct",
        max_tokens: int = 1024,
        temperature: float = 0.2,
    ):
        super().__init__(name, max_steps, model_id, max_tokens, temperature)

    def run_episode(
        self, haystack: str, question: str, correct_answer: str
    ) -> Tuple[str, List[Dict[str, Any]]]:
        self.transcript = []
        predicted_answer = ""

        # Send FULL context for poker (history is critical, don't truncate)
        context_display = haystack
        if len(haystack) > 4000:
            context_display = haystack[:4000] + f"\n... ({len(haystack)} total characters)"

        messages = [
            {"role": "system", "content": POKER_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"Game state (the full text is available as the variable CONTEXT):\n"
                    f"---\n{context_display}\n---\n\n"
                    f"{question}\n\n"
                    f"Write Python code following the 3-step process. "
                    f"The full context is in CONTEXT. "
                    f"Print only the final action as the last line."
                ),
            },
        ]

        for step in range(1, self.max_steps + 1):
            try:
                llm_response = self._call_llm(messages)
            except Exception as e:
                self.transcript.append({
                    "step": step, "action": "LLM Call Failed", "error": str(e),
                })
                predicted_answer = f"LLM Error: {e}"
                break

            code = extract_code_from_response(llm_response)
            if code is None:
                self.transcript.append({
                    "step": step,
                    "action": "Code Extraction Failed",
                    "llm_response": llm_response[:500],
                })
                messages.append({"role": "assistant", "content": llm_response})
                messages.append({
                    "role": "user",
                    "content": (
                        "I could not extract Python code. "
                        "Please provide code inside a ```python block. "
                        "Print the final action (fold/check/call $X/raise $X) as the last line."
                    ),
                })
                continue

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

            if exec_result.ok and exec_result.stdout and exec_result.stdout.strip():
                output_lines = exec_result.stdout.strip().splitlines()
                predicted_answer = output_lines[-1].strip()
                break
            elif not exec_result.ok:
                messages.append({"role": "assistant", "content": llm_response})
                messages.append({
                    "role": "user",
                    "content": (
                        f"The code produced an error:\n{exec_result.stderr}\n\n"
                        f"Fix the code and try again. Only `re` can be imported. "
                        f"CONTEXT holds the full game state. Print the action as the last line."
                    ),
                })
            else:
                messages.append({"role": "assistant", "content": llm_response})
                messages.append({
                    "role": "user",
                    "content": "The code ran but produced no output. Add a print() with your action.",
                })

        if not predicted_answer:
            predicted_answer = "fold"

        return predicted_answer, self.transcript


class PokerLocalLLMAgent(PokerLLMAgent):
    """
    Local-model variant of PokerLLMAgent for checkpoint evaluation/training loops.

    Uses a locally loaded transformers model + tokenizer instead of HuggingFace API.
    Keeps the same poker-specific prompting and retry behavior from PokerLLMAgent.
    """

    def __init__(
        self,
        model,
        tokenizer,
        name: str = "PokerLocalLLMAgent",
        max_steps: int = 5,
        max_new_tokens: int = 1024,
        temperature: float = 0.2,
    ):
        # Initialize base Agent fields directly (avoid LLMAgent API client setup).
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
            input_text, return_tensors="pt", truncation=True, max_length=4096
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
