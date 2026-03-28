"""
Reward functions for poker RL training.

Two modes:
- BC reward: binary action-type match (for filtering trajectories)
- RL reward: EV-based with action quality + reasoning bonus
"""

import re
import math
from typing import Tuple, Optional, List
from src.poker.environment import Card, GameState, HandEvaluator


def parse_action(text: str) -> Tuple[str, float]:
    """
    Parse an action string into (action_type, amount).
    Handles: "fold", "check", "call $6", "raise $12", "call 6", "Raise $12.0", etc.
    """
    text = text.strip().lower()

    if text == "fold":
        return ("fold", 0.0)
    if text == "check":
        return ("check", 0.0)

    m = re.match(r"(call|raise|bet)\s*\$?\s*(\d+\.?\d*)", text)
    if m:
        action = m.group(1)
        amount = float(m.group(2))
        if action == "bet":
            action = "raise"
        return (action, amount)

    # Fallback: just action type
    for action in ("fold", "check", "call", "raise"):
        if action in text:
            return (action, 0.0)

    return ("fold", 0.0)


def _action_type_match(predicted: str, correct: str) -> float:
    """Score how well predicted action type matches correct."""
    pred_type, _ = parse_action(predicted)
    corr_type, _ = parse_action(correct)

    if pred_type == corr_type:
        return 1.0

    # Partial credit: both staying in the pot (call vs raise)
    staying = {"call", "raise", "check"}
    if pred_type in staying and corr_type in staying:
        return 0.3

    # Fold when should play or play when should fold
    return 0.0


def _ev_reward(
    predicted: str,
    state: GameState,
    equity: Optional[float] = None,
) -> float:
    """
    Compute EV-based reward for a poker action.

    Uses pot odds vs equity to judge whether the action was +EV.
    """
    pred_type, pred_amount = parse_action(predicted)
    pot = state.pot
    to_call = state.to_call
    pot_odds = state.pot_odds

    # Compute equity if not provided (expensive, use fewer sims for speed)
    if equity is None:
        if state.board and state.hero_hole:
            equity = HandEvaluator.equity_estimate(
                state.hero_hole, state.board, num_simulations=200
            )
        elif state.hero_hole:
            # Preflop: rough equity estimate
            equity = HandEvaluator.equity_estimate(
                state.hero_hole, [], num_simulations=200
            )
        else:
            equity = 0.5

    if pred_type == "fold":
        # Folding is correct when equity < pot_odds
        if to_call == 0:
            return 0.0  # folding when you can check is bad
        if equity < pot_odds:
            return 1.0  # correct fold
        else:
            # Folding with positive equity is a mistake
            return max(0.0, 1.0 - (equity - pot_odds) * 3)

    if pred_type == "check":
        # Checking is neutral
        return 0.5

    if pred_type == "call":
        if to_call == 0:
            return 0.5  # calling 0 = checking, neutral
        # Calling is correct when equity > pot_odds
        ev = equity * (pot + to_call) - (1 - equity) * to_call
        # Sigmoid-like scaling
        return 1.0 / (1.0 + math.exp(-ev / (state.big_blind * 2)))

    if pred_type == "raise":
        # Raising is good with strong equity
        if equity > 0.6:
            return 0.9  # strong value raise
        elif equity > 0.4:
            return 0.5  # marginal raise
        else:
            # Bluff — could be good or bad, give partial credit
            return 0.3

    return 0.0


def compute_poker_reward(
    predicted_action: str,
    correct_action: str,
    state: GameState,
    parsed_stats: bool = False,
    num_steps: int = 1,
    max_steps: int = 5,
    equity: Optional[float] = None,
) -> float:
    """
    Full poker reward combining action match, EV, and reasoning quality.

    R = 0.5 * action_match + 0.3 * ev_reward + 0.2 * reasoning_bonus - step_penalty

    Args:
        predicted_action: model's output (e.g., "raise $10")
        correct_action: heuristic's answer
        state: GameState for EV computation
        parsed_stats: whether the model's code parsed opponent stats
        num_steps: REPL steps used
        max_steps: max allowed steps
        equity: pre-computed equity (optional, saves time)
    """
    action_match = _action_type_match(predicted_action, correct_action)
    ev = _ev_reward(predicted_action, state, equity)
    reasoning_bonus = 0.2 if parsed_stats else 0.0
    step_penalty = 0.05 * (num_steps / max_steps)

    reward = 0.5 * action_match + 0.3 * ev + 0.2 * reasoning_bonus - step_penalty
    return round(reward, 4)


def compute_poker_reward_simple(
    predicted_action: str,
    correct_action: str,
) -> float:
    """Simple reward for BC trajectory filtering: action type match."""
    return _action_type_match(predicted_action, correct_action)
