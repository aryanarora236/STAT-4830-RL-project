"""
Poker task generator for the RLM framework.

Generates (context, question, answer) tuples where:
- context: full poker game state as text (long context with hand history)
- question: "What should you do?"
- answer: the heuristic bot's decision (fold/check/call/raise $X)

The LLM agent receives the context, writes Python code to compute
pot odds / equity / hand strength, and outputs a decision.
"""

import random
from typing import Tuple, List, Optional

from src.poker.environment import (
    Card, Deck, GameState, PlayerState, Action, HandEvaluator,
    POSITIONS_6MAX, RANK_NAMES,
)
from src.poker.heuristic import HeuristicPokerBot, _hand_key, preflop_tier


def _random_stacks(n: int, bb: float) -> List[float]:
    """Generate random stack sizes in big blinds (50-200 BB)."""
    return [round(random.uniform(50, 200) * bb, 0) for _ in range(n)]


def _generate_hand_summary(deck: Deck, positions: List[str], bb: float) -> str:
    """Generate a fake previous hand summary for history context."""
    hero_pos = random.choice(positions)
    villain_pos = random.choice([p for p in positions if p != hero_pos])

    hole = deck.deal(2) if len(deck.cards) >= 2 else [Card(random.randint(2,14), random.choice("hdcs")),
                                                        Card(random.randint(2,14), random.choice("hdcs"))]
    board_size = random.choice([3, 4, 5])

    streets = ["Preflop", "Flop", "Turn", "River"][:1 + (board_size - 2) if board_size > 2 else 1]
    actions_desc = []

    # Simple narrative
    outcomes = [
        f"Hero ({hero_pos}) raised with {hole[0]}{hole[1]}, {villain_pos} called. "
        f"Hero bet flop, {villain_pos} folded. Hero won ${random.randint(5,30):.0f}.",

        f"{villain_pos} raised, Hero ({hero_pos}) called. "
        f"Board ran out, {villain_pos} bet river. Hero folded.",

        f"Hero ({hero_pos}) raised with {hole[0]}{hole[1]}, {villain_pos} 3-bet. "
        f"Hero called. Hero hit top pair on flop and won ${random.randint(10,60):.0f}.",

        f"{villain_pos} opened, Hero ({hero_pos}) called with {hole[0]}{hole[1]}. "
        f"Checked to river. Hero bet, {villain_pos} folded.",
    ]
    return random.choice(outcomes)


def generate_poker_task(
    street: Optional[str] = None,
    num_history_hands: int = 5,
) -> Tuple[str, str, str]:
    """
    Generate a single poker task.

    Returns:
        (context, question, answer)
        - context: full game state text
        - question: "What should you do? (fold/check/call/raise)"
        - answer: e.g. "raise $6" or "fold" or "call $10"
    """
    if street is None:
        street = random.choice(["preflop", "flop", "turn", "river"])

    bb = 2.0
    sb = 1.0
    positions = POSITIONS_6MAX[:6]
    hero_pos = random.choice(positions)

    # Create deck and deal
    deck = Deck()
    hero_hole = deck.deal(2)

    # Deal board based on street
    board = []
    if street == "flop":
        board = deck.deal(3)
    elif street == "turn":
        board = deck.deal(4)
    elif street == "river":
        board = deck.deal(5)

    # Generate player states
    stacks = _random_stacks(len(positions), bb)
    players = []
    for i, pos in enumerate(positions):
        players.append(PlayerState(
            position=pos,
            stack=stacks[i],
            is_active=True if pos == hero_pos else random.random() > 0.3,
        ))

    # Generate some betting actions
    actions = []
    pot = sb + bb  # blinds

    if street == "preflop":
        # Some players may have acted before hero
        hero_idx = positions.index(hero_pos)
        for i, pos in enumerate(positions):
            if pos == hero_pos:
                break
            if pos == "SB":
                continue  # SB already posted
            if pos == "BB":
                continue  # BB already posted
            p = players[i]
            if not p.is_active:
                actions.append(Action(pos, "fold"))
            elif random.random() < 0.3:
                raise_amt = round(random.uniform(2, 4) * bb, 0)
                actions.append(Action(pos, "raise", raise_amt))
                pot += raise_amt
            else:
                actions.append(Action(pos, "fold"))
                p.is_active = False
    else:
        # Postflop: generate some prior action
        active_before_hero = [
            p for p in players
            if p.is_active and p.position != hero_pos
        ]
        # Random preflop pot
        pot = round(random.uniform(6, 30), 0)

        # Some postflop action
        for p in active_before_hero[:2]:
            if random.random() < 0.4:
                bet_amt = round(random.uniform(0.3, 0.8) * pot, 0)
                actions.append(Action(p.position, "bet", bet_amt))
                pot += bet_amt

    # Determine current bet for hero
    current_bet = 0
    hero_bet = 0
    for a in actions:
        if a.action in ("raise", "bet"):
            current_bet = a.amount
    if hero_pos == "BB" and street == "preflop":
        hero_bet = bb

    # Build game state
    state = GameState(
        num_players=6,
        small_blind=sb,
        big_blind=bb,
        hero_position=hero_pos,
        hero_hole=hero_hole,
        players=players,
        board=board,
        street=street,
        pot=pot,
        current_bet=current_bet,
        hero_bet_this_round=hero_bet,
        actions=actions,
    )

    # Generate hand history for long context
    history = []
    for i in range(num_history_hands):
        history_deck = Deck()
        summary = _generate_hand_summary(history_deck, positions, bb)
        history.append(f"  Hand #{i+1}: {summary}")
    state.hand_history = history

    # Get heuristic decision
    bot = HeuristicPokerBot()
    action, amount = bot.decide(state)
    explanation = bot.explain(state)

    # Format answer
    if amount > 0:
        answer = f"{action} ${amount:.0f}"
    else:
        answer = action

    # Format context and question
    context = state.format_context()
    question = (
        "What should you do? Analyze the hand and decide: fold, check, call, or raise.\n"
        "Consider: hand strength, pot odds, position, and opponent tendencies from history.\n"
        f"Output exactly one of: fold / check / call $X / raise $X"
    )

    return context, question, answer


def generate_preflop_task() -> Tuple[str, str, str]:
    return generate_poker_task(street="preflop")


def generate_postflop_task() -> Tuple[str, str, str]:
    return generate_poker_task(street=random.choice(["flop", "turn", "river"]))


# ── Poker-specific system prompt for LLM agent ───────────────────────

POKER_SYSTEM_PROMPT = """You are a poker-playing AI agent with access to a Python REPL.

You will receive a poker game state as context. Your job:
1. Write Python code to analyze the hand (hand strength, pot odds, equity)
2. Output your decision as the LAST line of stdout

Available in the REPL:
- CONTEXT: string containing the full game state
- You can parse the context to extract hand info, board, pot, etc.
- You can compute pot odds, hand rankings, and estimated equity

Your final print statement must be exactly one of:
- "fold"
- "check"
- "call $X" (where X is the amount)
- "raise $X" (where X is the amount)

Think step by step:
1. Parse your hole cards and the board
2. Evaluate your hand strength
3. Calculate pot odds if facing a bet
4. Consider your position
5. Make a decision based on hand strength vs pot odds
"""
