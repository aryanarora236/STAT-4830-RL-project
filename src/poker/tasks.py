"""
Poker task generator for the RLM framework.

Generates (context, question, answer) tuples where:
- context: full poker game state as text (with structured hand history)
- question: "What should you do?"
- answer: the heuristic bot's decision

Hand histories are generated from consistent opponent profiles (archetypes)
so the model must learn to parse history and extract opponent tendencies.
"""

import random
from typing import Tuple, List, Optional, Dict

from src.poker.environment import (
    Card, Deck, GameState, PlayerState, Action, HandRecord,
    HandEvaluator, OpponentProfile, OPPONENT_ARCHETYPES,
    POSITIONS_6MAX, RANK_NAMES,
)
from src.poker.heuristic import HeuristicPokerBot, _hand_key, preflop_tier


# ── Opponent Profile Assignment ───────────────────────────────────────

def _assign_profiles(positions: List[str], hero_pos: str) -> Dict[str, OpponentProfile]:
    """Assign a random archetype to each opponent."""
    archetypes = list(OPPONENT_ARCHETYPES.values())
    profiles = {}
    for pos in positions:
        if pos != hero_pos:
            profiles[pos] = random.choice(archetypes)
    return profiles


# ── Structured Hand History Generation ────────────────────────────────

def _generate_hand_record(
    hand_num: int,
    positions: List[str],
    profiles: Dict[str, OpponentProfile],
    bb: float,
) -> HandRecord:
    """
    Generate a single hand record consistent with opponent profiles.

    Each opponent's actions are probabilistically determined by their
    profile stats (VPIP, PFR, aggression, fold-to-cbet, etc.).
    """
    deck = Deck()
    actions_by_street: Dict[str, List[Action]] = {}
    board_by_street: Dict[str, List[Card]] = {}
    active = {pos: True for pos in positions}
    pot = 1.5 * bb  # blinds

    # ── Preflop ──
    preflop_actions = []
    first_raiser = None
    current_bet = bb

    for pos in positions:
        if not active[pos]:
            continue
        profile = profiles.get(pos)
        if not profile:
            # Hero position — just play generically
            if random.random() < 0.25:
                raise_amt = round(random.uniform(2.5, 3.5) * bb, 0)
                preflop_actions.append(Action(pos, "raise", raise_amt))
                current_bet = raise_amt
                pot += raise_amt
                if not first_raiser:
                    first_raiser = pos
            elif random.random() < 0.4:
                preflop_actions.append(Action(pos, "call", current_bet))
                pot += current_bet
            else:
                preflop_actions.append(Action(pos, "fold"))
                active[pos] = False
            continue

        # Opponent acts according to profile
        if first_raiser is None:
            # No raise yet — decide to raise or fold
            if random.random() < profile.pfr:
                raise_amt = round(random.uniform(2.5, 3.5) * bb, 0)
                preflop_actions.append(Action(pos, "raise", raise_amt))
                current_bet = raise_amt
                pot += raise_amt
                first_raiser = pos
            elif random.random() < profile.vpip:
                preflop_actions.append(Action(pos, "call", current_bet))
                pot += current_bet
            else:
                preflop_actions.append(Action(pos, "fold"))
                active[pos] = False
        else:
            # Facing a raise
            if random.random() < profile.three_bet_pct:
                raise_amt = round(current_bet * random.uniform(2.5, 3.5), 0)
                preflop_actions.append(Action(pos, "raise", raise_amt))
                current_bet = raise_amt
                pot += raise_amt
            elif random.random() < profile.vpip:
                preflop_actions.append(Action(pos, "call", current_bet))
                pot += current_bet
            else:
                preflop_actions.append(Action(pos, "fold"))
                active[pos] = False

    actions_by_street["preflop"] = preflop_actions

    # Count active players
    active_count = sum(1 for v in active.values() if v)
    if active_count < 2:
        active_players = [pos for pos, a in active.items() if a]
        winner = active_players[0] if active_players else positions[0]
        return HandRecord(
            hand_number=hand_num,
            actions_by_street=actions_by_street,
            board_by_street=board_by_street,
            winner=winner,
            pot_won=pot,
            players_involved=positions,
        )

    # ── Postflop streets ──
    for street, num_cards in [("flop", 3), ("turn", 1), ("river", 1)]:
        board_cards = deck.deal(num_cards)
        prev_board = board_by_street.get(
            {"flop": None, "turn": "flop", "river": "turn"}[street], []
        )
        all_board = prev_board + board_cards
        board_by_street[street] = all_board

        street_actions = []
        current_bet = 0
        cbet_made = False

        for pos in positions:
            if not active[pos]:
                continue
            profile = profiles.get(pos)

            if not profile:
                # Hero — simple random action
                if current_bet > 0:
                    if random.random() < 0.4:
                        street_actions.append(Action(pos, "call", current_bet))
                        pot += current_bet
                    else:
                        street_actions.append(Action(pos, "fold"))
                        active[pos] = False
                else:
                    if random.random() < 0.3:
                        bet_amt = round(random.uniform(0.3, 0.7) * pot, 0)
                        street_actions.append(Action(pos, "bet", bet_amt))
                        current_bet = bet_amt
                        pot += bet_amt
                    else:
                        street_actions.append(Action(pos, "check"))
                continue

            # Opponent acts based on profile
            if current_bet > 0:
                # Facing a bet
                is_cbet = (street == "flop" and cbet_made)

                if is_cbet and random.random() < profile.fold_to_cbet:
                    street_actions.append(Action(pos, "fold"))
                    active[pos] = False
                elif random.random() < (1.0 / (1.0 + profile.aggression)):
                    # Passive: call
                    street_actions.append(Action(pos, "call", current_bet))
                    pot += current_bet
                elif random.random() < profile.aggression / 4.0:
                    # Aggressive: raise
                    raise_amt = round(current_bet * random.uniform(2.0, 3.0), 0)
                    street_actions.append(Action(pos, "raise", raise_amt))
                    pot += raise_amt
                    current_bet = raise_amt
                else:
                    street_actions.append(Action(pos, "call", current_bet))
                    pot += current_bet
            else:
                # No bet yet
                # C-bet if preflop raiser
                if street == "flop" and pos == first_raiser:
                    if random.random() < 0.65:
                        bet_amt = round(random.uniform(0.4, 0.7) * pot, 0)
                        street_actions.append(Action(pos, "bet", bet_amt))
                        current_bet = bet_amt
                        pot += bet_amt
                        cbet_made = True
                        continue

                # Normal betting based on aggression
                if random.random() < profile.aggression / 5.0:
                    bet_amt = round(random.uniform(0.3, 0.7) * pot, 0)
                    street_actions.append(Action(pos, "bet", bet_amt))
                    current_bet = bet_amt
                    pot += bet_amt
                else:
                    street_actions.append(Action(pos, "check"))

        actions_by_street[street] = street_actions
        active_count = sum(1 for v in active.values() if v)
        if active_count < 2:
            break

    # Pick winner from active players
    active_players = [pos for pos, a in active.items() if a]
    winner = random.choice(active_players)

    return HandRecord(
        hand_number=hand_num,
        actions_by_street=actions_by_street,
        board_by_street=board_by_street,
        winner=winner,
        pot_won=round(pot, 0),
        players_involved=positions,
    )


# ── Task Generation ──────────────────────────────────────────────────

def generate_poker_task(
    street: Optional[str] = None,
    num_history_hands: int = 15,
) -> Tuple[str, str, str]:
    """
    Generate a single poker task with structured hand history.

    Returns:
        (context, question, answer)
    """
    if street is None:
        street = random.choice(["preflop", "flop", "turn", "river"])

    bb = 2.0
    sb = 1.0
    positions = POSITIONS_6MAX[:6]
    hero_pos = random.choice(positions)

    # Assign consistent opponent profiles
    profiles = _assign_profiles(positions, hero_pos)

    # Create deck and deal
    deck = Deck()
    hero_hole = deck.deal(2)

    # Deal board
    board = []
    if street == "flop":
        board = deck.deal(3)
    elif street == "turn":
        board = deck.deal(4)
    elif street == "river":
        board = deck.deal(5)

    # Generate player states
    players = []
    for pos in positions:
        stack = round(random.uniform(50, 200) * bb, 0)
        players.append(PlayerState(
            position=pos,
            stack=stack,
            is_active=True if pos == hero_pos else random.random() > 0.25,
            profile=profiles.get(pos),
        ))

    # Generate betting actions for current hand
    actions = []
    pot = sb + bb

    if street == "preflop":
        for i, pos in enumerate(positions):
            if pos == hero_pos:
                break
            if pos in ("SB", "BB"):
                continue
            p = players[i]
            if not p.is_active:
                actions.append(Action(pos, "fold"))
                continue
            profile = profiles.get(pos)
            if profile and random.random() < profile.pfr:
                raise_amt = round(random.uniform(2.5, 3.5) * bb, 0)
                actions.append(Action(pos, "raise", raise_amt))
                pot += raise_amt
            elif profile and random.random() < profile.vpip:
                actions.append(Action(pos, "call", bb))
                pot += bb
            else:
                actions.append(Action(pos, "fold"))
                p.is_active = False
    else:
        pot = round(random.uniform(6, 30), 0)
        active_opps = [p for p in players if p.is_active and p.position != hero_pos]
        for p in active_opps[:2]:
            profile = profiles.get(p.position)
            if profile and random.random() < profile.aggression / 4.0:
                bet_amt = round(random.uniform(0.3, 0.8) * pot, 0)
                actions.append(Action(p.position, "bet", bet_amt))
                pot += bet_amt

    # Determine current bet
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

    # Generate hand history from consistent profiles
    history = []
    for i in range(num_history_hands):
        record = _generate_hand_record(i + 1, positions, profiles, bb)
        history.append(record)
    state.hand_history = history

    # Get heuristic decision with full reasoning
    bot = HeuristicPokerBot()
    trace = bot.decide_with_reasoning(state)

    # Format answer
    if trace.final_amount > 0:
        answer = f"{trace.final_action} ${trace.final_amount:.0f}"
    else:
        answer = trace.final_action

    # Format context and question
    context = state.format_context()
    question = (
        "What should you do? Analyze the hand and decide: fold, check, call, or raise.\n"
        "You MUST use the hand history to identify opponent tendencies before deciding.\n"
        "Steps:\n"
        "  1. Parse previous hands to compute opponent stats (VPIP, PFR, aggression, fold-to-cbet)\n"
        "  2. Evaluate your hand strength, pot odds, and draws\n"
        "  3. Adjust your decision based on opponent profile\n"
        "Output exactly one of: fold / check / call $X / raise $X"
    )

    return context, question, answer


def generate_preflop_task() -> Tuple[str, str, str]:
    return generate_poker_task(street="preflop")


def generate_postflop_task() -> Tuple[str, str, str]:
    return generate_poker_task(street=random.choice(["flop", "turn", "river"]))


def generate_poker_task_with_trace(
    street: Optional[str] = None,
    num_history_hands: int = 15,
) -> Tuple[str, str, str, str]:
    """
    Same as generate_poker_task but also returns the full reasoning trace.

    Returns:
        (context, question, answer, reasoning_trace)
    """
    if street is None:
        street = random.choice(["preflop", "flop", "turn", "river"])

    bb = 2.0
    sb = 1.0
    positions = POSITIONS_6MAX[:6]
    hero_pos = random.choice(positions)
    profiles = _assign_profiles(positions, hero_pos)

    deck = Deck()
    hero_hole = deck.deal(2)

    board = []
    if street == "flop":
        board = deck.deal(3)
    elif street == "turn":
        board = deck.deal(4)
    elif street == "river":
        board = deck.deal(5)

    players = []
    for pos in positions:
        stack = round(random.uniform(50, 200) * bb, 0)
        players.append(PlayerState(
            position=pos,
            stack=stack,
            is_active=True if pos == hero_pos else random.random() > 0.25,
            profile=profiles.get(pos),
        ))

    actions = []
    pot = sb + bb

    if street == "preflop":
        for i, pos in enumerate(positions):
            if pos == hero_pos:
                break
            if pos in ("SB", "BB"):
                continue
            p = players[i]
            if not p.is_active:
                actions.append(Action(pos, "fold"))
                continue
            profile = profiles.get(pos)
            if profile and random.random() < profile.pfr:
                raise_amt = round(random.uniform(2.5, 3.5) * bb, 0)
                actions.append(Action(pos, "raise", raise_amt))
                pot += raise_amt
            elif profile and random.random() < profile.vpip:
                actions.append(Action(pos, "call", bb))
                pot += bb
            else:
                actions.append(Action(pos, "fold"))
                p.is_active = False
    else:
        pot = round(random.uniform(6, 30), 0)
        active_opps = [p for p in players if p.is_active and p.position != hero_pos]
        for p in active_opps[:2]:
            profile = profiles.get(p.position)
            if profile and random.random() < profile.aggression / 4.0:
                bet_amt = round(random.uniform(0.3, 0.8) * pot, 0)
                actions.append(Action(p.position, "bet", bet_amt))
                pot += bet_amt

    current_bet = 0
    hero_bet = 0
    for a in actions:
        if a.action in ("raise", "bet"):
            current_bet = a.amount
    if hero_pos == "BB" and street == "preflop":
        hero_bet = bb

    state = GameState(
        num_players=6, small_blind=sb, big_blind=bb,
        hero_position=hero_pos, hero_hole=hero_hole,
        players=players, board=board, street=street,
        pot=pot, current_bet=current_bet,
        hero_bet_this_round=hero_bet, actions=actions,
    )

    history = []
    for i in range(num_history_hands):
        record = _generate_hand_record(i + 1, positions, profiles, bb)
        history.append(record)
    state.hand_history = history

    bot = HeuristicPokerBot()
    trace = bot.decide_with_reasoning(state)

    if trace.final_amount > 0:
        answer = f"{trace.final_action} ${trace.final_amount:.0f}"
    else:
        answer = trace.final_action

    context = state.format_context()
    question = (
        "What should you do? Analyze the hand and decide: fold, check, call, or raise.\n"
        "You MUST use the hand history to identify opponent tendencies before deciding.\n"
        "Steps:\n"
        "  1. Parse previous hands to compute opponent stats (VPIP, PFR, aggression, fold-to-cbet)\n"
        "  2. Evaluate your hand strength, pot odds, and draws\n"
        "  3. Adjust your decision based on opponent profile\n"
        "Output exactly one of: fold / check / call $X / raise $X"
    )

    return context, question, answer, trace.format()


# ── Poker-specific system prompt for LLM agent ───────────────────────

POKER_SYSTEM_PROMPT = """You are a poker-playing AI agent with access to a Python REPL.

You will receive a poker game state as context, including hand history from previous hands.
Your job is to follow this 3-step process:

STEP 1 - RETRIEVE: Parse the hand history to compute opponent stats.
Write Python code to extract from CONTEXT:
- VPIP (% of hands each opponent voluntarily put money in)
- PFR (% of hands each opponent raised preflop)
- Aggression factor (bets+raises / calls postflop)
- Fold to c-bet % (how often they fold to continuation bets)

STEP 2 - COMPUTE: Evaluate the current hand.
- Parse your hole cards and board
- Determine hand strength (pair, flush draw, etc.)
- Calculate pot odds if facing a bet
- Consider position

STEP 3 - DECIDE: Combine hand analysis with opponent profile.
- Against loose-passive (high VPIP, low aggression): value bet thinner, bluff less
- Against tight (low VPIP): respect their bets, fold marginal hands
- Against aggressive (high aggression): call down wider, they may be bluffing
- Against high fold-to-cbet: bluff more on flops

Available in the REPL:
- CONTEXT: string containing the full game state + hand history
- Standard Python (re, collections, etc.)

Your final print statement must be exactly one of:
- "fold"
- "check"
- "call $X" (where X is the amount)
- "raise $X" (where X is the amount)
"""
