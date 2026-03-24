"""
Heuristic poker bot: tight-aggressive (TAG) strategy.

This serves as the "solver" baseline for trajectory collection.
The LLM agent will be trained to match and then surpass this strategy.
"""

from typing import List, Tuple
from src.poker.environment import (
    Card, GameState, HandEvaluator, HandRank, RANK_FROM_CHAR,
)


# ── Preflop Hand Rankings ─────────────────────────────────────────────

# Hand tiers based on standard TAG preflop charts.
# Tier 1 (premium): always raise/reraise
# Tier 2 (strong): raise, call 3-bets
# Tier 3 (playable): raise in position, call raises
# Tier 4 (speculative): call in late position if cheap
# Tier 5 (trash): fold

def _hand_key(hole: List[Card]) -> str:
    """
    Convert hole cards to a canonical key like 'AKs', 'QJo', 'TT'.
    Suited = same suit, 's' suffix. Off-suit = 'o' suffix. Pairs have no suffix.
    Higher rank first.
    """
    c1, c2 = hole
    r1, r2 = c1.rank, c2.rank
    if r1 < r2:
        r1, r2 = r2, r1

    from src.poker.environment import RANK_NAMES
    high = RANK_NAMES[r1]
    low = RANK_NAMES[r2]

    if r1 == r2:
        return f"{high}{low}"
    elif c1.suit == c2.suit:
        return f"{high}{low}s"
    else:
        return f"{high}{low}o"


# Preflop tiers
TIER_1 = {"AA", "KK", "QQ", "AKs"}
TIER_2 = {"JJ", "TT", "AKo", "AQs", "AQo", "AJs", "KQs"}
TIER_3 = {
    "99", "88", "ATs", "AJo", "ATo", "KJs", "KQo", "KTs",
    "QJs", "QTs", "JTs",
}
TIER_4 = {
    "77", "66", "55", "44", "33", "22",
    "A9s", "A8s", "A7s", "A6s", "A5s", "A4s", "A3s", "A2s",
    "K9s", "Q9s", "J9s", "T9s", "98s", "87s", "76s", "65s", "54s",
    "KJo", "QJo", "JTo",
}


def preflop_tier(hole: List[Card]) -> int:
    """Return hand tier 1-5 (1 = best, 5 = trash)."""
    key = _hand_key(hole)
    if key in TIER_1:
        return 1
    if key in TIER_2:
        return 2
    if key in TIER_3:
        return 3
    if key in TIER_4:
        return 4
    return 5


# ── Postflop Hand Strength ────────────────────────────────────────────

def postflop_strength(hole: List[Card], board: List[Card]) -> Tuple[str, float]:
    """
    Categorize postflop hand strength.

    Returns:
        (category, strength_score)
        category: "monster", "strong", "medium", "weak", "nothing"
        strength_score: 0.0 to 1.0
    """
    all_cards = hole + board
    score = HandEvaluator.evaluate(all_cards)
    hand_rank = HandRank(score[0])

    if hand_rank >= HandRank.STRAIGHT:
        return ("monster", 0.95)
    if hand_rank == HandRank.THREE_OF_A_KIND:
        return ("strong", 0.80)
    if hand_rank == HandRank.TWO_PAIR:
        return ("strong", 0.75)
    if hand_rank == HandRank.ONE_PAIR:
        # Distinguish top pair, middle pair, bottom pair
        pair_rank = score[1]  # rank of the pair
        board_ranks = sorted([c.rank for c in board], reverse=True)
        if pair_rank >= board_ranks[0]:
            # Top pair or overpair
            return ("medium", 0.60)
        elif len(board_ranks) > 1 and pair_rank >= board_ranks[1]:
            return ("weak", 0.40)
        else:
            return ("weak", 0.30)

    return ("nothing", 0.15)


def has_flush_draw(hole: List[Card], board: List[Card]) -> bool:
    """Check if hero has 4 cards to a flush."""
    all_cards = hole + board
    suit_counts = {}
    for c in all_cards:
        suit_counts[c.suit] = suit_counts.get(c.suit, 0) + 1
    for suit, count in suit_counts.items():
        if count == 4 and any(c.suit == suit for c in hole):
            return True
    return False


def has_straight_draw(hole: List[Card], board: List[Card]) -> bool:
    """Check if hero has an open-ended straight draw (4 consecutive ranks)."""
    all_ranks = sorted(set(c.rank for c in hole + board))
    for i in range(len(all_ranks) - 3):
        window = all_ranks[i:i+4]
        if window[-1] - window[0] == 3:
            # Check that at least one hole card is in the window
            hole_ranks = {c.rank for c in hole}
            if hole_ranks & set(window):
                return True
    return False


# ── Decision Logic ────────────────────────────────────────────────────

class HeuristicPokerBot:
    """
    TAG heuristic poker bot.

    Decision logic:
    - Preflop: tier-based open/call/fold
    - Postflop: hand strength + pot odds + draws → bet/check/call/fold
    """

    def decide(self, state: GameState) -> Tuple[str, float]:
        """
        Returns (action, amount) where action is one of:
        "fold", "check", "call", "raise"
        """
        if state.street == "preflop":
            return self._preflop_decision(state)
        else:
            return self._postflop_decision(state)

    def _preflop_decision(self, state: GameState) -> Tuple[str, float]:
        tier = preflop_tier(state.hero_hole)
        to_call = state.to_call
        bb = state.big_blind
        pot = state.pot

        # Position advantage: later positions play looser
        late_position = state.hero_position in ("CO", "BTN")
        middle_position = state.hero_position in ("MP",)

        # Facing a raise?
        facing_raise = to_call > bb

        if tier == 1:
            # Premium: always raise/reraise
            raise_size = max(3 * bb, 3 * to_call)
            return ("raise", min(raise_size, self._hero_stack(state)))

        if tier == 2:
            if facing_raise:
                # Call raises, 3-bet sometimes
                if to_call <= 4 * bb:
                    return ("call", to_call)
                else:
                    return ("call", to_call)
            # Open raise
            return ("raise", min(3 * bb, self._hero_stack(state)))

        if tier == 3:
            if facing_raise:
                if to_call <= 3 * bb:
                    return ("call", to_call)
                return ("fold", 0)
            if late_position or middle_position:
                return ("raise", min(2.5 * bb, self._hero_stack(state)))
            return ("call", to_call) if to_call <= bb else ("fold", 0)

        if tier == 4:
            if facing_raise:
                # Only call small raises in position
                if late_position and to_call <= 2.5 * bb:
                    return ("call", to_call)
                return ("fold", 0)
            if late_position:
                return ("raise", min(2.5 * bb, self._hero_stack(state)))
            return ("fold", 0) if to_call > bb else ("check", 0)

        # Tier 5: trash
        if to_call == 0:
            return ("check", 0)
        return ("fold", 0)

    def _postflop_decision(self, state: GameState) -> Tuple[str, float]:
        category, strength = postflop_strength(state.hero_hole, state.board)
        to_call = state.to_call
        pot = state.pot
        pot_odds = state.pot_odds

        # Check for draws
        flush_draw = has_flush_draw(state.hero_hole, state.board)
        straight_draw = has_straight_draw(state.hero_hole, state.board)
        has_draw = flush_draw or straight_draw
        draw_equity = 0.0
        if flush_draw:
            draw_equity += 0.35  # ~9 outs ≈ 35% on flop
        if straight_draw:
            draw_equity += 0.17  # ~4-8 outs

        if category == "monster":
            # Bet big for value
            bet_size = pot * 0.75
            if to_call > 0:
                # Raise
                return ("raise", min(to_call + pot * 0.75, self._hero_stack(state)))
            return ("raise", min(bet_size, self._hero_stack(state)))

        if category == "strong":
            if to_call > 0:
                if pot_odds < strength:
                    return ("call", to_call)
                return ("fold", 0)
            # Bet for value
            bet_size = pot * 0.5
            return ("raise", min(bet_size, self._hero_stack(state)))

        if category == "medium":
            if to_call > 0:
                if pot_odds < strength:
                    return ("call", to_call)
                return ("fold", 0)
            # Small bet or check
            bet_size = pot * 0.33
            return ("raise", min(bet_size, self._hero_stack(state)))

        if category == "weak":
            if to_call > 0:
                # Only call with draws and good pot odds
                effective_strength = strength + draw_equity
                if pot_odds < effective_strength and to_call < pot * 0.3:
                    return ("call", to_call)
                return ("fold", 0)
            if has_draw:
                # Semi-bluff
                return ("raise", min(pot * 0.5, self._hero_stack(state)))
            return ("check", 0)

        # Nothing
        if to_call > 0:
            if has_draw and pot_odds < draw_equity:
                return ("call", to_call)
            return ("fold", 0)
        if has_draw:
            # Semi-bluff occasionally
            return ("raise", min(pot * 0.5, self._hero_stack(state)))
        return ("check", 0)

    def _hero_stack(self, state: GameState) -> float:
        hero = next(p for p in state.players if p.position == state.hero_position)
        return hero.stack

    def explain(self, state: GameState) -> str:
        """
        Return a text explanation of the decision — useful for
        generating training data and understanding the reasoning.
        """
        action, amount = self.decide(state)
        lines = []

        if state.street == "preflop":
            tier = preflop_tier(state.hero_hole)
            key = _hand_key(state.hero_hole)
            lines.append(f"Hand: {key} (Tier {tier})")
            lines.append(f"Position: {state.hero_position}")
            lines.append(f"To call: ${state.to_call:.0f}")
        else:
            cat, strength = postflop_strength(state.hero_hole, state.board)
            score = HandEvaluator.evaluate(state.hero_hole + state.board)
            hand_name = HandEvaluator.hand_name(score)
            lines.append(f"Made hand: {hand_name} ({cat}, strength={strength:.2f})")
            lines.append(f"Pot: ${state.pot:.0f}, To call: ${state.to_call:.0f}")
            if state.to_call > 0:
                lines.append(f"Pot odds: {state.pot_odds:.1%}")
            fd = has_flush_draw(state.hero_hole, state.board)
            sd = has_straight_draw(state.hero_hole, state.board)
            if fd:
                lines.append("Flush draw: yes")
            if sd:
                lines.append("Straight draw: yes")

        if amount > 0:
            lines.append(f"Decision: {action} ${amount:.0f}")
        else:
            lines.append(f"Decision: {action}")

        return "\n".join(lines)
