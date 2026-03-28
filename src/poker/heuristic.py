"""
Heuristic poker bot: tight-aggressive (TAG) strategy with opponent modeling.

Three-step reasoning flow (retrieve → compute → decide):
  1. RETRIEVE: Parse hand history → compute per-opponent stats
  2. COMPUTE:  Hand strength + pot odds + opponent adjustments
  3. DECIDE:   Final action incorporating all information

This serves as the baseline for trajectory collection. The LLM agent
will be trained to match and then surpass this strategy via RL.
"""

import re
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional
from src.poker.environment import (
    Card, GameState, HandEvaluator, HandRank, HandRecord,
    OpponentProfile, RANK_FROM_CHAR,
)


# ── Preflop Hand Rankings ─────────────────────────────────────────────

def _hand_key(hole: List[Card]) -> str:
    """
    Convert hole cards to a canonical key like 'AKs', 'QJo', 'TT'.
    """
    from src.poker.environment import RANK_NAMES
    c1, c2 = hole
    r1, r2 = c1.rank, c2.rank
    if r1 < r2:
        r1, r2 = r2, r1
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


# ── Step 1: RETRIEVE — Parse History → Opponent Stats ─────────────────

@dataclass
class OpponentStats:
    """Stats computed from hand history for a single opponent."""
    position: str
    hands_seen: int = 0
    vpip_hands: int = 0        # hands where they voluntarily put money in
    pfr_hands: int = 0         # hands where they raised preflop
    postflop_bets: int = 0     # bets + raises postflop
    postflop_calls: int = 0    # calls postflop
    cbet_opportunities: int = 0  # times they were preflop raiser and saw flop
    cbet_made: int = 0           # times they continuation bet
    fold_to_cbet_opps: int = 0   # times they faced a cbet
    fold_to_cbet_did: int = 0    # times they folded to a cbet
    three_bet_opps: int = 0      # times they could 3-bet
    three_bet_did: int = 0       # times they 3-bet

    @property
    def vpip(self) -> float:
        return self.vpip_hands / max(self.hands_seen, 1)

    @property
    def pfr(self) -> float:
        return self.pfr_hands / max(self.hands_seen, 1)

    @property
    def aggression(self) -> float:
        return self.postflop_bets / max(self.postflop_calls, 1)

    @property
    def fold_to_cbet(self) -> float:
        return self.fold_to_cbet_did / max(self.fold_to_cbet_opps, 1)

    @property
    def three_bet_pct(self) -> float:
        return self.three_bet_did / max(self.three_bet_opps, 1)

    def summary(self) -> str:
        return (
            f"{self.position}: {self.hands_seen} hands | "
            f"VPIP {self.vpip:.0%} | PFR {self.pfr:.0%} | "
            f"Agg {self.aggression:.1f} | "
            f"Fold-to-CBet {self.fold_to_cbet:.0%}"
        )


def parse_opponent_stats(history: List[HandRecord]) -> Dict[str, OpponentStats]:
    """
    RETRIEVE step: parse hand history records and compute per-opponent stats.

    This is the key function the LLM must learn to replicate — it extracts
    actionable intelligence from raw hand history (long context).
    """
    stats: Dict[str, OpponentStats] = {}

    for record in history:
        for pos in record.players_involved:
            if pos not in stats:
                stats[pos] = OpponentStats(position=pos)
            stats[pos].hands_seen += 1

        # Preflop analysis
        preflop_actions = record.actions_by_street.get("preflop", [])
        first_raiser = None
        raise_count = 0

        for action in preflop_actions:
            pos = action.player
            if pos not in stats:
                continue

            if action.action in ("call", "raise", "bet"):
                stats[pos].vpip_hands += 1

            if action.action == "raise":
                raise_count += 1
                if raise_count == 1:
                    first_raiser = pos
                    stats[pos].pfr_hands += 1
                elif raise_count >= 2:
                    stats[pos].three_bet_did += 1

            # Everyone who sees a raise has a 3-bet opportunity
            if action.action == "raise" and raise_count == 1:
                # Mark subsequent players as having 3-bet opportunity
                for later_action in preflop_actions:
                    if later_action.player != pos and later_action.player in stats:
                        stats[later_action.player].three_bet_opps += 1

        # Postflop analysis
        for street in ["flop", "turn", "river"]:
            street_actions = record.actions_by_street.get(street, [])

            for i, action in enumerate(street_actions):
                pos = action.player
                if pos not in stats:
                    continue

                if action.action in ("bet", "raise"):
                    stats[pos].postflop_bets += 1

                    # C-bet detection: first bet on flop by preflop raiser
                    if street == "flop" and i == 0 and pos == first_raiser:
                        stats[pos].cbet_made += 1

                if action.action == "call":
                    stats[pos].postflop_calls += 1

                if action.action == "fold":
                    # Fold to c-bet: folding on flop to the preflop raiser's bet
                    if street == "flop" and first_raiser:
                        flop_bettors = [a.player for a in street_actions if a.action in ("bet", "raise")]
                        if first_raiser in flop_bettors:
                            stats[pos].fold_to_cbet_opps += 1
                            stats[pos].fold_to_cbet_did += 1

            # Track c-bet opportunities
            if street == "flop" and first_raiser and first_raiser in stats:
                if first_raiser in [a.player for a in street_actions]:
                    stats[first_raiser].cbet_opportunities += 1

        # Track fold-to-cbet opportunities for players who didn't fold
        if "flop" in record.actions_by_street and first_raiser:
            flop_actions = record.actions_by_street["flop"]
            flop_bettors = [a.player for a in flop_actions if a.action in ("bet", "raise")]
            if first_raiser in flop_bettors:
                for action in flop_actions:
                    if action.player != first_raiser and action.player in stats:
                        if action.action != "fold":
                            stats[action.player].fold_to_cbet_opps += 1

    return stats


# ── Step 2: COMPUTE — Hand Strength + Draws ───────────────────────────

def postflop_strength(hole: List[Card], board: List[Card]) -> Tuple[str, float]:
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
        pair_rank = score[1]
        board_ranks = sorted([c.rank for c in board], reverse=True)
        if pair_rank >= board_ranks[0]:
            return ("medium", 0.60)
        elif len(board_ranks) > 1 and pair_rank >= board_ranks[1]:
            return ("weak", 0.40)
        else:
            return ("weak", 0.30)
    return ("nothing", 0.15)


def has_flush_draw(hole: List[Card], board: List[Card]) -> bool:
    suit_counts = {}
    for c in hole + board:
        suit_counts[c.suit] = suit_counts.get(c.suit, 0) + 1
    for suit, count in suit_counts.items():
        if count == 4 and any(c.suit == suit for c in hole):
            return True
    return False


def has_straight_draw(hole: List[Card], board: List[Card]) -> bool:
    all_ranks = sorted(set(c.rank for c in hole + board))
    for i in range(len(all_ranks) - 3):
        window = all_ranks[i:i+4]
        if window[-1] - window[0] == 3:
            hole_ranks = {c.rank for c in hole}
            if hole_ranks & set(window):
                return True
    return False


# ── Step 3: DECIDE — Adjusted Strategy ────────────────────────────────

@dataclass
class ReasoningTrace:
    """Full trace of the 3-step reasoning process."""
    # Step 1: Retrieve
    opponent_stats: Dict[str, OpponentStats]
    active_opponent_summaries: List[str]

    # Step 2: Compute
    hand_key: str = ""
    preflop_tier: int = 0
    hand_category: str = ""
    hand_strength: float = 0.0
    hand_name: str = ""
    pot_odds: float = 0.0
    flush_draw: bool = False
    straight_draw: bool = False

    # Adjustments from opponent stats
    adjustments: List[str] = None

    # Step 3: Decide
    base_action: str = ""
    base_amount: float = 0.0
    final_action: str = ""
    final_amount: float = 0.0
    adjustment_applied: str = ""

    def __post_init__(self):
        if self.adjustments is None:
            self.adjustments = []

    def format(self) -> str:
        lines = []
        lines.append("=== STEP 1: RETRIEVE (opponent stats from history) ===")
        if self.active_opponent_summaries:
            for s in self.active_opponent_summaries:
                lines.append(f"  {s}")
        else:
            lines.append("  No history available")

        lines.append("")
        lines.append("=== STEP 2: COMPUTE (hand analysis) ===")
        if self.preflop_tier > 0:
            lines.append(f"  Hand: {self.hand_key} (Tier {self.preflop_tier})")
        if self.hand_name:
            lines.append(f"  Made hand: {self.hand_name} ({self.hand_category}, strength={self.hand_strength:.2f})")
        if self.pot_odds > 0:
            lines.append(f"  Pot odds: {self.pot_odds:.1%}")
        if self.flush_draw:
            lines.append(f"  Flush draw: yes")
        if self.straight_draw:
            lines.append(f"  Straight draw: yes")
        if self.adjustments:
            lines.append(f"  Opponent adjustments:")
            for adj in self.adjustments:
                lines.append(f"    - {adj}")

        lines.append("")
        lines.append("=== STEP 3: DECIDE ===")
        base = f"{self.base_action} ${self.base_amount:.0f}" if self.base_amount else self.base_action
        final = f"{self.final_action} ${self.final_amount:.0f}" if self.final_amount else self.final_action
        lines.append(f"  Base decision: {base}")
        if self.adjustment_applied:
            lines.append(f"  Adjustment: {self.adjustment_applied}")
        lines.append(f"  Final decision: {final}")

        return "\n".join(lines)


class HeuristicPokerBot:
    """
    TAG heuristic poker bot with opponent modeling.

    Uses a 3-step retrieve→compute→decide flow that the LLM
    will learn to replicate and eventually surpass.
    """

    def decide(self, state: GameState) -> Tuple[str, float]:
        """Returns (action, amount)."""
        trace = self.decide_with_reasoning(state)
        return trace.final_action, trace.final_amount

    def decide_with_reasoning(self, state: GameState) -> ReasoningTrace:
        """Full decision with reasoning trace for trajectory data."""

        # ── Step 1: RETRIEVE ──
        opp_stats = parse_opponent_stats(state.hand_history)
        active_positions = [p.position for p in state.active_opponents]
        active_summaries = [
            opp_stats[pos].summary()
            for pos in active_positions if pos in opp_stats
        ]

        trace = ReasoningTrace(
            opponent_stats=opp_stats,
            active_opponent_summaries=active_summaries,
        )

        # ── Step 2: COMPUTE ──
        trace.pot_odds = state.pot_odds

        if state.street == "preflop":
            trace.hand_key = _hand_key(state.hero_hole)
            trace.preflop_tier = preflop_tier(state.hero_hole)
            base_action, base_amount = self._preflop_base(state, trace)
        else:
            cat, strength = postflop_strength(state.hero_hole, state.board)
            trace.hand_category = cat
            trace.hand_strength = strength
            score = HandEvaluator.evaluate(state.hero_hole + state.board)
            trace.hand_name = HandEvaluator.hand_name(score)
            trace.flush_draw = has_flush_draw(state.hero_hole, state.board)
            trace.straight_draw = has_straight_draw(state.hero_hole, state.board)
            base_action, base_amount = self._postflop_base(state, trace)

        trace.base_action = base_action
        trace.base_amount = base_amount

        # ── Step 3: DECIDE (adjust based on opponent stats) ──
        final_action, final_amount, adj_note = self._apply_adjustments(
            state, trace, base_action, base_amount, opp_stats
        )
        trace.final_action = final_action
        trace.final_amount = final_amount
        trace.adjustment_applied = adj_note

        return trace

    # ── Preflop base strategy ──

    def _preflop_base(self, state: GameState, trace: ReasoningTrace) -> Tuple[str, float]:
        tier = trace.preflop_tier
        to_call = state.to_call
        bb = state.big_blind

        late_position = state.hero_position in ("CO", "BTN")
        middle_position = state.hero_position in ("MP",)
        facing_raise = to_call > bb

        if tier == 1:
            return ("raise", min(max(3 * bb, 3 * to_call), self._hero_stack(state)))
        if tier == 2:
            if facing_raise:
                return ("call", to_call)
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
                if late_position and to_call <= 2.5 * bb:
                    return ("call", to_call)
                return ("fold", 0)
            if late_position:
                return ("raise", min(2.5 * bb, self._hero_stack(state)))
            return ("fold", 0) if to_call > bb else ("check", 0)
        # Tier 5
        if to_call == 0:
            return ("check", 0)
        return ("fold", 0)

    # ── Postflop base strategy ──

    def _postflop_base(self, state: GameState, trace: ReasoningTrace) -> Tuple[str, float]:
        category = trace.hand_category
        strength = trace.hand_strength
        to_call = state.to_call
        pot = state.pot
        pot_odds = state.pot_odds
        has_draw = trace.flush_draw or trace.straight_draw

        draw_equity = 0.0
        if trace.flush_draw:
            draw_equity += 0.35
        if trace.straight_draw:
            draw_equity += 0.17

        if category == "monster":
            if to_call > 0:
                return ("raise", min(to_call + pot * 0.75, self._hero_stack(state)))
            return ("raise", min(pot * 0.75, self._hero_stack(state)))

        if category == "strong":
            if to_call > 0:
                if pot_odds < strength:
                    return ("call", to_call)
                return ("fold", 0)
            return ("raise", min(pot * 0.5, self._hero_stack(state)))

        if category == "medium":
            if to_call > 0:
                if pot_odds < strength:
                    return ("call", to_call)
                return ("fold", 0)
            return ("raise", min(pot * 0.33, self._hero_stack(state)))

        if category == "weak":
            if to_call > 0:
                effective = strength + draw_equity
                if pot_odds < effective and to_call < pot * 0.3:
                    return ("call", to_call)
                return ("fold", 0)
            if has_draw:
                return ("raise", min(pot * 0.5, self._hero_stack(state)))
            return ("check", 0)

        # Nothing
        if to_call > 0:
            if has_draw and pot_odds < draw_equity:
                return ("call", to_call)
            return ("fold", 0)
        if has_draw:
            return ("raise", min(pot * 0.5, self._hero_stack(state)))
        return ("check", 0)

    # ── Opponent-adjusted decisions ──

    def _apply_adjustments(
        self,
        state: GameState,
        trace: ReasoningTrace,
        base_action: str,
        base_amount: float,
        opp_stats: Dict[str, OpponentStats],
    ) -> Tuple[str, float, str]:
        """
        Adjust the base decision using opponent stats from history.
        Returns (action, amount, adjustment_note).
        """
        action, amount = base_action, base_amount
        note = "no adjustment (insufficient history)"

        # Get the primary villain (the one who bet, or the most active opponent)
        villain_pos = self._primary_villain(state)
        if not villain_pos or villain_pos not in opp_stats:
            return action, amount, note

        villain = opp_stats[villain_pos]

        # Need enough data to make adjustments (at least 5 hands)
        if villain.hands_seen < 5:
            return action, amount, note

        adjustments = trace.adjustments

        # ── Preflop adjustments ──
        if state.street == "preflop":
            # Against a fish (loose-passive): widen value range, raise more for value
            if villain.vpip > 0.40 and villain.aggression < 1.0:
                adjustments.append(f"{villain_pos} is loose-passive (VPIP={villain.vpip:.0%})")
                if trace.preflop_tier == 3 and base_action == "fold":
                    action, amount = "call", state.to_call
                    note = f"calling vs fish {villain_pos} (VPIP={villain.vpip:.0%}, normally would fold)"
                elif trace.preflop_tier <= 2 and base_action == "call":
                    # Raise for value against a calling station
                    action = "raise"
                    amount = min(4 * state.big_blind, self._hero_stack(state))
                    note = f"raising for value vs fish {villain_pos} (VPIP={villain.vpip:.0%})"

            # Against a maniac (loose-aggressive): tighten up, let them bluff
            elif villain.vpip > 0.40 and villain.aggression > 2.5:
                adjustments.append(f"{villain_pos} is a maniac (VPIP={villain.vpip:.0%}, Agg={villain.aggression:.1f})")
                if trace.preflop_tier <= 2 and base_action == "call":
                    # Trap with premiums
                    action = "call"
                    note = f"trapping maniac {villain_pos} (Agg={villain.aggression:.1f})"

            # Against a rock (tight): respect their raises
            elif villain.vpip < 0.18:
                adjustments.append(f"{villain_pos} is tight (VPIP={villain.vpip:.0%})")
                if state.to_call > 0 and trace.preflop_tier >= 3:
                    action, amount = "fold", 0
                    note = f"folding to tight player {villain_pos} raise (VPIP={villain.vpip:.0%})"

            # Exploit high fold-to-3bet
            if villain.three_bet_pct < 0.04 and villain.pfr > 0.15:
                adjustments.append(f"{villain_pos} rarely 3-bets ({villain.three_bet_pct:.0%})")

        # ── Postflop adjustments ──
        else:
            # Against high fold-to-cbet: bluff more
            if villain.fold_to_cbet > 0.60 and villain.fold_to_cbet_opps >= 3:
                adjustments.append(f"{villain_pos} folds to c-bets {villain.fold_to_cbet:.0%} of the time")
                if base_action == "check" and trace.hand_category in ("weak", "nothing"):
                    action = "raise"
                    amount = min(state.pot * 0.5, self._hero_stack(state))
                    note = f"bluffing vs {villain_pos} (fold-to-cbet={villain.fold_to_cbet:.0%})"

            # Against passive player who bets: they likely have it
            if villain.aggression < 0.8 and state.to_call > 0:
                adjustments.append(f"{villain_pos} is passive (Agg={villain.aggression:.1f}), bet likely means strength")
                if trace.hand_category == "medium" and base_action == "call":
                    action, amount = "fold", 0
                    note = f"folding to passive player {villain_pos} bet (Agg={villain.aggression:.1f}, they usually have it)"

            # Against aggressive player: call down wider
            if villain.aggression > 2.5 and state.to_call > 0:
                adjustments.append(f"{villain_pos} is very aggressive (Agg={villain.aggression:.1f}), could be bluffing")
                if trace.hand_category in ("medium", "weak") and base_action == "fold":
                    if state.pot_odds < 0.35:
                        action = "call"
                        amount = state.to_call
                        note = f"calling down vs aggressive {villain_pos} (Agg={villain.aggression:.1f})"

            # Against loose player postflop: value bet thinner
            if villain.vpip > 0.40 and base_action == "check":
                if trace.hand_category == "medium":
                    adjustments.append(f"{villain_pos} is loose (VPIP={villain.vpip:.0%}), value bet thinner")
                    action = "raise"
                    amount = min(state.pot * 0.4, self._hero_stack(state))
                    note = f"thin value bet vs loose {villain_pos} (VPIP={villain.vpip:.0%})"

        if not adjustments:
            note = "no exploitable patterns found"

        return action, amount, note

    # ── Helpers ──

    def _primary_villain(self, state: GameState) -> Optional[str]:
        """Find the main opponent in this hand (last bettor or most active)."""
        # If someone bet, that's the villain
        for action in reversed(state.actions):
            if action.action in ("bet", "raise") and action.player != state.hero_position:
                return action.player
        # Otherwise pick first active opponent
        for p in state.players:
            if p.position != state.hero_position and p.is_active:
                return p.position
        return None

    def _hero_stack(self, state: GameState) -> float:
        hero = next(p for p in state.players if p.position == state.hero_position)
        return hero.stack

    def explain(self, state: GameState) -> str:
        """Return full reasoning trace as text."""
        trace = self.decide_with_reasoning(state)
        return trace.format()
