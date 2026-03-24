"""
Poker environment: cards, deck, hand evaluation, and game state.

Supports Texas Hold'em with standard hand rankings.
"""

import random
from dataclasses import dataclass, field
from enum import IntEnum
from itertools import combinations
from typing import List, Tuple, Optional


# ── Cards ─────────────────────────────────────────────────────────────

SUITS = ["h", "d", "c", "s"]
SUIT_NAMES = {"h": "hearts", "d": "diamonds", "c": "clubs", "s": "spades"}
RANK_NAMES = {
    2: "2", 3: "3", 4: "4", 5: "5", 6: "6", 7: "7", 8: "8",
    9: "9", 10: "T", 11: "J", 12: "Q", 13: "K", 14: "A",
}
RANK_FROM_CHAR = {v: k for k, v in RANK_NAMES.items()}


@dataclass(frozen=True, order=True)
class Card:
    rank: int   # 2–14 (14 = Ace)
    suit: str   # h, d, c, s

    def __str__(self):
        return f"{RANK_NAMES[self.rank]}{self.suit}"

    def __repr__(self):
        return str(self)

    @classmethod
    def from_str(cls, s: str) -> "Card":
        """Parse 'Ah', 'Td', '2c', etc."""
        rank_char, suit = s[0], s[1]
        return cls(rank=RANK_FROM_CHAR[rank_char], suit=suit)


class Deck:
    def __init__(self):
        self.cards = [Card(r, s) for r in range(2, 15) for s in SUITS]
        random.shuffle(self.cards)

    def deal(self, n: int = 1) -> List[Card]:
        dealt = self.cards[:n]
        self.cards = self.cards[n:]
        return dealt

    def remove(self, cards: List[Card]):
        """Remove specific cards (for setting up known states)."""
        for c in cards:
            if c in self.cards:
                self.cards.remove(c)


# ── Hand Evaluation ───────────────────────────────────────────────────

class HandRank(IntEnum):
    HIGH_CARD = 0
    ONE_PAIR = 1
    TWO_PAIR = 2
    THREE_OF_A_KIND = 3
    STRAIGHT = 4
    FLUSH = 5
    FULL_HOUSE = 6
    FOUR_OF_A_KIND = 7
    STRAIGHT_FLUSH = 8


HAND_RANK_NAMES = {
    HandRank.HIGH_CARD: "High Card",
    HandRank.ONE_PAIR: "One Pair",
    HandRank.TWO_PAIR: "Two Pair",
    HandRank.THREE_OF_A_KIND: "Three of a Kind",
    HandRank.STRAIGHT: "Straight",
    HandRank.FLUSH: "Flush",
    HandRank.FULL_HOUSE: "Full House",
    HandRank.FOUR_OF_A_KIND: "Four of a Kind",
    HandRank.STRAIGHT_FLUSH: "Straight Flush",
}


def _eval_5(cards: List[Card]) -> Tuple[int, ...]:
    """
    Evaluate a 5-card hand. Returns a tuple for comparison:
    (hand_rank, *kickers) — higher is better.
    """
    ranks = sorted([c.rank for c in cards], reverse=True)
    suits = [c.suit for c in cards]

    is_flush = len(set(suits)) == 1

    # Check straight (including A-2-3-4-5 wheel)
    unique_ranks = sorted(set(ranks), reverse=True)
    is_straight = False
    straight_high = 0
    if len(unique_ranks) == 5:
        if unique_ranks[0] - unique_ranks[4] == 4:
            is_straight = True
            straight_high = unique_ranks[0]
        elif unique_ranks == [14, 5, 4, 3, 2]:  # wheel
            is_straight = True
            straight_high = 5

    # Count rank frequencies
    freq = {}
    for r in ranks:
        freq[r] = freq.get(r, 0) + 1
    counts = sorted(freq.values(), reverse=True)
    # Sort ranks by (frequency, rank) descending for kicker ordering
    ranked_by_freq = sorted(freq.keys(), key=lambda r: (freq[r], r), reverse=True)

    if is_straight and is_flush:
        return (HandRank.STRAIGHT_FLUSH, straight_high)
    if counts == [4, 1]:
        return (HandRank.FOUR_OF_A_KIND, *ranked_by_freq)
    if counts == [3, 2]:
        return (HandRank.FULL_HOUSE, *ranked_by_freq)
    if is_flush:
        return (HandRank.FLUSH, *ranks)
    if is_straight:
        return (HandRank.STRAIGHT, straight_high)
    if counts == [3, 1, 1]:
        return (HandRank.THREE_OF_A_KIND, *ranked_by_freq)
    if counts == [2, 2, 1]:
        return (HandRank.TWO_PAIR, *ranked_by_freq)
    if counts == [2, 1, 1, 1]:
        return (HandRank.ONE_PAIR, *ranked_by_freq)
    return (HandRank.HIGH_CARD, *ranks)


class HandEvaluator:
    """Evaluate the best 5-card hand from up to 7 cards."""

    @staticmethod
    def evaluate(cards: List[Card]) -> Tuple[int, ...]:
        """Returns comparable tuple — higher is better."""
        if len(cards) < 5:
            raise ValueError(f"Need at least 5 cards, got {len(cards)}")
        if len(cards) == 5:
            return _eval_5(cards)
        # Best of all C(n, 5) combos
        return max(_eval_5(list(combo)) for combo in combinations(cards, 5))

    @staticmethod
    def hand_name(score: Tuple[int, ...]) -> str:
        return HAND_RANK_NAMES[HandRank(score[0])]

    @staticmethod
    def equity_estimate(hole: List[Card], board: List[Card], num_simulations: int = 1000) -> float:
        """
        Monte Carlo equity estimate: fraction of times our hand wins
        against a random opponent hand over num_simulations trials.
        """
        wins = 0
        ties = 0
        known = set(hole + board)

        for _ in range(num_simulations):
            deck = [Card(r, s) for r in range(2, 15) for s in SUITS
                    if Card(r, s) not in known]
            random.shuffle(deck)

            # Deal opponent 2 cards
            opp_hole = deck[:2]
            remaining = deck[2:]

            # Complete the board to 5 cards
            cards_needed = 5 - len(board)
            full_board = board + remaining[:cards_needed]

            our_score = HandEvaluator.evaluate(hole + full_board)
            opp_score = HandEvaluator.evaluate(opp_hole + full_board)

            if our_score > opp_score:
                wins += 1
            elif our_score == opp_score:
                ties += 1

        return (wins + ties * 0.5) / num_simulations


# ── Game State ────────────────────────────────────────────────────────

POSITIONS_6MAX = ["UTG", "MP", "CO", "BTN", "SB", "BB"]

@dataclass
class PlayerState:
    position: str
    stack: float
    is_active: bool = True  # still in the hand
    total_bet: float = 0.0  # total chips put in this hand

@dataclass
class Action:
    player: str      # position name
    action: str      # "fold", "check", "call", "raise", "bet"
    amount: float = 0.0

    def __str__(self):
        if self.action in ("fold", "check"):
            return f"{self.player} {self.action}s"
        elif self.action == "call":
            return f"{self.player} calls ${self.amount:.0f}"
        elif self.action in ("raise", "bet"):
            return f"{self.player} {'raises' if self.action == 'raise' else 'bets'} ${self.amount:.0f}"
        return f"{self.player} {self.action} ${self.amount:.0f}"


@dataclass
class GameState:
    """Full state of a poker hand."""
    # Table info
    num_players: int = 6
    small_blind: float = 1.0
    big_blind: float = 2.0

    # Player info
    hero_position: str = "BTN"
    hero_hole: List[Card] = field(default_factory=list)
    players: List[PlayerState] = field(default_factory=list)

    # Board
    board: List[Card] = field(default_factory=list)
    street: str = "preflop"  # preflop, flop, turn, river

    # Betting
    pot: float = 0.0
    current_bet: float = 0.0
    hero_bet_this_round: float = 0.0
    actions: List[Action] = field(default_factory=list)

    # History (for long context)
    hand_history: List[str] = field(default_factory=list)

    @property
    def to_call(self) -> float:
        return max(0, self.current_bet - self.hero_bet_this_round)

    @property
    def pot_odds(self) -> float:
        """Pot odds as a ratio: call_amount / (pot + call_amount)."""
        call_amt = self.to_call
        if call_amt == 0:
            return 0.0
        return call_amt / (self.pot + call_amt)

    def format_context(self) -> str:
        """Format the game state as a text context string for the LLM."""
        lines = []
        lines.append("=== POKER HAND ===")
        lines.append(f"Table: {self.num_players}-max No-Limit Hold'em")
        lines.append(f"Blinds: ${self.small_blind:.0f}/${self.big_blind:.0f}")
        lines.append("")

        # Hero info
        hero_player = next(p for p in self.players if p.position == self.hero_position)
        lines.append(f"Your Position: {self.hero_position}")
        lines.append(f"Your Stack: ${hero_player.stack:.0f}")
        lines.append(f"Your Hand: {self.hero_hole[0]} {self.hero_hole[1]}")
        lines.append("")

        # Opponents
        lines.append("Opponents:")
        for p in self.players:
            if p.position != self.hero_position:
                status = "active" if p.is_active else "folded"
                lines.append(f"  {p.position}: ${p.stack:.0f} ({status})")
        lines.append("")

        # Board
        if self.board:
            board_str = " ".join(str(c) for c in self.board)
            lines.append(f"Community Cards: {board_str} ({self.street.capitalize()})")
        else:
            lines.append(f"Street: Preflop (no community cards)")
        lines.append("")

        # Pot and betting
        lines.append(f"Pot: ${self.pot:.0f}")
        if self.to_call > 0:
            lines.append(f"To Call: ${self.to_call:.0f}")
            lines.append(f"Pot Odds: {self.pot_odds:.1%}")
        lines.append("")

        # Action history
        if self.actions:
            lines.append("Betting History:")
            current_street = "preflop"
            for a in self.actions:
                lines.append(f"  {a}")
            lines.append("")

        # Multi-hand history (long context)
        if self.hand_history:
            lines.append("=== PREVIOUS HANDS ===")
            for h in self.hand_history:
                lines.append(h)
            lines.append("")

        return "\n".join(lines)
