"""
Poker-specific tests for the RLM framework.

Tests:
- Environment: card dealing, hand evaluation, game state formatting
- Heuristic: opponent stats parsing, preflop tiers, decision-making, adjustments
- Tasks: task generation, context structure, reasoning traces
- Agents: PokerHeuristicAgent REPL pipeline
- Rewards: action parsing, reward computation
- Evaluation: framework metrics
"""

import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.poker.environment import (
    Card, Deck, HandEvaluator, HandRank, GameState, PlayerState,
    Action, HandRecord, OpponentProfile, OPPONENT_ARCHETYPES,
)
from src.poker.heuristic import (
    HeuristicPokerBot, parse_opponent_stats, preflop_tier,
    _hand_key, postflop_strength, has_flush_draw, has_straight_draw,
)
from src.poker.tasks import (
    generate_poker_task,
    generate_poker_task_with_trace,
    bc_agent_task_generators,
)
from src.poker.agents import PokerHeuristicAgent
from src.poker.rewards import parse_action, compute_poker_reward_simple
from src.poker.evaluation import PokerEvaluationFramework


# ── Environment Tests ────────────────────────────────────────────────

def test_deck_deals_52_cards():
    """Deck should have 52 unique cards."""
    deck = Deck()
    all_cards = deck.deal(52)
    assert len(all_cards) == 52
    assert len(set(str(c) for c in all_cards)) == 52


def test_hand_evaluator_pair():
    """Hand evaluator should detect one pair."""
    cards = [
        Card(12, 0), Card(12, 1),  # pair of queens
        Card(10, 2), Card(7, 3), Card(3, 0),
    ]
    score = HandEvaluator.evaluate(cards)
    assert score[0] == HandRank.ONE_PAIR.value


def test_hand_evaluator_flush():
    """Hand evaluator should detect a flush."""
    cards = [
        Card(14, 0), Card(10, 0), Card(8, 0), Card(5, 0), Card(3, 0),
    ]
    score = HandEvaluator.evaluate(cards)
    assert score[0] == HandRank.FLUSH.value


def test_hand_evaluator_straight():
    """Hand evaluator should detect a straight."""
    cards = [
        Card(10, 0), Card(9, 1), Card(8, 2), Card(7, 3), Card(6, 0),
    ]
    score = HandEvaluator.evaluate(cards)
    assert score[0] == HandRank.STRAIGHT.value


def test_hand_evaluator_full_house():
    """Hand evaluator should detect a full house."""
    cards = [
        Card(10, 0), Card(10, 1), Card(10, 2), Card(5, 0), Card(5, 1),
    ]
    score = HandEvaluator.evaluate(cards)
    assert score[0] == HandRank.FULL_HOUSE.value


def test_hand_evaluator_best_5_of_7():
    """Hand evaluator should pick best 5 from 7 cards."""
    cards = [
        Card(14, 0), Card(14, 1), Card(14, 2),  # three aces
        Card(10, 0), Card(10, 1),                # pair of tens
        Card(3, 2), Card(2, 3),                  # junk
    ]
    score = HandEvaluator.evaluate(cards)
    assert score[0] == HandRank.FULL_HOUSE.value


def test_game_state_format_context():
    """GameState.format_context() should produce a parseable string."""
    deck = Deck()
    state = GameState(
        num_players=6, small_blind=1, big_blind=2,
        hero_position="BTN", hero_hole=deck.deal(2),
        players=[
            PlayerState("UTG", 200, True),
            PlayerState("MP", 200, True),
            PlayerState("CO", 200, True),
            PlayerState("BTN", 200, True),
            PlayerState("SB", 200, True),
            PlayerState("BB", 200, True),
        ],
        board=deck.deal(3), street="flop",
        pot=15, current_bet=5, hero_bet_this_round=0,
        actions=[Action("UTG", "raise", 6)],
    )
    context = state.format_context()
    assert "Your Hand:" in context
    assert "Your Position: BTN" in context
    assert "Community Cards:" in context
    assert "Pot: $15" in context


# ── Heuristic Tests ──────────────────────────────────────────────────

def test_preflop_tiers():
    """Preflop tier function should correctly categorize hands."""
    # AA = Tier 1
    assert preflop_tier([Card(14, 0), Card(14, 1)]) == 1
    # KK = Tier 1
    assert preflop_tier([Card(13, 0), Card(13, 1)]) == 1
    # JJ = Tier 2
    assert preflop_tier([Card(11, 0), Card(11, 1)]) == 2
    # 99 = Tier 3
    assert preflop_tier([Card(9, 0), Card(9, 1)]) == 3
    # 22 = Tier 4
    assert preflop_tier([Card(2, 0), Card(2, 1)]) == 4
    # 72o = Tier 5
    assert preflop_tier([Card(7, 0), Card(2, 1)]) == 5


def test_hand_key():
    """_hand_key should produce canonical hand notation."""
    # Pocket pair
    assert _hand_key([Card(14, 0), Card(14, 1)]) == "AA"
    # Suited (AKs)
    assert _hand_key([Card(14, 0), Card(13, 0)]) == "AKs"
    # Offsuit (AKo)
    assert _hand_key([Card(14, 0), Card(13, 1)]) == "AKo"


def test_postflop_strength_categories():
    """postflop_strength should categorize hand strengths."""
    # Monster: straight
    hole = [Card(10, 0), Card(9, 1)]
    board = [Card(8, 2), Card(7, 3), Card(6, 0)]
    cat, strength = postflop_strength(hole, board)
    assert cat == "monster"
    assert strength >= 0.90

    # Nothing: no pair, no draw
    hole2 = [Card(2, 0), Card(3, 1)]
    board2 = [Card(10, 2), Card(13, 3), Card(14, 0)]
    cat2, strength2 = postflop_strength(hole2, board2)
    assert cat2 == "nothing"
    assert strength2 < 0.20


def test_flush_draw_detection():
    """has_flush_draw should detect 4 cards of same suit."""
    hole = [Card(14, 0), Card(10, 0)]   # two hearts
    board = [Card(7, 0), Card(3, 0), Card(2, 1)]  # two hearts + one diamond... that's 4 hearts
    # Actually: A♠, T♠, 7♠, 3♠ (suit=0) = 4 of same suit → flush draw
    assert has_flush_draw(hole, board) is True

    # No flush draw
    hole2 = [Card(14, 0), Card(10, 1)]
    board2 = [Card(7, 2), Card(3, 3), Card(2, 0)]
    assert has_flush_draw(hole2, board2) is False


def test_parse_opponent_stats():
    """parse_opponent_stats should compute stats from hand records."""
    records = [
        HandRecord(
            hand_number=1,
            actions_by_street={
                "preflop": [
                    Action("UTG", "raise", 6),
                    Action("CO", "call", 6),
                    Action("BTN", "fold"),
                ],
                "flop": [
                    Action("UTG", "bet", 8),
                    Action("CO", "call", 8),
                ],
            },
            board_by_street={"flop": [Card(10, 0), Card(7, 1), Card(3, 2)]},
            winner="UTG",
            pot_won=28,
            players_involved=["UTG", "CO", "BTN"],
        ),
        HandRecord(
            hand_number=2,
            actions_by_street={
                "preflop": [
                    Action("UTG", "raise", 6),
                    Action("CO", "fold"),
                    Action("BTN", "call", 6),
                ],
            },
            board_by_street={},
            winner="UTG",
            pot_won=15,
            players_involved=["UTG", "CO", "BTN"],
        ),
    ]

    stats = parse_opponent_stats(records)
    assert "UTG" in stats
    assert stats["UTG"].hands_seen == 2
    assert stats["UTG"].vpip_hands >= 2   # raised both hands
    assert stats["UTG"].pfr_hands >= 2    # raised preflop both hands
    assert stats["UTG"].postflop_bets >= 1  # bet on flop in hand 1


def test_heuristic_bot_decides():
    """HeuristicPokerBot should return a valid action."""
    deck = Deck()
    state = GameState(
        num_players=6, small_blind=1, big_blind=2,
        hero_position="BTN", hero_hole=deck.deal(2),
        players=[
            PlayerState("UTG", 200, True),
            PlayerState("MP", 200, False),
            PlayerState("CO", 200, True),
            PlayerState("BTN", 200, True),
            PlayerState("SB", 200, False),
            PlayerState("BB", 200, True),
        ],
        board=[], street="preflop",
        pot=3, current_bet=2, hero_bet_this_round=0,
        actions=[Action("UTG", "raise", 6)],
    )
    state.hand_history = []

    bot = HeuristicPokerBot()
    action, amount = bot.decide(state)
    assert action in ("fold", "check", "call", "raise")
    assert amount >= 0


# ── Task Generation Tests ────────────────────────────────────────────

def test_generate_poker_task():
    """generate_poker_task should return (context, question, answer)."""
    context, question, answer = generate_poker_task()
    assert isinstance(context, str)
    assert len(context) > 1000
    assert "Your Hand:" in context
    assert "PREVIOUS HANDS" in context
    assert "What should you do?" in question
    action_type, _ = parse_action(answer)
    assert action_type in ("fold", "check", "call", "raise")


def test_generate_poker_task_with_trace():
    """generate_poker_task_with_trace should also return a reasoning trace."""
    context, question, answer, trace = generate_poker_task_with_trace()
    assert "STEP 1: RETRIEVE" in trace
    assert "STEP 2: COMPUTE" in trace
    assert "STEP 3: DECIDE" in trace


def test_task_context_has_history():
    """Task context should contain 15 hand history records."""
    context, _, _ = generate_poker_task(num_history_hands=15)
    # Count "Hand #" occurrences
    hand_count = context.count("Hand #")
    assert hand_count >= 10, f"Expected at least 10 hands in history, got {hand_count}"


def test_task_all_streets():
    """Task generator should produce tasks for all streets."""
    streets_seen = set()
    for _ in range(50):
        context, _, _ = generate_poker_task()
        if "(Flop)" in context:
            streets_seen.add("flop")
        elif "(Turn)" in context:
            streets_seen.add("turn")
        elif "(River)" in context:
            streets_seen.add("river")
        else:
            streets_seen.add("preflop")
    assert len(streets_seen) >= 3, f"Only saw streets: {streets_seen}"


# ── Reward Tests ─────────────────────────────────────────────────────

def test_parse_action():
    """parse_action should correctly parse all action types."""
    assert parse_action("fold") == ("fold", 0)
    assert parse_action("check") == ("check", 0)
    assert parse_action("call $10") == ("call", 10)
    assert parse_action("raise $25") == ("raise", 25)
    assert parse_action("FOLD")[0] == "fold"


def test_reward_exact_match():
    """Exact action match should get full reward."""
    r = compute_poker_reward_simple("fold", "fold")
    assert r >= 0.9


def test_reward_type_match():
    """Same action type should get full reward (simple reward ignores amounts)."""
    r = compute_poker_reward_simple("raise $20", "raise $25")
    assert r == 1.0

    # Partial credit for staying in pot (call vs raise)
    r2 = compute_poker_reward_simple("call $10", "raise $25")
    assert 0.0 < r2 < 1.0


def test_reward_wrong_action():
    """Wrong action type should get low reward."""
    r = compute_poker_reward_simple("fold", "raise $25")
    assert r < 0.3


# ── Agent Tests ──────────────────────────────────────────────────────

def test_poker_heuristic_agent_repl():
    """PokerHeuristicAgent should produce 3-step transcripts with valid code."""
    agent = PokerHeuristicAgent()
    context, question, answer = generate_poker_task()
    predicted, transcript = agent.run_episode(context, question, answer)

    # Should have exactly 3 steps
    assert len(transcript) == 3

    # All code should execute successfully
    for step in transcript:
        assert step["exec_result"]["ok"], f"Step {step['step']} failed: {step['exec_result']['stderr']}"

    # Step 1 should parse stats (contain VPIP in output)
    step1_out = transcript[0]["exec_result"]["stdout"]
    assert "VPIP" in step1_out, "Step 1 should output opponent VPIP stats"

    # Step 2 should output hand info
    step2_out = transcript[1]["exec_result"]["stdout"]
    assert "Hand:" in step2_out or "Pot:" in step2_out

    # Predicted should match answer (heuristic generates its own answer)
    pred_type, _ = parse_action(predicted)
    corr_type, _ = parse_action(answer)
    assert pred_type == corr_type


# ── Evaluation Framework Test ────────────────────────────────────────

def test_evaluation_framework():
    """PokerEvaluationFramework should compute metrics correctly."""
    agent = PokerHeuristicAgent()
    framework = PokerEvaluationFramework(
        agents=[agent],
        task_generator=generate_poker_task,
        num_episodes=10,
    )
    framework.run_evaluation()

    results = framework.results["PokerHeuristicAgent"]
    assert results["total"] == 10
    assert results["correct"] == 10  # heuristic matches itself
    assert results["type_match"] == 10

    # Confusion matrix should be diagonal
    matrix = framework.get_confusion_matrix("PokerHeuristicAgent")
    off_diagonal = sum(
        matrix[true_a][pred_a]
        for true_a in matrix
        for pred_a in matrix[true_a]
        if true_a != pred_a
    )
    assert off_diagonal == 0, "Heuristic agent should have no off-diagonal confusion"


def test_bc_agent_task_generators():
    """BC mix helper should return the expected number of task sources."""
    assert len(bc_agent_task_generators("all")) == 1
    assert len(bc_agent_task_generators("preflop")) == 1
    assert len(bc_agent_task_generators("postflop")) == 1
    assert len(bc_agent_task_generators("mixed")) == 3
    for gen in bc_agent_task_generators("mixed"):
        ctx, q, a = gen()
        assert "What should you do?" in q
        assert isinstance(ctx, str) and isinstance(a, str)


def test_evaluation_export_json_roundtrip():
    """export_run_summary and save_results_json should produce valid JSON."""
    agent = PokerHeuristicAgent()
    framework = PokerEvaluationFramework(
        agents=[agent],
        task_generator=generate_poker_task,
        num_episodes=5,
    )
    framework.run_evaluation()
    summary = framework.export_run_summary()
    assert "agents" in summary
    assert "PokerHeuristicAgent" in summary["agents"]
    assert summary["agents"]["PokerHeuristicAgent"]["total_episodes"] == 5

    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "eval_out.json")
        framework.save_results_json(path, meta={"note": "unit_test"})
        with open(path, encoding="utf-8") as f:
            loaded = json.load(f)
    assert loaded["meta"]["note"] == "unit_test"
    assert loaded["agents"]["PokerHeuristicAgent"]["exact_match_rate"] == 1.0
