"""
Slumbot session runner — plays N hands using a heuristic agent,
records results, and writes graphs to docs/results/slumbot_session_20260513/.

Usage:
    python scripts/run_slumbot_session.py --user aryana --password <pw> --hands 50
"""

import argparse
import csv
import json
import os
import re
import time
from pathlib import Path

import httpx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

# ── Constants (Slumbot heads-up $50/$100, 200 bb deep) ─────────────────────
SLUMBOT_URL = "https://slumbot.com/slumbot"
SB, BB, STACK = 50, 100, 20_000

RESULTS_DIR = Path(__file__).parent.parent / "docs/results/slumbot_session_20260513"

# ── Card helpers ─────────────────────────────────────────────────────────────
RANK_ORDER = "23456789TJQKA"
SUIT_ORDER  = "cdhs"

def card_rank(c: str) -> int:
    return RANK_ORDER.index(c[0].upper() if c[0] in "TJQKA" else c[0])

def hand_strength_preflop(hole: list[str]) -> float:
    """Return [0,1] heuristic strength for a 2-card hand."""
    if len(hole) < 2:
        return 0.3
    r0, r1 = card_rank(hole[0]), card_rank(hole[1])
    hi, lo = max(r0, r1), min(r0, r1)
    suited = hole[0][-1] == hole[1][-1]
    # Pairs
    if hi == lo:
        return 0.5 + hi / 26.0          # AA=1.0, 22≈0.54
    gap = hi - lo
    base = (hi + lo) / 26.0             # high-card value [0,1]
    suited_bonus = 0.06 if suited else 0.0
    gap_penalty  = gap * 0.03
    return min(1.0, max(0.0, base + suited_bonus - gap_penalty))

def hand_strength_postflop(hole: list[str], board: list[str]) -> float:
    """Very rough post-flop strength: pair/trip/flush-draw detection."""
    if not board:
        return hand_strength_preflop(hole)
    all_cards = hole + board
    ranks = [card_rank(c) for c in all_cards]
    suits = [c[-1] for c in all_cards]
    rank_counts = {}
    for r in ranks:
        rank_counts[r] = rank_counts.get(r, 0) + 1
    suit_counts = {}
    for s in suits:
        suit_counts[s] = suit_counts.get(s, 0) + 1

    has_pair  = any(v >= 2 for v in rank_counts.values())
    has_trips = any(v >= 3 for v in rank_counts.values())
    has_quads = any(v >= 4 for v in rank_counts.values())
    has_fd    = any(v >= 4 for v in suit_counts.values())  # flush draw
    top_pair  = any(card_rank(c) == max(card_rank(b) for b in board)
                    for c in hole if c in all_cards)

    score = 0.3
    if has_pair:  score += 0.15
    if top_pair:  score += 0.10
    if has_trips: score += 0.30
    if has_quads: score += 0.50
    if has_fd:    score += 0.12
    # high-card bonus
    score += max(card_rank(c) for c in hole) / 80.0
    return min(1.0, score)

# ── Slumbot state walker (mirrors notebook Cell 30) ──────────────────────────
def _compute_state(action_str: str, client_pos: int) -> dict:
    streets = action_str.split("/") if action_str else [""]
    street_idx = len(streets) - 1
    total_invested = [SB, BB]
    last_street_bets = [SB, BB]

    for s_idx, s_actions in enumerate(streets):
        street_bets = [SB, BB] if s_idx == 0 else [0, 0]
        first_actor_pos = 0 if s_idx == 0 else 1
        action_idx = 0
        i = 0
        while i < len(s_actions):
            c = s_actions[i]
            if c == "b":
                j = i + 1
                while j < len(s_actions) and s_actions[j].isdigit():
                    j += 1
                amt = int(s_actions[i + 1: j])
                who = (first_actor_pos + action_idx) % 2
                total_invested[who] += max(amt - street_bets[who], 0)
                street_bets[who] = amt
                action_idx += 1
                i = j
            elif c == "c":
                who = (first_actor_pos + action_idx) % 2
                top = max(street_bets)
                total_invested[who] += max(top - street_bets[who], 0)
                street_bets[who] = top
                action_idx += 1
                i += 1
            elif c in "kf":
                action_idx += 1
                i += 1
            else:
                i += 1
        last_street_bets = street_bets

    if client_pos == 0:
        our_inv, opp_inv = total_invested[1], total_invested[0]
        our_sb, opp_sb   = last_street_bets[1], last_street_bets[0]
    else:
        our_inv, opp_inv = total_invested[0], total_invested[1]
        our_sb, opp_sb   = last_street_bets[0], last_street_bets[1]

    return {
        "street": street_idx,
        "pot": our_inv + opp_inv,
        "our_invested": our_inv,
        "opp_invested": opp_inv,
        "to_call": max(0, opp_inv - our_inv),
        "our_remaining": STACK - our_inv,
        "our_street_bet": our_sb,
        "opp_street_bet": opp_sb,
    }

# ── Heuristic decision (no LLM) ──────────────────────────────────────────────
def heuristic_decision(state: dict, hole: list[str], board: list[str]) -> str:
    """
    Strength-based heuristic:
      strong  (>0.70): raise/bet aggressively
      medium  (>0.45): call/check
      weak    (<=0.45): fold to bets, check otherwise
    """
    if state["street"] == 0:
        strength = hand_strength_preflop(hole)
    else:
        strength = hand_strength_postflop(hole, board)

    to_call   = state["to_call"]
    remaining = state["our_remaining"]
    street    = state["street"]
    our_sb    = state["our_street_bet"]
    opp_sb    = state["opp_street_bet"]
    check_tok = "k" if street > 0 else "c"

    if strength > 0.70:
        # Raise / bet
        if to_call > 0:
            target = max(opp_sb * 2, opp_sb + BB)
        else:
            target = max(BB, opp_sb + BB)
        target = min(target, our_sb + remaining)
        if target <= opp_sb:
            return "c" if to_call > 0 else check_tok
        return f"b{target}"
    elif strength > 0.45:
        return "c" if to_call > 0 else check_tok
    else:
        return "f" if to_call > 0 else check_tok

# ── Slumbot REST client ───────────────────────────────────────────────────────
class SlumbotSession:
    def __init__(self, username, password):
        self.username   = username
        self.password   = password
        self.token      = ""
        self.hands_played   = 0
        self.total_winnings = 0

    def login(self) -> bool:
        r = httpx.post(
            f"{SLUMBOT_URL}/api/login",
            json={"username": self.username, "password": self.password},
            headers={"Content-Type": "application/json"},
            timeout=15,
        )
        if r.status_code != 200:
            return False
        self.token = r.json().get("token", "")
        return bool(self.token)

    def new_hand(self) -> dict:
        body = {"token": self.token} if self.token else {}
        r = httpx.post(f"{SLUMBOT_URL}/api/new_hand", json=body, timeout=15)
        data = r.json()
        if "token" in data:
            self.token = data["token"]
        return data

    def act(self, incr: str) -> dict:
        r = httpx.post(
            f"{SLUMBOT_URL}/api/act",
            json={"token": self.token, "incr": incr},
            timeout=15,
        )
        data = r.json()
        if "token" in data:
            self.token = data["token"]
        return data

# ── Play one hand ─────────────────────────────────────────────────────────────
def play_one_hand(session: SlumbotSession, hand_num: int, verbose=True) -> dict | None:
    data = session.new_hand()
    if "error" in data:
        print(f"  Error: {data['error']}")
        return None

    client_pos = data.get("client_pos", 0)
    hole       = data.get("hole_cards", [])
    position   = "BB" if client_pos == 0 else "BTN"

    if verbose:
        print(f"\n--- Hand #{hand_num}: {' '.join(hole)} ({position}) ---")

    # Slumbot immediate fold
    initial_action = data.get("action", "")
    if initial_action.endswith("f"):
        session.hands_played   += 1
        session.total_winnings += SB
        if verbose:
            print(f"  Slumbot folded preflop. +{SB/BB:.1f} bb")
        return {"hand": hand_num, "winnings_bb": SB / BB, "position": position,
                "hole": hole, "board": [], "turns": 0, "result": "slumbot_fold_preflop"}

    turns  = 0
    result = "unknown"
    for _ in range(25):
        action_str = data.get("action", "")
        board      = data.get("board", []) or []
        winnings   = data.get("winnings")

        if winnings is not None and turns > 0:
            break

        state = _compute_state(action_str, client_pos)
        incr  = heuristic_decision(state, hole, board)
        turns += 1

        if verbose:
            board_s = " ".join(board) if board else "—"
            print(f"  T{turns}: pot=${state['pot']} to_call=${state['to_call']} "
                  f"board=[{board_s}]  action={incr!r}")

        data = session.act(incr)

    winnings = data.get("winnings", 0) or 0
    session.hands_played   += 1
    session.total_winnings += winnings
    w_bb = winnings / BB
    if verbose:
        sign = "+" if w_bb >= 0 else ""
        print(f"  Result: {sign}{w_bb:.1f} bb")

    return {
        "hand": hand_num,
        "winnings_bb": w_bb,
        "position": position,
        "hole": hole,
        "board": data.get("board", []) or [],
        "turns": turns,
        "result": "win" if w_bb > 0 else ("loss" if w_bb < 0 else "chop"),
    }

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--user",     required=True)
    parser.add_argument("--password", required=True)
    parser.add_argument("--hands",    type=int, default=50)
    parser.add_argument("--verbose",  action="store_true", default=True)
    args = parser.parse_args()

    session = SlumbotSession(args.user, args.password)
    print(f"Logging in as '{args.user}'...")
    if not session.login():
        print("Login failed.")
        return
    print(f"Logged in. Playing {args.hands} hands vs Slumbot (heads-up, $50/$100)...\n")

    records = []
    for i in range(1, args.hands + 1):
        try:
            r = play_one_hand(session, i, verbose=args.verbose)
            if r:
                records.append(r)
            time.sleep(0.15)
        except Exception as e:
            print(f"  Hand {i} error: {e!r}")
            time.sleep(1.0)

    # ── Save raw records ──────────────────────────────────────────────────────
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = RESULTS_DIR / "slumbot_results.csv"
    fields = ["hand", "winnings_bb", "position", "hole", "board", "turns", "result"]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for rec in records:
            row = dict(rec)
            row["hole"]  = " ".join(rec["hole"])
            row["board"] = " ".join(rec["board"])
            w.writerow(row)
    print(f"\nSaved {len(records)} hands → {csv_path}")

    # ── Summary stats ─────────────────────────────────────────────────────────
    if not records:
        print("No records — exiting.")
        return

    winnings = [r["winnings_bb"] for r in records]
    cumulative = list(np.cumsum(winnings))
    n = len(records)
    total_bb   = sum(winnings)
    bb_per_100 = total_bb / n * 100
    wins   = sum(1 for w in winnings if w > 0)
    losses = sum(1 for w in winnings if w < 0)
    chops  = n - wins - losses
    avg_pot_size = np.mean([abs(w) for w in winnings if w != 0]) if any(w != 0 for w in winnings) else 0

    summary = {
        "hands_played": n,
        "total_bb": round(total_bb, 2),
        "bb_per_100": round(bb_per_100, 2),
        "win_rate": round(wins / n, 3),
        "wins": wins, "losses": losses, "chops": chops,
        "avg_abs_swing_bb": round(avg_pot_size, 2),
        "max_win_bb": round(max(winnings), 2),
        "max_loss_bb": round(min(winnings), 2),
    }
    with open(RESULTS_DIR / "slumbot_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "="*54)
    print(f"  RLM (heuristic) vs Slumbot — {n} hands")
    print("="*54)
    print(f"  Total:     {'+' if total_bb >= 0 else ''}{total_bb:.1f} bb")
    print(f"  bb/100:    {bb_per_100:+.1f}")
    print(f"  Win rate:  {wins}/{n}  ({wins/n*100:.1f}%)")
    print(f"  Losses:    {losses}   Chops: {chops}")
    print("="*54)

    # ── Plots ─────────────────────────────────────────────────────────────────
    hands_axis = list(range(1, n + 1))
    ema_alpha  = 0.15
    ema = []
    e   = winnings[0]
    for w in winnings:
        e = ema_alpha * w + (1 - ema_alpha) * e
        ema.append(e)

    colors = {"win": "#27ae60", "loss": "#e74c3c", "chop": "#95a5a6",
              "slumbot_fold_preflop": "#3498db"}
    bar_colors = [colors.get(r["result"], "#95a5a6") for r in records]

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle("RLM Heuristic Agent vs Slumbot\n(Heads-up NLHE, $50/$100, 200bb)",
                 fontsize=13, fontweight="bold", y=1.01)

    # 1) Cumulative P/L
    ax = axes[0, 0]
    ax.plot(hands_axis, cumulative, color="#2980b9", linewidth=2, label="Cumulative bb")
    ax.axhline(0, color="grey", linewidth=0.8, linestyle="--")
    ax.fill_between(hands_axis, cumulative, 0,
                    where=[c >= 0 for c in cumulative], alpha=0.15, color="#27ae60")
    ax.fill_between(hands_axis, cumulative, 0,
                    where=[c < 0 for c in cumulative], alpha=0.15, color="#e74c3c")
    ax.set_xlabel("Hand #"); ax.set_ylabel("Cumulative (bb)")
    ax.set_title("Cumulative Profit / Loss"); ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    # 2) Per-hand P/L bar chart
    ax = axes[0, 1]
    ax.bar(hands_axis, winnings, color=bar_colors, alpha=0.85, width=0.8)
    ax.axhline(0, color="grey", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Hand #"); ax.set_ylabel("bb")
    ax.set_title("Per-Hand Result")
    from matplotlib.patches import Patch
    legend_handles = [
        Patch(color="#27ae60", label="Win"),
        Patch(color="#e74c3c", label="Loss"),
        Patch(color="#3498db", label="Slumbot folds pre"),
        Patch(color="#95a5a6", label="Chop"),
    ]
    ax.legend(handles=legend_handles, fontsize=8)
    ax.grid(axis="y", alpha=0.3)

    # 3) EMA win rate
    ax = axes[1, 0]
    ax.plot(hands_axis, ema, color="#8e44ad", linewidth=2, label="EMA winnings (α=0.15)")
    ax.axhline(0, color="grey", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Hand #"); ax.set_ylabel("EMA bb per hand")
    ax.set_title("Smoothed Performance (EMA)"); ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    # 4) Outcome distribution
    ax = axes[1, 1]
    cats   = ["Win", "Loss", "Chop / SB\nblind steal"]
    counts = [wins, losses, chops]
    cols   = ["#27ae60", "#e74c3c", "#3498db"]
    bars = ax.bar(cats, counts, color=cols, alpha=0.85)
    ax.set_ylabel("Hands"); ax.set_title("Outcome Distribution")
    ax.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    for b in bars:
        h = b.get_height()
        if h > 0:
            ax.text(b.get_x() + b.get_width() / 2, h + 0.3,
                    str(int(h)), ha="center", va="bottom", fontsize=11, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    fig_path = RESULTS_DIR / "slumbot_performance.png"
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved plot → {fig_path}")

    # 5) Simple action distribution (folded vs called vs raised)
    fig2, ax2 = plt.subplots(figsize=(7, 4))
    by_pos = {"BB": {"wins": 0, "losses": 0, "hands": 0},
              "BTN": {"wins": 0, "losses": 0, "hands": 0}}
    for r in records:
        pos = r["position"]
        if pos not in by_pos:
            by_pos[pos] = {"wins": 0, "losses": 0, "hands": 0}
        by_pos[pos]["hands"] += 1
        if r["winnings_bb"] > 0:
            by_pos[pos]["wins"] += 1
        elif r["winnings_bb"] < 0:
            by_pos[pos]["losses"] += 1

    positions = list(by_pos.keys())
    pos_wins   = [by_pos[p]["wins"]   for p in positions]
    pos_losses = [by_pos[p]["losses"] for p in positions]
    pos_other  = [by_pos[p]["hands"] - by_pos[p]["wins"] - by_pos[p]["losses"] for p in positions]

    x = np.arange(len(positions))
    w = 0.25
    ax2.bar(x - w, pos_wins,   w, label="Win",  color="#27ae60", alpha=0.85)
    ax2.bar(x,     pos_losses, w, label="Loss", color="#e74c3c", alpha=0.85)
    ax2.bar(x + w, pos_other,  w, label="Chop/SB steal", color="#3498db", alpha=0.85)
    ax2.set_xticks(x); ax2.set_xticklabels(positions, fontsize=11)
    ax2.set_ylabel("Hands"); ax2.set_title("Win/Loss by Position (BB vs BTN)")
    ax2.legend(fontsize=9); ax2.grid(axis="y", alpha=0.3)
    ax2.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    plt.tight_layout()
    fig2_path = RESULTS_DIR / "slumbot_by_position.png"
    plt.savefig(fig2_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved plot → {fig2_path}")

    print("\nDone.")

if __name__ == "__main__":
    main()
