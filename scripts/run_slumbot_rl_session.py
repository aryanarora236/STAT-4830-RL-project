"""
RLM vs Slumbot — minimal GPU session.
Loads the best_by_eval RL checkpoint, plays N hands, saves results.

Usage:
    python scripts/run_slumbot_rl_session.py \
        --user aryana --password 'Aryan715738!' --hands 10
"""

import argparse, csv, json, os, re, sys, time
from pathlib import Path

import httpx
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch
import matplotlib.ticker as mticker

sys.path.insert(0, str(Path(__file__).parent.parent))

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

# ── Constants ─────────────────────────────────────────────────────────────────
SLUMBOT_URL = "https://slumbot.com/slumbot"
SB, BB, STACK = 50, 100, 20_000
BASE_MODEL = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
CKPT_CANDIDATES = [
    "docs/results/poker_rl_expA_partial_20260422/poker_rl_expA_evalselect_20260421_long/best_by_eval",
    "docs/results/poker_rl_expB_mixed20_20260422/poker_rl_expB_mixed20_20260422/best_by_eval",
]
RESULTS_DIR = Path("docs/results/slumbot_rl_session_20260513")

# ── Slumbot state walker ───────────────────────────────────────────────────────
def _compute_state(action_str, client_pos):
    streets = action_str.split("/") if action_str else [""]
    street_idx = len(streets) - 1
    total_invested = [SB, BB]
    last_street_bets = [SB, BB]
    for s_idx, s_actions in enumerate(streets):
        street_bets = [SB, BB] if s_idx == 0 else [0, 0]
        first_actor_pos = 0 if s_idx == 0 else 1
        action_idx = i = 0
        while i < len(s_actions):
            c = s_actions[i]
            if c == "b":
                j = i + 1
                while j < len(s_actions) and s_actions[j].isdigit():
                    j += 1
                amt = int(s_actions[i+1:j])
                who = (first_actor_pos + action_idx) % 2
                total_invested[who] += max(amt - street_bets[who], 0)
                street_bets[who] = amt
                action_idx += 1; i = j
            elif c == "c":
                who = (first_actor_pos + action_idx) % 2
                top = max(street_bets)
                total_invested[who] += max(top - street_bets[who], 0)
                street_bets[who] = top
                action_idx += 1; i += 1
            elif c in "kf":
                action_idx += 1; i += 1
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
        "street": street_idx, "pot": our_inv + opp_inv,
        "our_invested": our_inv, "opp_invested": opp_inv,
        "to_call": max(0, opp_inv - our_inv),
        "our_remaining": STACK - our_inv,
        "our_street_bet": our_sb, "opp_street_bet": opp_sb,
    }

STREET_NAMES = ["Preflop", "Flop", "Turn", "River"]

def build_context(state, hole, board, client_pos, action_str, hand_num):
    position = "BB" if client_pos == 0 else "BTN"
    opp_pos   = "BTN" if client_pos == 0 else "BB"
    opp_stack = STACK - state["opp_invested"]
    street_name = STREET_NAMES[min(state["street"], 3)]
    lines = [
        "=== POKER HAND ===",
        "Table: Heads-up No-Limit Hold'em",
        f"Blinds: ${SB}/${BB}", "",
        f"Your Position: {position}",
        f"Your Stack: ${state['our_remaining']:.0f}",
        f"Your Hand: {hole[0]} {hole[1]}", "",
        "Opponents:",
        f"  {opp_pos} (Slumbot): ${opp_stack:.0f} (active)", "",
    ]
    if board:
        lines.append(f"Community Cards: {' '.join(board)} ({street_name})")
    else:
        lines.append("Street: Preflop (no community cards)")
    lines += ["", f"Pot: ${state['pot']:.0f}"]
    if state["to_call"] > 0:
        pot_odds = state["to_call"] / (state["pot"] + state["to_call"]) * 100
        lines += [f"To Call: ${state['to_call']:.0f}", f"Pot Odds: {pot_odds:.1f}%"]
    lines.append("")
    if action_str:
        lines += ["Betting History (Slumbot wire format):", f"  {action_str}", ""]
    return "\n".join(lines)

def parse_rlm_action(text):
    lines = (text or "").strip().lower().splitlines()
    last = lines[-1].strip() if lines else ""
    if last.startswith("fold"):  return ("fold", 0.0)
    if last.startswith("check"): return ("check", 0.0)
    m = re.match(r"(call|raise|bet)\s*\$?\s*(\d+\.?\d*)", last)
    if m:
        kind = "raise" if m.group(1) == "bet" else m.group(1)
        return (kind, float(m.group(2)))
    for kw in ("fold", "check", "call", "raise"):
        if kw in last: return (kw, 0.0)
    return ("fold", 0.0)

def rlm_to_incr(predicted, state):
    action_type, _ = parse_rlm_action(predicted)
    to_call = state["to_call"]; street = state["street"]
    our_sb = state["our_street_bet"]; opp_sb = state["opp_street_bet"]
    remaining = state["our_remaining"]
    check_tok = "k" if street > 0 else "c"
    if action_type == "fold":   return "f" if to_call > 0 else check_tok
    if action_type in ("check","call"): return "c" if to_call > 0 else check_tok
    if action_type == "raise":
        target = max(opp_sb*2, opp_sb+BB) if to_call > 0 else max(BB, opp_sb+BB)
        target = min(target, our_sb + remaining)
        if target <= opp_sb: return "c" if to_call > 0 else check_tok
        return f"b{target}"
    return "f" if to_call > 0 else check_tok

# ── LLM inference ──────────────────────────────────────────────────────────────
def llm_decide(model, tokenizer, context, state):
    prompt = (
        f"You are a poker agent. Given this game state, write Python code that "
        f"reads CONTEXT and prints one of: fold, check, call $X, raise $X.\n\n"
        f"CONTEXT = '''{context}'''\n\n"
        f"Write Python code to decide the best action:\n```python\n"
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs, max_new_tokens=256, temperature=0.2, top_p=0.9,
            do_sample=True, pad_token_id=tokenizer.eos_token_id,
        )
    generated = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    # Try to exec the code
    code_match = re.search(r"```python\n(.*?)```", generated, re.DOTALL)
    if not code_match:
        code_match = re.search(r"```\n(.*?)```", generated, re.DOTALL)
    real_code = False
    stdout_result = ""
    if code_match:
        code = code_match.group(1)
        if len(code.strip().splitlines()) > 1:  # not just a one-liner
            real_code = True
        import io, contextlib
        buf = io.StringIO()
        try:
            with contextlib.redirect_stdout(buf):
                exec(code, {"CONTEXT": context})
            stdout_result = buf.getvalue().strip()
        except Exception as e:
            stdout_result = ""
    if not stdout_result:
        # fallback: scan generated text for action
        stdout_result = generated
    return stdout_result, real_code, generated

# ── Slumbot client ─────────────────────────────────────────────────────────────
class SlumbotSession:
    def __init__(self, u, p):
        self.username = u; self.password = p
        self.token = ""; self.hands_played = 0; self.total_winnings = 0
    def login(self):
        r = httpx.post(f"{SLUMBOT_URL}/api/login",
            json={"username": self.username, "password": self.password},
            headers={"Content-Type": "application/json"}, timeout=15)
        if r.status_code != 200: return False
        self.token = r.json().get("token", ""); return bool(self.token)
    def new_hand(self):
        body = {"token": self.token} if self.token else {}
        r = httpx.post(f"{SLUMBOT_URL}/api/new_hand", json=body, timeout=15)
        d = r.json();
        if "token" in d: self.token = d["token"]
        return d
    def act(self, incr):
        r = httpx.post(f"{SLUMBOT_URL}/api/act",
            json={"token": self.token, "incr": incr}, timeout=15)
        d = r.json()
        if "token" in d: self.token = d["token"]
        return d

def play_one_hand(session, model, tokenizer, hand_num):
    data = session.new_hand()
    if "error" in data: return None
    client_pos = data.get("client_pos", 0)
    hole = data.get("hole_cards", [])
    position = "BB" if client_pos == 0 else "BTN"
    print(f"\n--- Hand #{hand_num}: {' '.join(hole)} ({position}) ---")

    # Immediate Slumbot fold
    if data.get("action", "").endswith("f"):
        session.hands_played += 1; session.total_winnings += SB
        print(f"  Slumbot folded preflop. +{SB/BB:.1f} bb")
        return {"hand": hand_num, "winnings_bb": SB/BB, "position": position,
                "hole": hole, "board": [], "turns": 0,
                "result": "slumbot_fold_preflop", "real_code_turns": 0}

    real_code_turns = turns = 0
    for _ in range(20):
        action_str = data.get("action", "")
        board = data.get("board", []) or []
        winnings = data.get("winnings")
        if winnings is not None and turns > 0: break

        state = _compute_state(action_str, client_pos)
        context = build_context(state, hole, board, client_pos, action_str, hand_num)

        predicted, real_code, raw = llm_decide(model, tokenizer, context, state)
        if real_code: real_code_turns += 1
        incr = rlm_to_incr(predicted, state)
        turns += 1

        board_s = " ".join(board) if board else "—"
        code_tag = "✓CODE" if real_code else "text"
        print(f"  T{turns}: pot=${state['pot']} to_call=${state['to_call']} "
              f"board=[{board_s}]  [{code_tag}] pred={predicted!r:.30} -> {incr!r}")
        data = session.act(incr)

    winnings = data.get("winnings", 0) or 0
    session.hands_played += 1; session.total_winnings += winnings
    w_bb = winnings / BB
    sign = "+" if w_bb >= 0 else ""
    print(f"  Result: {sign}{w_bb:.1f} bb  (code used {real_code_turns}/{turns} turns)")
    return {
        "hand": hand_num, "winnings_bb": w_bb, "position": position,
        "hole": hole, "board": data.get("board", []) or [],
        "turns": turns, "real_code_turns": real_code_turns,
        "result": "win" if w_bb > 0 else ("loss" if w_bb < 0 else "chop"),
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--user",     required=True)
    parser.add_argument("--password", required=True)
    parser.add_argument("--hands",    type=int, default=10)
    args = parser.parse_args()

    # ── Load model ─────────────────────────────────────────────────────────────
    ckpt = None
    for c in CKPT_CANDIDATES:
        if os.path.isdir(c):
            ckpt = c; break
    if not ckpt:
        print("No checkpoint found. Tried:", CKPT_CANDIDATES); return

    print(f"Loading base model {BASE_MODEL} ...")
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, quantization_config=bnb, device_map="auto", trust_remote_code=True)
    print(f"Loading LoRA adapter from {ckpt} ...")
    model = PeftModel.from_pretrained(base, ckpt)
    model.eval()
    print("Model ready.\n")

    # ── Slumbot session ────────────────────────────────────────────────────────
    session = SlumbotSession(args.user, args.password)
    if not session.login():
        print("Login failed."); return
    print(f"Logged in as '{args.user}'. Playing {args.hands} hands vs Slumbot...\n")

    records = []
    for i in range(1, args.hands + 1):
        try:
            r = play_one_hand(session, model, tokenizer, i)
            if r: records.append(r)
            time.sleep(0.2)
        except Exception as e:
            print(f"  Hand {i} error: {e!r}"); time.sleep(1.0)

    # ── Save results ───────────────────────────────────────────────────────────
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_DIR / "rl_results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["hand","winnings_bb","position","hole","board","turns","real_code_turns","result"])
        w.writeheader()
        for rec in records:
            row = dict(rec); row["hole"] = " ".join(rec["hole"]); row["board"] = " ".join(rec["board"])
            w.writerow(row)

    n = len(records)
    winnings = [r["winnings_bb"] for r in records]
    total_bb = sum(winnings)
    wins = sum(1 for w in winnings if w > 0)
    real_code_rate = np.mean([r["real_code_turns"] / max(r["turns"],1) for r in records if r["turns"] > 0])

    summary = {
        "hands_played": n, "total_bb": round(total_bb,2),
        "bb_per_100": round(total_bb/n*100,2) if n else 0,
        "win_rate": round(wins/n,3) if n else 0,
        "real_code_rate": round(float(real_code_rate),3),
    }
    with open(RESULTS_DIR / "rl_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "="*54)
    print(f"  RLM (RL checkpoint) vs Slumbot — {n} hands")
    print("="*54)
    print(f"  Total:          {'+' if total_bb>=0 else ''}{total_bb:.1f} bb")
    print(f"  bb/100:         {total_bb/n*100:+.1f}")
    print(f"  Win rate:       {wins}/{n} ({wins/n*100:.0f}%)")
    print(f"  Real code used: {real_code_rate*100:.0f}% of decision turns")
    print("="*54)
    print(f"\nSaved to {RESULTS_DIR}/")

    # ── Comparison plot vs heuristic ───────────────────────────────────────────
    heuristic_bb100 = -73.0
    rl_bb100 = total_bb / n * 100 if n else 0
    heuristic_code = 0.0
    rl_code = real_code_rate * 100

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    fig.suptitle("RL Checkpoint vs Heuristic — Slumbot Comparison", fontsize=13, fontweight="bold")

    # bb/100
    ax = axes[0]
    bars = ax.bar(["Heuristic", "RL Checkpoint"], [heuristic_bb100, rl_bb100],
                  color=["#e74c3c", "#27ae60" if rl_bb100 > heuristic_bb100 else "#e67e22"], alpha=0.85)
    ax.axhline(0, color="grey", linewidth=0.8, linestyle="--")
    for b in bars:
        h = b.get_height()
        ax.text(b.get_x()+b.get_width()/2, h + (2 if h>=0 else -4),
                f"{h:+.1f}", ha="center", fontsize=12, fontweight="bold")
    ax.set_ylabel("bb / 100 hands"); ax.set_title("Win Rate (bb/100)")
    ax.grid(axis="y", alpha=0.3)

    # Real code usage
    ax = axes[1]
    bars2 = ax.bar(["Heuristic\n(no LLM)", "RL Checkpoint"], [heuristic_code, rl_code],
                   color=["#95a5a6", "#2980b9"], alpha=0.85)
    for b in bars2:
        h = b.get_height()
        ax.text(b.get_x()+b.get_width()/2, h+1, f"{h:.0f}%", ha="center", fontsize=12, fontweight="bold")
    ax.set_ylim(0, 115); ax.set_ylabel("% turns where model wrote real Python")
    ax.set_title("REPL Code Usage")
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    fig_path = RESULTS_DIR / "rl_vs_heuristic_slumbot.png"
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    print(f"Saved comparison plot → {fig_path}")

if __name__ == "__main__":
    main()
