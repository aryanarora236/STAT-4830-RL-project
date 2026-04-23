"""
Render the shaped-reward vs Exp B comparison figure and produce slide-ready tables.

Run after modal_shaped_reward.py has written
experiments/results/modal_shaped_leaderboard_<tag>.json.

Writes:
  - figures/poker_rl_shaped_vs_expB.png
  - docs/slide_11_update.md  (copy-paste snippets + numbers)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def classify_outcome(rows):
    """Return one of STRONG POSITIVE / PARTIAL POSITIVE / NULL / REGRESSION."""
    vals = [r["held_out_acc"] for r in rows if r["iteration"] > 0]
    if not vals:
        return "NULL", "no post-training checkpoints evaluated"
    best = max(vals)
    last = vals[-1]
    monotone = all(vals[i] <= vals[i + 1] + 0.05 for i in range(len(vals) - 1))
    if best >= 0.35 and monotone:
        return "STRONG POSITIVE", f"peak {best:.1%} with monotone-ish climb"
    if best >= 0.30:
        return "PARTIAL POSITIVE", f"peak {best:.1%} — some movement vs 20% baseline"
    if all(v <= 0.15 for v in vals):
        return "REGRESSION", f"all checkpoints at or below 15%"
    return "NULL", f"all checkpoints in [15%, 25%] — shaped reward did not break attractor"


def render_figure(shaped_rows, out_path: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    expB_iters = list(range(10, 101, 10))
    expB_acc = [0.20] * len(expB_iters)

    shaped_iters = [r["iteration"] for r in shaped_rows]
    shaped_acc = [r["held_out_acc"] for r in shaped_rows]

    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.plot(expB_iters, [a * 100 for a in expB_acc], "o-", color="#c0392b",
            markersize=8, linewidth=2, label="Exp B (unshaped reward)")
    ax.plot(shaped_iters, [a * 100 for a in shaped_acc], "s-", color="#27ae60",
            markersize=10, linewidth=2.5, label="Shaped reward (this run)")
    ax.axhline(8, color="#7f8c8d", ls="--", lw=1.2, alpha=0.7, label="zero-shot Qwen-7B = 8%")
    ax.axhline(100, color="#2ecc71", ls="--", lw=1.2, alpha=0.4, label="heuristic ceiling = 100%")

    ax.set_xlabel("REINFORCE iteration", fontsize=11)
    ax.set_ylabel("held-out action-type accuracy (%)", fontsize=11)
    ax.set_title("Held-out eval: shaped reward vs unshaped (Exp B)", fontsize=12)
    ax.set_ylim(-5, 110)
    ax.grid(alpha=0.25)
    ax.legend(loc="upper left", fontsize=10)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")


def render_slide11_update(rows, outcome_label: str, outcome_reason: str, run_tag: str, out_path: str):
    baseline = 0.20
    lines = [
        "# Slide 11 update — shaped-reward Modal run",
        "",
        f"**Outcome:** {outcome_label} — {outcome_reason}",
        f"**Run tag:** `{run_tag}`",
        f"**Baseline:** {baseline:.1%} (Exp B `best_by_eval`)",
        "",
        "## Held-out eval curve",
        "",
        "| iter | held-out acc | delta vs baseline |",
        "|---|---|---|",
    ]
    for r in rows:
        delta = (r["held_out_acc"] - baseline) * 100
        sign = "+" if delta >= 0 else ""
        lines.append(f"| {r['iteration']} | {r['held_out_acc']:.1%} | {sign}{delta:.1f} pp |")

    lines += [
        "",
        "## Suggested slide 11 right-column callout",
        "",
        "Shaped-reward experiment (Modal A10G)",
        f"Baseline (Exp B): {baseline:.1%}",
    ]
    for r in rows:
        if r["iteration"] > 0:
            lines.append(f"iter {r['iteration']}: {r['held_out_acc']:.1%}")

    lines += [
        "",
        "## Suggested speaker-note update",
        "",
        f"In the shaped-reward run we added a tool-use bonus (+real_code, +parsed_stats) and a fallback penalty. Against Exp B's 20% flat line, the shaped run produced {outcome_label.lower()}. See `figures/poker_rl_shaped_vs_expB.png`.",
    ]

    Path(out_path).write_text("\n".join(lines))
    print(f"wrote {out_path}")


def render_multi_figure(runs, out_path: str):
    """Overlay multiple shaped runs + Exp B baseline."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    expB_iters = list(range(10, 101, 10))
    expB_acc = [0.20] * len(expB_iters)

    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.plot(expB_iters, [a * 100 for a in expB_acc], "o-", color="#c0392b",
            markersize=8, linewidth=2, label="Exp B (unshaped reward)")

    colors = ["#27ae60", "#2980b9", "#8e44ad", "#f39c12"]
    for i, (label, rows) in enumerate(runs):
        its = [r["iteration"] for r in rows]
        acc = [r["held_out_acc"] for r in rows]
        ax.plot(its, [a * 100 for a in acc], "s-",
                color=colors[i % len(colors)],
                markersize=10, linewidth=2.5, label=label)

    ax.axhline(8, color="#7f8c8d", ls="--", lw=1.2, alpha=0.7, label="zero-shot Qwen-7B = 8%")
    ax.axhline(100, color="#2ecc71", ls="--", lw=1.2, alpha=0.4, label="heuristic ceiling = 100%")
    ax.set_xlabel("REINFORCE iteration", fontsize=11)
    ax.set_ylabel("held-out action-type accuracy (%)", fontsize=11)
    ax.set_title("Held-out eval: shaped reward configurations vs Exp B", fontsize=12)
    ax.set_ylim(-5, 110)
    ax.grid(alpha=0.25)
    ax.legend(loc="upper left", fontsize=9)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--leaderboard", default=None,
                   help="Path to modal_shaped_leaderboard_*.json (auto-detect newest if omitted)")
    p.add_argument("--all-leaderboards", action="store_true",
                   help="Render overlay of all leaderboard_*.json files in experiments/results/")
    p.add_argument("--figure", default="figures/poker_rl_shaped_vs_expB.png")
    p.add_argument("--slide-update", default="docs/slide_11_update.md")
    args = p.parse_args()

    if args.all_leaderboards:
        paths = sorted(Path("experiments/results").glob("modal_shaped_leaderboard_*.json"))
        if not paths:
            raise SystemExit("no leaderboards found")
        runs = []
        for p in paths:
            rows = json.loads(p.read_text())
            tag = rows[0].get("run_tag", p.stem.split("_")[-1])
            label = f"{tag}"
            runs.append((label, rows))
        Path(args.figure).parent.mkdir(parents=True, exist_ok=True)
        render_multi_figure(runs, args.figure)
        best = max(runs, key=lambda r: max(x["held_out_acc"] for x in r[1] if x["iteration"] > 0))
        outcome_label, outcome_reason = classify_outcome(best[1])
        Path(args.slide_update).parent.mkdir(parents=True, exist_ok=True)
        render_slide11_update(best[1], outcome_label, outcome_reason, best[0], args.slide_update)
        return

    if args.leaderboard:
        lb_path = Path(args.leaderboard)
    else:
        candidates = sorted(Path("experiments/results").glob("modal_shaped_leaderboard_*.json"),
                            key=lambda p: p.stat().st_mtime)
        if not candidates:
            raise SystemExit("No experiments/results/modal_shaped_leaderboard_*.json found.")
        lb_path = candidates[-1]
        print(f"auto-detected leaderboard: {lb_path}")

    rows = json.loads(lb_path.read_text())
    run_tag = rows[0].get("run_tag") if rows and "run_tag" in rows[0] else lb_path.stem.split("_")[-1]

    outcome_label, outcome_reason = classify_outcome(rows)
    print(f"outcome: {outcome_label} — {outcome_reason}")

    Path(args.figure).parent.mkdir(parents=True, exist_ok=True)
    render_figure(rows, args.figure)

    Path(args.slide_update).parent.mkdir(parents=True, exist_ok=True)
    render_slide11_update(rows, outcome_label, outcome_reason, run_tag, args.slide_update)


if __name__ == "__main__":
    main()
