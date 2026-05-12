#!/usr/bin/env python3
"""
Regenerate REINFORCE learning curves from training_history.csv.

The poker trainer also saves training_curves.png at the end of a run; use this
script when you only have the CSV/JSON checked in, or to customize the figure.

  python scripts/plot_training_history.py
  python scripts/plot_training_history.py docs/results/poker_rl_medium_simple_20260420/training_history.csv -o /tmp/curves.png
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from typing import List

DEFAULT_CSV = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "docs",
    "results",
    "poker_rl_long_simple_120iters_20260420",
    "training_history.csv",
)


def _read_csv(path: str) -> tuple[List[float], List[float], List[float], List[float]]:
    acc_raw: List[float] = []
    r_raw: List[float] = []
    r_ema: List[float] = []
    a_ema: List[float] = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            acc_raw.append(float(row["accuracy"]))
            r_raw.append(float(row["avg_reward"]))
            r_ema.append(float(row["reward_ema"]))
            a_ema.append(float(row["accuracy_ema"]))
    return acc_raw, r_raw, r_ema, a_ema


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "csv_path",
        nargs="?",
        default=DEFAULT_CSV,
        help="Path to training_history.csv",
    )
    parser.add_argument(
        "-o",
        "--out",
        default=None,
        help="Output PNG (default: same directory as csv, name training_curves.png)",
    )
    parser.add_argument(
        "--ema-gamma",
        type=float,
        default=0.9,
        help="Label for legend (should match the run; does not recompute EMA)",
    )
    args = parser.parse_args()

    if not os.path.isfile(args.csv_path):
        print(f"File not found: {args.csv_path}", file=sys.stderr)
        sys.exit(1)

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as e:
        print("Install matplotlib: pip install matplotlib", file=sys.stderr)
        raise SystemExit(1) from e

    acc_raw, r_raw, r_ema, a_ema = _read_csv(args.csv_path)
    n: int = len(acc_raw)
    iters: List[int] = list(range(1, n + 1))

    out_path: str
    if args.out:
        out_path = args.out
    else:
        out_path = os.path.join(os.path.dirname(args.csv_path), "training_curves.png")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].plot(iters, r_raw, "o-", alpha=0.3, color="#3498db", markersize=3, label="Raw (per batch)")
    axes[0].plot(iters, r_ema, "-", color="#2c3e50", linewidth=2, label=f"EMA (gamma={args.ema_gamma})")
    axes[0].set_xlabel("Iteration")
    axes[0].set_ylabel("Reward")
    axes[0].set_title("Reward per batch + EMA")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(iters, acc_raw, "o-", alpha=0.3, color="#e74c3c", markersize=3, label="Raw (per batch)")
    axes[1].plot(iters, a_ema, "-", color="#2c3e50", linewidth=2, label=f"EMA (gamma={args.ema_gamma})")
    axes[1].set_xlabel("Iteration")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title("Accuracy per batch + EMA")
    axes[1].set_ylim(-0.05, 1.05)
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    parent: str = os.path.dirname(os.path.abspath(out_path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
