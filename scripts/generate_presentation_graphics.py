"""
Presentation-ready results graphics generator.

Status: scaffold only. Not wired into any pipeline. Run manually when you want
a refreshed batch of figures for the final presentation deck.

Reads:
  - experiments/results/modal_shaped_leaderboard_*.json
  - docs/results/poker_zero_shot_baseline_25eps.txt (optional)

Writes (under figures/):
  - presentation_held_out_overlay.png   multi-run held-out accuracy overlay
  - presentation_delta_vs_baseline.png  per-run delta vs Exp B bar chart
  - presentation_run_summary.png        small-multiples per run
  - presentation_summary_table.md       slide-ready markdown table

Usage:
    python scripts/generate_presentation_graphics.py
    python scripts/generate_presentation_graphics.py --results-dir experiments/results
    python scripts/generate_presentation_graphics.py --only overlay,delta
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


EXP_B_BASELINE = 0.20
ZERO_SHOT_QWEN_7B = 0.08
HEURISTIC_CEILING = 1.00

PALETTE = [
    "#27ae60",
    "#2980b9",
    "#8e44ad",
    "#f39c12",
    "#16a085",
    "#c0392b",
    "#34495e",
]


@dataclass
class RunSeries:
    tag: str
    iterations: list[int]
    held_out_acc: list[float]

    @property
    def post_train_iters(self) -> list[int]:
        return [it for it in self.iterations if it > 0]

    @property
    def post_train_acc(self) -> list[float]:
        return [a for it, a in zip(self.iterations, self.held_out_acc) if it > 0]

    def peak(self) -> float:
        vals = self.post_train_acc
        return max(vals) if vals else 0.0

    def final(self) -> float:
        vals = self.post_train_acc
        return vals[-1] if vals else 0.0


def load_runs(results_dir: Path) -> list[RunSeries]:
    paths = sorted(results_dir.glob("modal_shaped_leaderboard_*.json"))
    runs: list[RunSeries] = []
    for p in paths:
        rows = json.loads(p.read_text())
        if not rows:
            continue
        tag = rows[0].get("run_tag") or p.stem.replace("modal_shaped_leaderboard_", "")
        runs.append(
            RunSeries(
                tag=tag,
                iterations=[r["iteration"] for r in rows],
                held_out_acc=[r["held_out_acc"] for r in rows],
            )
        )
    return runs


def _setup_axes(ax, title: str) -> None:
    ax.set_xlabel("REINFORCE iteration", fontsize=11)
    ax.set_ylabel("held-out action-type accuracy (%)", fontsize=11)
    ax.set_title(title, fontsize=12)
    ax.set_ylim(-5, 110)
    ax.grid(alpha=0.25)


def render_overlay(runs: list[RunSeries], out_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 5.5))

    expB_iters = list(range(10, 101, 10))
    ax.plot(
        expB_iters,
        [EXP_B_BASELINE * 100] * len(expB_iters),
        "o-",
        color="#c0392b",
        markersize=8,
        linewidth=2,
        label=f"Exp B baseline = {EXP_B_BASELINE:.0%}",
    )

    for i, run in enumerate(runs):
        color = PALETTE[i % len(PALETTE)]
        ax.plot(
            run.iterations,
            [a * 100 for a in run.held_out_acc],
            "s-",
            color=color,
            markersize=9,
            linewidth=2.4,
            label=run.tag,
        )

    ax.axhline(
        ZERO_SHOT_QWEN_7B * 100,
        color="#7f8c8d",
        ls="--",
        lw=1.2,
        alpha=0.7,
        label=f"zero-shot Qwen-7B = {ZERO_SHOT_QWEN_7B:.0%}",
    )
    ax.axhline(
        HEURISTIC_CEILING * 100,
        color="#2ecc71",
        ls="--",
        lw=1.2,
        alpha=0.4,
        label=f"heuristic ceiling = {HEURISTIC_CEILING:.0%}",
    )

    _setup_axes(ax, "Held-out eval: shaped-reward configurations vs Exp B")
    ax.legend(loc="upper left", fontsize=9)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")


def render_delta_bars(runs: list[RunSeries], out_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not runs:
        print("no runs to render delta bars for; skipping")
        return

    labels = [r.tag for r in runs]
    peak_delta = [(r.peak() - EXP_B_BASELINE) * 100 for r in runs]
    final_delta = [(r.final() - EXP_B_BASELINE) * 100 for r in runs]

    x = list(range(len(labels)))
    width = 0.38

    fig, ax = plt.subplots(figsize=(max(7, 1.2 * len(labels) + 4), 5))
    ax.bar(
        [xi - width / 2 for xi in x],
        peak_delta,
        width,
        label="peak  delta",
        color="#27ae60",
    )
    ax.bar(
        [xi + width / 2 for xi in x],
        final_delta,
        width,
        label="final delta",
        color="#2980b9",
    )

    ax.axhline(0, color="#34495e", lw=1.0)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=9)
    ax.set_ylabel("delta vs Exp B baseline (pp)", fontsize=11)
    ax.set_title("Held-out delta vs unshaped baseline (peak and final)", fontsize=12)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="best", fontsize=10)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")


def render_small_multiples(runs: list[RunSeries], out_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not runs:
        print("no runs to render small multiples for; skipping")
        return

    n = len(runs)
    cols = min(3, n)
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(4.5 * cols, 3.6 * rows), squeeze=False)
    for i, run in enumerate(runs):
        ax = axes[i // cols][i % cols]
        ax.plot(
            run.iterations,
            [a * 100 for a in run.held_out_acc],
            "s-",
            color=PALETTE[i % len(PALETTE)],
            markersize=8,
            linewidth=2.2,
        )
        ax.axhline(EXP_B_BASELINE * 100, color="#c0392b", ls="--", lw=1.0, alpha=0.7)
        ax.set_ylim(-5, 110)
        ax.grid(alpha=0.25)
        ax.set_title(run.tag, fontsize=10)
        ax.set_xlabel("iteration", fontsize=9)
        ax.set_ylabel("held-out acc (%)", fontsize=9)

    for j in range(n, rows * cols):
        axes[j // cols][j % cols].axis("off")

    fig.suptitle("Per-run held-out accuracy curves", fontsize=12)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")


def render_summary_table(runs: list[RunSeries], out_path: Path) -> None:
    lines = [
        "# Presentation summary table",
        "",
        f"Baseline (Exp B): **{EXP_B_BASELINE:.1%}** held-out action-type accuracy.",
        "",
        "| run | iterations | peak acc | final acc | peak delta | final delta |",
        "|---|---|---|---|---|---|",
    ]
    for run in runs:
        peak_d = (run.peak() - EXP_B_BASELINE) * 100
        final_d = (run.final() - EXP_B_BASELINE) * 100
        peak_sign = "+" if peak_d >= 0 else ""
        final_sign = "+" if final_d >= 0 else ""
        lines.append(
            f"| `{run.tag}` | {len(run.post_train_iters)} | "
            f"{run.peak():.1%} | {run.final():.1%} | "
            f"{peak_sign}{peak_d:.1f} pp | {final_sign}{final_d:.1f} pp |"
        )
    out_path.write_text("\n".join(lines) + "\n")
    print(f"wrote {out_path}")


RENDERERS = {
    "overlay": ("presentation_held_out_overlay.png", render_overlay),
    "delta": ("presentation_delta_vs_baseline.png", render_delta_bars),
    "small_multiples": ("presentation_run_summary.png", render_small_multiples),
    "table": ("presentation_summary_table.md", render_summary_table),
}


def parse_only(only: str | None) -> Iterable[str]:
    if not only:
        return RENDERERS.keys()
    requested = [s.strip() for s in only.split(",") if s.strip()]
    unknown = [r for r in requested if r not in RENDERERS]
    if unknown:
        raise SystemExit(f"unknown renderer(s): {unknown}; valid: {list(RENDERERS)}")
    return requested


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results-dir", default="experiments/results")
    p.add_argument("--out-dir", default="figures")
    p.add_argument(
        "--only",
        default=None,
        help="comma-separated subset of: overlay, delta, small_multiples, table",
    )
    args = p.parse_args()

    results_dir = Path(args.results_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    runs = load_runs(results_dir)
    if not runs:
        raise SystemExit(f"no leaderboards found under {results_dir}")
    print(f"loaded {len(runs)} run(s): {[r.tag for r in runs]}")

    for key in parse_only(args.only):
        filename, fn = RENDERERS[key]
        fn(runs, out_dir / filename)


if __name__ == "__main__":
    main()
