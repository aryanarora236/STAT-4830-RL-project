"""
Print a head-to-head markdown comparison between two shaped-reward leaderboards.

Reads two leaderboard JSONs and emits a markdown table aligning iterations
across both runs, with per-iteration delta and overall peak/final summary.

Usage:
    python scripts/compare_runs.py \
        experiments/results/modal_shaped_leaderboard_v1det.json \
        experiments/results/modal_shaped_leaderboard_v2.json

    # write to file instead of stdout
    python scripts/compare_runs.py a.json b.json --out docs/results/compare_v1_v2.md

    # override labels (default: derived from run_tag or filename)
    python scripts/compare_runs.py a.json b.json --label-a v1-detached --label-b v2-tuned
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


EXP_B_BASELINE = 0.20


def load_leaderboard(path: Path) -> tuple[str, dict[int, float]]:
    rows = json.loads(path.read_text())
    if not rows:
        raise SystemExit(f"{path} is empty")
    tag = rows[0].get("run_tag") or path.stem.replace("modal_shaped_leaderboard_", "")
    series = {int(r["iteration"]): float(r["held_out_acc"]) for r in rows}
    return tag, series


def render_comparison(
    label_a: str,
    series_a: dict[int, float],
    label_b: str,
    series_b: dict[int, float],
) -> str:
    iterations = sorted(set(series_a) | set(series_b))

    lines = [
        f"# Comparison: `{label_a}` vs `{label_b}`",
        "",
        f"Baseline (Exp B): {EXP_B_BASELINE:.1%}",
        "",
        f"| iter | {label_a} | {label_b} | b - a (pp) |",
        "|---|---|---|---|",
    ]
    for it in iterations:
        a = series_a.get(it)
        b = series_b.get(it)
        a_str = f"{a:.1%}" if a is not None else "—"
        b_str = f"{b:.1%}" if b is not None else "—"
        if a is not None and b is not None:
            d = (b - a) * 100
            sign = "+" if d >= 0 else ""
            d_str = f"{sign}{d:.1f}"
        else:
            d_str = "—"
        lines.append(f"| {it} | {a_str} | {b_str} | {d_str} |")

    def post_train_stats(series: dict[int, float]) -> tuple[float, float]:
        post = [(it, v) for it, v in series.items() if it > 0]
        if not post:
            return 0.0, 0.0
        post.sort()
        peak = max(v for _, v in post)
        final = post[-1][1]
        return peak, final

    peak_a, final_a = post_train_stats(series_a)
    peak_b, final_b = post_train_stats(series_b)

    lines += [
        "",
        "## Summary",
        "",
        f"- `{label_a}` peak: {peak_a:.1%}, final: {final_a:.1%}",
        f"- `{label_b}` peak: {peak_b:.1%}, final: {final_b:.1%}",
        f"- peak delta (b - a): {(peak_b - peak_a) * 100:+.1f} pp",
        f"- final delta (b - a): {(final_b - final_a) * 100:+.1f} pp",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("a", help="first leaderboard JSON")
    p.add_argument("b", help="second leaderboard JSON")
    p.add_argument("--label-a", default=None, help="override label for run A")
    p.add_argument("--label-b", default=None, help="override label for run B")
    p.add_argument("--out", default=None, help="write markdown to this path (default: stdout)")
    args = p.parse_args()

    tag_a, series_a = load_leaderboard(Path(args.a))
    tag_b, series_b = load_leaderboard(Path(args.b))

    label_a = args.label_a or tag_a
    label_b = args.label_b or tag_b

    md = render_comparison(label_a, series_a, label_b, series_b)

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(md)
        print(f"wrote {out_path}", file=sys.stderr)
    else:
        sys.stdout.write(md)


if __name__ == "__main__":
    main()
