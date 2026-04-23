"""
Export shaped-reward leaderboards to a single tidy CSV.

Status: standalone helper. Not wired into any pipeline. Useful when you want
to drop the held-out eval results into a spreadsheet, share with a non-Python
collaborator, or feed an external plotting tool.

Reads:
  experiments/results/modal_shaped_leaderboard_*.json

Writes:
  experiments/results/leaderboards_combined.csv

Schema:
  run_tag, source_file, iteration, held_out_acc, delta_vs_expB_pp

Usage:
    python scripts/export_leaderboards_to_csv.py
    python scripts/export_leaderboards_to_csv.py --results-dir experiments/results \
        --out experiments/results/leaderboards_combined.csv
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


EXP_B_BASELINE = 0.20


def collect_rows(results_dir: Path) -> list[dict]:
    paths = sorted(results_dir.glob("modal_shaped_leaderboard_*.json"))
    out: list[dict] = []
    for p in paths:
        rows = json.loads(p.read_text())
        if not rows:
            continue
        default_tag = p.stem.replace("modal_shaped_leaderboard_", "")
        for r in rows:
            tag = r.get("run_tag") or default_tag
            held_out = float(r["held_out_acc"])
            out.append(
                {
                    "run_tag": tag,
                    "source_file": p.name,
                    "iteration": int(r["iteration"]),
                    "held_out_acc": held_out,
                    "delta_vs_expB_pp": round((held_out - EXP_B_BASELINE) * 100, 2),
                }
            )
    return out


def write_csv(rows: list[dict], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["run_tag", "source_file", "iteration", "held_out_acc", "delta_vs_expB_pp"]
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} row(s) to {out_path}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results-dir", default="experiments/results")
    p.add_argument("--out", default="experiments/results/leaderboards_combined.csv")
    args = p.parse_args()

    results_dir = Path(args.results_dir)
    rows = collect_rows(results_dir)
    if not rows:
        raise SystemExit(f"no leaderboards found under {results_dir}")
    write_csv(rows, Path(args.out))


if __name__ == "__main__":
    main()
