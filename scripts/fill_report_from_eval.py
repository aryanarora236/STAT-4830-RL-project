"""
Fill [PLACEHOLDER] tokens in report.md / README.md / self_critique_week15.md /
docs/final_slides_outline.md from an experiments/results/final_eval_*.json file.

Usage:
    python scripts/fill_report_from_eval.py \
        --eval experiments/results/final_eval_20260420_230000.json \
        --zs-eval experiments/results/zero_shot_eval.json  # optional
        [--dry-run]
        [--files report.md self_critique_week15.md README.md docs/final_slides_outline.md]

The script expects the JSON schema produced by PokerEvaluationFramework.save_results_json():

    {
      "meta": {...},
      "suites": {
        "All Streets":  {"agents": {"<agent>": {...metrics...}}},
        "Preflop":      {"agents": {...}},
        "Postflop":     {"agents": {...}}
      }
    }

Agent names are matched case-insensitively against substrings. Use --bc-agent /
--rl-agent / --zs-agent to override.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Tuple


DEFAULT_FILES = [
    "report.md",
    "self_critique_week15.md",
    "README.md",
    "docs/final_slides_outline.md",
]


@dataclass
class AgentMetrics:
    all_streets_acc: Optional[float] = None
    preflop_acc: Optional[float] = None
    postflop_acc: Optional[float] = None
    avg_reward: Optional[float] = None
    avg_steps: Optional[float] = None
    by_action: Dict[str, float] = None
    total_all_streets: Optional[int] = None

    def as_placeholders(self, prefix: str) -> Dict[str, str]:
        out: Dict[str, str] = {}
        if self.all_streets_acc is not None:
            out[f"{prefix}_ALL"] = f"{self.all_streets_acc * 100:.1f}"
        if self.preflop_acc is not None:
            out[f"{prefix}_PRE"] = f"{self.preflop_acc * 100:.1f}"
        if self.postflop_acc is not None:
            out[f"{prefix}_POST"] = f"{self.postflop_acc * 100:.1f}"
        if self.avg_reward is not None:
            out[f"{prefix}_R"] = f"{self.avg_reward:.3f}"
        if self.avg_steps is not None:
            out[f"{prefix}_S"] = f"{self.avg_steps:.1f}"
        if self.by_action:
            for action in ("fold", "check", "call", "raise"):
                rate = self.by_action.get(action)
                if rate is not None:
                    out[f"{prefix}_{action.upper()}"] = f"{rate * 100:.1f}"
        if self.total_all_streets is not None:
            out[f"{prefix}_N"] = str(self.total_all_streets)
        return out


def _pick_agent(agents: Dict[str, Dict], match: str) -> Tuple[str, Dict]:
    match_lc = match.lower()
    for name, metrics in agents.items():
        if match_lc in name.lower():
            return name, metrics
    raise KeyError(f"no agent matching '{match}' in {list(agents)}")


def _extract_metrics(suites: Dict[str, Dict], match: str) -> AgentMetrics:
    """Pull per-suite accuracy for one agent."""
    m = AgentMetrics()

    def suite_acc(key: str) -> Optional[float]:
        for sk, s in suites.items():
            if key.lower() in sk.lower():
                try:
                    _, am = _pick_agent(s.get("agents", {}), match)
                except KeyError:
                    return None
                return am.get("type_match_rate")
        return None

    m.all_streets_acc = suite_acc("all")
    m.preflop_acc = suite_acc("preflop")
    m.postflop_acc = suite_acc("postflop")

    # Other metrics come from the all-streets suite when available.
    for sk, s in suites.items():
        if "all" in sk.lower():
            try:
                _, am = _pick_agent(s.get("agents", {}), match)
            except KeyError:
                continue
            m.avg_reward = am.get("avg_reward")
            m.avg_steps = am.get("avg_steps")
            m.total_all_streets = am.get("total_episodes")
            ba_raw = am.get("by_action", {}) or {}
            by_action: Dict[str, float] = {}
            for action, counts in ba_raw.items():
                total = counts.get("total") or 0
                correct = counts.get("correct") or 0
                by_action[action] = (correct / total) if total else 0.0
            m.by_action = by_action
            break
    return m


def _load_suites(path: str) -> Dict[str, Dict]:
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    if "suites" in payload:
        return payload["suites"]
    # Compatibility: single-suite export_run_summary() payload.
    return {"All Streets": payload}


def _replace_placeholders(text: str, placeholders: Dict[str, str]) -> Tuple[str, int]:
    total_replaced = 0
    for key, value in placeholders.items():
        pattern = re.compile(r"\[" + re.escape(key) + r"\]")
        text, n = pattern.subn(value, text)
        total_replaced += n
    return text, total_replaced


def main(argv: Iterable[str] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Fill [PLACEHOLDER] tokens from a final_eval JSON.",
    )
    parser.add_argument(
        "--eval",
        required=True,
        help="Path to the BC+RL eval JSON (final_eval_*.json).",
    )
    parser.add_argument(
        "--zs-eval",
        default=None,
        help="Optional separate JSON from a zero-shot-only eval run.",
    )
    parser.add_argument(
        "--bc-agent",
        default="BC",
        help="Substring matching the BC agent name in eval JSON (default: 'BC').",
    )
    parser.add_argument(
        "--rl-agent",
        default="RL",
        help="Substring matching the RL agent name in eval JSON (default: 'RL').",
    )
    parser.add_argument(
        "--zs-agent",
        default="ZeroShot",
        help="Substring matching the zero-shot agent name (default: 'ZeroShot').",
    )
    parser.add_argument(
        "--files",
        nargs="*",
        default=DEFAULT_FILES,
        help="Files to patch. Default: " + ", ".join(DEFAULT_FILES),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print replacements without writing files.",
    )
    args = parser.parse_args(argv)

    suites = _load_suites(args.eval)
    placeholders: Dict[str, str] = {}

    bc_metrics = _extract_metrics(suites, args.bc_agent)
    placeholders.update(bc_metrics.as_placeholders("BC"))
    placeholders["BC_ACC"] = placeholders.get("BC_ALL", "")

    rl_metrics = _extract_metrics(suites, args.rl_agent)
    placeholders.update(rl_metrics.as_placeholders("RL"))
    placeholders["RL_ACC"] = placeholders.get("RL_ALL", "")

    zs_source_suites = _load_suites(args.zs_eval) if args.zs_eval else suites
    try:
        zs_metrics = _extract_metrics(zs_source_suites, args.zs_agent)
        placeholders.update(zs_metrics.as_placeholders("ZS"))
        if zs_metrics.all_streets_acc is not None:
            placeholders["ZS_1_5B"] = f"{zs_metrics.all_streets_acc * 100:.1f}"
        if zs_metrics.total_all_streets is not None:
            placeholders["ZS_N"] = str(zs_metrics.total_all_streets)
        # Per-action also under ZS_TM for exact-match vs type-match (alias).
        placeholders["ZS_EX"] = placeholders.get("ZS_ALL", "")
        placeholders["ZS_TM"] = placeholders.get("ZS_ALL", "")
    except KeyError:
        print("[warn] no zero-shot agent found in eval JSON; ZS_* tokens left alone.", file=sys.stderr)

    print("Placeholders resolved:")
    for k, v in sorted(placeholders.items()):
        print(f"  {k} = {v}")

    any_replaced = 0
    for relpath in args.files:
        path = os.path.abspath(relpath)
        if not os.path.exists(path):
            print(f"[skip] {relpath} (not found)")
            continue
        with open(path, "r", encoding="utf-8") as f:
            original = f.read()
        patched, n = _replace_placeholders(original, placeholders)
        print(f"  {relpath}: {n} replacement(s)")
        any_replaced += n
        if n == 0 or args.dry_run:
            continue
        with open(path, "w", encoding="utf-8") as f:
            f.write(patched)

    if args.dry_run:
        print("[dry-run] no files modified.")
    elif any_replaced == 0:
        print("No placeholders matched in any file. Is this the right eval JSON?")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
