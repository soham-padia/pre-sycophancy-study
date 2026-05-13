#!/usr/bin/env python3
"""Cross-model first-flip analysis across all judged models.

For each model computes:
  - Ever-flip rate
  - First-flip turn distribution
  - Flip rate by question type
  - Average first-flip turn

Usage:
    python analysis/compare_models.py
    python analysis/compare_models.py --output analysis_claude/cross_model_comparison.txt
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
ANALYSIS_DIR = REPO_ROOT / "analysis_claude"

MODEL_FILES = {
    "DeepSeek-R1-Distill-Qwen-7B": ANALYSIS_DIR / "claude_judgements.csv",
    "Gemma-2-9B":                  ANALYSIS_DIR / "gemma_judgements_haiku.csv",
    "Llama-3.1-8B-Instruct":       ANALYSIS_DIR / "llama31_judgements_haiku.csv",
    "Qwen2.5-7B-Instruct":         ANALYSIS_DIR / "qwen25_judgements_haiku.csv",
    "Qwen3.5-9B":                  ANALYSIS_DIR / "qwen35_judgements_haiku.csv",
}

DEFAULT_OUTPUT = ANALYSIS_DIR / "cross_model_comparison.txt"


def load_model(path: Path, model_name: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["judgement"] = df["judgement"].astype(str).str.lower() == "true"
    df["model_label"] = model_name
    return df


def first_flip_stats(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["question_type", "question", "turn"])

    def _first_flip(group):
        flipped = group[group["judgement"] == True]
        if len(flipped) == 0:
            return pd.Series({"ever_flipped": False, "first_flip_turn": None})
        return pd.Series({"ever_flipped": True, "first_flip_turn": int(flipped["turn"].min())})

    return (
        df.groupby(["question_type", "question"])
        .apply(_first_flip, include_groups=False)
        .reset_index()
    )


def section(title: str) -> str:
    return f"\n{'=' * 70}\n{title}\n{'=' * 70}"


def fmt_pct(n, d):
    return f"{n}/{d} ({100 * n / d:.1f}%)" if d else "0/0"


def build_report(model_dfs: dict[str, pd.DataFrame]) -> str:
    lines = ["CROSS-MODEL FIRST-FLIP ANALYSIS  (Judge: Claude Haiku 4.5)"]

    # ------------------------------------------------------------------ #
    # Overall ever-flip rate per model
    # ------------------------------------------------------------------ #
    lines.append(section("EVER-FLIP RATE BY MODEL"))
    lines.append(f"  {'Model':<35} {'Ever-flipped':>15}  {'Never-flipped':>15}  {'Avg 1st flip turn':>18}")

    rows_by_model = {}
    for model_name, df in model_dfs.items():
        stats = first_flip_stats(df)
        rows_by_model[model_name] = stats
        n = len(stats)
        flipped = stats["ever_flipped"].sum()
        avg_turn = stats[stats["ever_flipped"]]["first_flip_turn"].mean()
        lines.append(
            f"  {model_name:<35} {fmt_pct(flipped, n):>15}  {fmt_pct(n - flipped, n):>15}  {avg_turn:>18.2f}"
        )

    # ------------------------------------------------------------------ #
    # First-flip turn distribution per model
    # ------------------------------------------------------------------ #
    lines.append(section("FIRST-FLIP TURN DISTRIBUTION"))
    lines.append(f"  {'Model':<35} {'Turn1':>7} {'Turn2':>7} {'Turn3':>7} {'Turn4':>7} {'Turn5':>7}")
    for model_name, stats in rows_by_model.items():
        flipped = stats[stats["ever_flipped"]]
        counts = flipped["first_flip_turn"].value_counts().sort_index()
        total_flipped = len(flipped)
        row = f"  {model_name:<35}"
        for t in range(1, 6):
            c = counts.get(t, 0)
            row += f" {c:>4}({100*c/total_flipped:.0f}%)" if total_flipped else f" {'0(0%)':>7}"
        lines.append(row)

    # ------------------------------------------------------------------ #
    # Ever-flip rate by question type per model
    # ------------------------------------------------------------------ #
    lines.append(section("EVER-FLIP RATE BY QUESTION TYPE"))
    qtypes = ["base", "critical", "presupposition"]
    lines.append(f"  {'Model':<35} {'base':>12} {'critical':>12} {'presupposition':>16}")
    for model_name, df in model_dfs.items():
        stats = first_flip_stats(df)
        row = f"  {model_name:<35}"
        for qtype in qtypes:
            grp = stats[stats["question_type"] == qtype]
            if len(grp) == 0:
                row += f" {'N/A':>12}"
                continue
            flipped = grp["ever_flipped"].sum()
            row += f" {fmt_pct(flipped, len(grp)):>12}"
        lines.append(row)

    # ------------------------------------------------------------------ #
    # Flip rate by turn (per-turn, for reference)
    # ------------------------------------------------------------------ #
    lines.append(section("PER-TURN FLIP RATE (first-flip turns only, i.e. where flip first occurred)"))
    lines.append(f"  {'Model':<35} {'Turn1':>8} {'Turn2':>8} {'Turn3':>8} {'Turn4':>8} {'Turn5':>8}")
    for model_name, stats in rows_by_model.items():
        flipped = stats[stats["ever_flipped"]]
        total_q = len(stats)
        row = f"  {model_name:<35}"
        for t in range(1, 6):
            c = (flipped["first_flip_turn"] == t).sum()
            row += f" {100*c/total_q:>7.1f}%"
        lines.append(row)

    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    model_dfs = {}
    for model_name, path in MODEL_FILES.items():
        if not path.exists():
            print(f"[skip] {model_name} — file not found: {path}")
            continue
        model_dfs[model_name] = load_model(path, model_name)
        print(f"[loaded] {model_name}: {len(model_dfs[model_name])} rows")

    if not model_dfs:
        raise SystemExit("No model files found.")

    report = build_report(model_dfs)
    print(report)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report + "\n", encoding="utf-8")
    print(f"\nReport saved to {args.output}")


if __name__ == "__main__":
    main()
