#!/usr/bin/env python3
"""Plot question-level sycophancy flip rates by model and question type.

This script reads false-presupposition metadata JSON artifacts from `data/`,
labels turn-level flips using the same keyword heuristic as the probe code,
and produces:

1. A summary CSV with counts and rates.
2. A grouped bar chart of "ever flipped" question rate by model/question type.

Usage:
    python analysis/plot_flip_rates.py
"""

from __future__ import annotations

import csv
import json
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"
OUTPUT_DIR = REPO_ROOT / "analysis" / "plots"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from flip_labeling import response_flipped

MODELS = {
    "DeepSeek-R1-Distill-Qwen-7B": DATA_DIR / "DeepSeek-R1-Distill-Qwen-7B",
    "Llama-3.1-8B-Instruct": DATA_DIR / "Llama-3.1-8B-Instruct",
    "Gemma-2-9B": DATA_DIR / "Gemma-2-9B",
    "Qwen2.5-7B-Instruct": DATA_DIR / "Qwen2.5-7B-Instruct",
    "Qwen3.5-9B": DATA_DIR / "Qwen3.5-9B",
}

QUESTION_TYPES = ["base", "critical", "presupposition"]
def resolve_model_dir(model_dir: Path) -> Path:
    """Support either direct model folders or a single nested model subfolder."""
    if (model_dir / "base_multiturn_metadata.json").exists():
        return model_dir

    children = [p for p in model_dir.iterdir() if p.is_dir() and not p.name.startswith(".")] if model_dir.exists() else []
    for child in children:
        if (child / "base_multiturn_metadata.json").exists():
            return child

    return model_dir


def load_flip_stats(model_dir: Path, question_type: str) -> dict | None:
    model_dir = resolve_model_dir(model_dir)
    metadata_path = model_dir / f"{question_type}_multiturn_metadata.json"
    if not metadata_path.exists():
        return None

    with metadata_path.open(encoding="utf-8") as f:
        metadata = json.load(f)

    total_questions = 0
    ever_flip_questions = 0
    total_flip_events = 0

    for question, turns in metadata.items():
        del question  # question text is not needed for aggregate stats
        per_question_flips = []
        for turn in range(1, 6):
            turn_key = f"Turn_{turn}"
            response = turns.get(turn_key, {}).get("assistant_response", "")
            flipped = response_flipped(response)
            per_question_flips.append(flipped)

        total_questions += 1
        total_flip_events += sum(per_question_flips)
        if any(per_question_flips):
            ever_flip_questions += 1

    if total_questions == 0:
        return {
            "questions": 0,
            "ever_flip_questions": 0,
            "ever_flip_rate": 0.0,
            "flip_events": 0,
            "event_rate": 0.0,
        }

    return {
        "questions": total_questions,
        "ever_flip_questions": ever_flip_questions,
        "ever_flip_rate": ever_flip_questions / total_questions,
        "flip_events": total_flip_events,
        "event_rate": total_flip_events / (total_questions * 5),
    }


def collect_results() -> list[dict]:
    rows = []
    for model_name, model_dir in MODELS.items():
        for question_type in QUESTION_TYPES:
            stats = load_flip_stats(model_dir, question_type)
            if stats is None:
                continue
            rows.append({
                "model": model_name,
                "question_type": question_type,
                **stats,
            })
    return rows


def write_summary_csv(rows: list[dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "model",
        "question_type",
        "questions",
        "ever_flip_questions",
        "ever_flip_rate",
        "flip_events",
        "event_rate",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_grouped_bars(rows: list[dict], output_path: Path) -> None:
    available_models = [model for model in MODELS if any(r["model"] == model for r in rows)]
    if not available_models:
        raise RuntimeError("No model metadata files found under data/.")

    x = np.arange(len(QUESTION_TYPES))
    width = 0.8 / max(len(available_models), 1)

    fig, ax = plt.subplots(figsize=(12, 7))

    for i, model_name in enumerate(available_models):
        rates = []
        for question_type in QUESTION_TYPES:
            match = next(
                (r for r in rows if r["model"] == model_name and r["question_type"] == question_type),
                None,
            )
            rates.append((match["ever_flip_rate"] * 100.0) if match else 0.0)

        offsets = x - 0.4 + (i + 0.5) * width
        bars = ax.bar(offsets, rates, width=width, label=model_name)

        for bar, rate in zip(bars, rates):
            if rate > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.8,
                    f"{rate:.1f}%",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                    rotation=90,
                )

    ax.set_title("Flip Rate by Model and Question Type", fontsize=14)
    ax.set_ylabel("Questions with at least one flip (%)")
    ax.set_xlabel("Question type")
    ax.set_xticks(x)
    ax.set_xticklabels([q.title() for q in QUESTION_TYPES])
    ax.set_ylim(0, 100)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(title="Model", fontsize=9)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    rows = collect_results()
    if not rows:
        raise SystemExit("No metadata files found. Download model output folders into data/ first.")

    summary_csv = OUTPUT_DIR / "flip_rate_summary.csv"
    chart_png = OUTPUT_DIR / "flip_rate_by_model_question_type.png"

    write_summary_csv(rows, summary_csv)
    plot_grouped_bars(rows, chart_png)

    print(f"Wrote summary CSV: {summary_csv}")
    print(f"Wrote bar chart:   {chart_png}")


if __name__ == "__main__":
    main()
