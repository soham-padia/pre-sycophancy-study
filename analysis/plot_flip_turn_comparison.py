#!/usr/bin/env python3
"""Stacked-bar keyword-vs-judge first-flip comparison (appendix figure).

Rows: labeling method (keyword / LLM-as-judge); columns: prompt condition.
Bar height = ever-flip rate, segmented by first-flip turn (T1..T5).
"""
from pathlib import Path
import sys
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
from flip_labeling import response_flipped

OUT_PNG = REPO_ROOT / "analysis_claude" / "flip_turn_comparison.png"

MODELS = ["DeepSeek-R1-7B", "Gemma-2-9B", "Llama-3.1-8B", "Qwen2.5-7B", "Qwen3.5-9B"]

KEYWORD_DIRS = {
    "DeepSeek-R1-7B": REPO_ROOT / "data" / "DeepSeek-R1-Distill-Qwen-7B",
    "Gemma-2-9B":     REPO_ROOT / "data" / "Gemma-2-9B",
    "Llama-3.1-8B":   REPO_ROOT / "data" / "Llama-3.1-8B-Instruct",
    "Qwen2.5-7B":     REPO_ROOT / "data" / "Qwen2.5-7B-Instruct",
    "Qwen3.5-9B":     REPO_ROOT / "data" / "Qwen3.5-9B",
}

JUDGE_CSVS = {
    "DeepSeek-R1-7B": (REPO_ROOT / "analysis_claude" / "claude_judgements.csv",        "DeepSeek-R1-Distill-Qwen-7B"),
    "Gemma-2-9B":     (REPO_ROOT / "analysis_claude" / "gemma_judgements_haiku.csv",   "Gemma-2-9B"),
    "Llama-3.1-8B":   (REPO_ROOT / "analysis_claude" / "llama31_judgements_haiku.csv", "Llama-3.1-8B-Instruct"),
    "Qwen2.5-7B":     (REPO_ROOT / "analysis_claude" / "qwen25_judgements_haiku.csv",  "Qwen2.5-7B-Instruct"),
    "Qwen3.5-9B":     (REPO_ROOT / "analysis_claude" / "qwen35_judgements_haiku.csv",  "Qwen3.5-9B"),
}

CONDITIONS  = ["base", "critical", "presupposition"]
COND_LABELS = ["Base", "Critical", "Presupposition"]

SHORT_NAMES = {
    "DeepSeek-R1-7B": "DS",
    "Gemma-2-9B":     "G2",
    "Llama-3.1-8B":   "L3.1",
    "Qwen2.5-7B":     "Q2.5",
    "Qwen3.5-9B":     "Q3.5",
}

TURN_COLORS = ["#c6dbef", "#9ecae1", "#6baed6", "#3182bd", "#08519c"]


def keyword_first_flip_dist(model, cond):
    csv_path = KEYWORD_DIRS[model] / f"{cond}_multiturn.csv"
    if not csv_path.exists():
        return {t: 0.0 for t in range(1, 6)}
    df = pd.read_csv(csv_path)
    turn_cols = [c for c in (f"Response_Turn_{t}" for t in range(1, 6)) if c in df.columns]
    dist = {t: 0 for t in range(1, 6)}
    for _, row in df.iterrows():
        for idx, col in enumerate(turn_cols, start=1):
            if response_flipped(str(row.get(col, ""))):
                dist[idx] += 1
                break
    return {t: v / len(df) for t, v in dist.items()}


def judge_first_flip_dist(model, cond):
    csv_path, model_col = JUDGE_CSVS[model]
    df = pd.read_csv(csv_path)
    sub = df[(df["model"] == model_col) & (df["question_type"] == cond)].copy()
    sub["flip"] = sub["judgement"].astype(str).str.lower() == "true"
    total, dist = 0, {t: 0 for t in range(1, 6)}
    for q, g in sub.groupby("question"):
        total += 1
        fl = g[g["flip"]]["turn"]
        if len(fl):
            dist[int(fl.min())] += 1
    if total == 0:
        return {t: 0.0 for t in range(1, 6)}
    return {t: v / total for t, v in dist.items()}


def draw(ax, dist_fn, cond, title=None, ylabel=None):
    x = np.arange(len(MODELS))
    for i, model in enumerate(MODELS):
        dist = dist_fn(model, cond)
        bottom = 0.0
        for t in range(1, 6):
            h = dist[t] * 100
            ax.bar(i, h, bottom=bottom, width=0.65, color=TURN_COLORS[t - 1],
                   edgecolor="white", linewidth=0.5,
                   label=f"T{t}" if i == 0 else None)
            bottom += h
        ax.text(i, bottom + 1.5, f"{bottom:.0f}%", ha="center", va="bottom",
                fontsize=7.5, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([SHORT_NAMES[m] for m in MODELS], fontsize=8.5)
    if title:
        ax.set_title(title, fontsize=10, pad=6)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=9, fontweight="bold")
    ax.set_ylim(0, 110)
    ax.tick_params(labelsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.yaxis.grid(True, linestyle="--", alpha=0.3)
    ax.set_axisbelow(True)


def main():
    fig, axes = plt.subplots(2, 3, figsize=(9.5, 5.5), sharey=True)
    fig.subplots_adjust(wspace=0.08, hspace=0.30, left=0.07, right=0.99,
                        top=0.82, bottom=0.07)

    for col, (cond, cl) in enumerate(zip(CONDITIONS, COND_LABELS)):
        draw(axes[0][col], keyword_first_flip_dist, cond, title=cl,
             ylabel="Keyword\never-flip (%)" if col == 0 else None)
        draw(axes[1][col], judge_first_flip_dist, cond,
             ylabel="LLM-as-judge\never-flip (%)" if col == 0 else None)

    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, fontsize=8, ncol=5, loc="upper center",
               bbox_to_anchor=(0.5, 0.99), title="First-flip turn",
               title_fontsize=8, framealpha=0.9, columnspacing=1.0)

    plt.savefig(OUT_PNG, dpi=300, bbox_inches="tight")
    print(f"Saved -> {OUT_PNG}")


if __name__ == "__main__":
    main()
