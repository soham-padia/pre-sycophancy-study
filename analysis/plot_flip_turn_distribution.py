#!/usr/bin/env python3
"""Stacked-bar first-flip turn distribution (LLM-as-judge labels, main figure).

One panel per prompt condition; x-axis = models; bar height = ever-flip rate,
segmented by the pressure turn (T1..T5) at which questions first flip.
"""
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_PNG = REPO_ROOT / "analysis_claude" / "flip_turn_distribution.png"

JUDGE_CSVS = {
    "DeepSeek-R1-7B":  (REPO_ROOT / "analysis_claude" / "claude_judgements.csv",        "DeepSeek-R1-Distill-Qwen-7B"),
    "Gemma-2-9B":      (REPO_ROOT / "analysis_claude" / "gemma_judgements_haiku.csv",   "Gemma-2-9B"),
    "Llama-3.1-8B":    (REPO_ROOT / "analysis_claude" / "llama31_judgements_haiku.csv", "Llama-3.1-8B-Instruct"),
    "Qwen2.5-7B":      (REPO_ROOT / "analysis_claude" / "qwen25_judgements_haiku.csv",  "Qwen2.5-7B-Instruct"),
    "Qwen3.5-9B":      (REPO_ROOT / "analysis_claude" / "qwen35_judgements_haiku.csv",  "Qwen3.5-9B"),
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

# Sequential turn colors: mild doubt (light) -> strong contradiction (dark)
TURN_COLORS = ["#c6dbef", "#9ecae1", "#6baed6", "#3182bd", "#08519c"]


def first_flip_dist(df, model_col, cond):
    """{turn: fraction of ALL questions first-flipping at that turn}."""
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


def draw(ax, dists, models, title, show_ylabel):
    x = np.arange(len(models))
    for i, model in enumerate(models):
        bottom = 0.0
        for t in range(1, 6):
            h = dists[model][t] * 100
            ax.bar(i, h, bottom=bottom, width=0.65, color=TURN_COLORS[t - 1],
                   edgecolor="white", linewidth=0.5,
                   label=f"T{t}" if i == 0 else None)
            bottom += h
        ax.text(i, bottom + 1.5, f"{bottom:.0f}%", ha="center", va="bottom",
                fontsize=8, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([SHORT_NAMES[m] for m in models], fontsize=9)
    ax.set_title(title, fontsize=10, pad=6)
    ax.set_ylim(0, 108)
    if show_ylabel:
        ax.set_ylabel("Ever-flip rate (%)", fontsize=9)
    ax.tick_params(labelsize=8.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.yaxis.grid(True, linestyle="--", alpha=0.3)
    ax.set_axisbelow(True)


def main():
    models = list(JUDGE_CSVS.keys())
    dfs = {m: pd.read_csv(p) for m, (p, _) in JUDGE_CSVS.items()}

    fig, axes = plt.subplots(1, 3, figsize=(9.5, 3.1), sharey=True)
    fig.subplots_adjust(wspace=0.08, left=0.06, right=0.99, top=0.74, bottom=0.12)

    for ax, cond, cl in zip(axes, CONDITIONS, COND_LABELS):
        dists = {m: first_flip_dist(dfs[m], JUDGE_CSVS[m][1], cond) for m in models}
        draw(ax, dists, models, cl, show_ylabel=(cond == "base"))

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, fontsize=8, ncol=5, loc="upper center",
               bbox_to_anchor=(0.5, 1.02), title="First-flip turn",
               title_fontsize=8, framealpha=0.9, columnspacing=1.0)

    plt.savefig(OUT_PNG, dpi=300, bbox_inches="tight")
    print(f"Saved -> {OUT_PNG}")


if __name__ == "__main__":
    main()
