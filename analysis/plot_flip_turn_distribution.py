#!/usr/bin/env python3
"""Bubble grid: first-flip turn distribution by model and question type."""
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

QTYPES       = ["base", "critical", "presupposition"]
QTYPE_LABELS = ["Base", "Critical", "Presupposition"]

# Short pressure-level labels for x-axis
PRESSURE = [
    '"Are you sure?"',
    '"You might be mistaken"',
    '"Pretty sure that\'s wrong"',
    '"I checked a textbook"',
    '"I\'m confident you\'re wrong"',
]

MODEL_COLORS = {
    "DeepSeek-R1-7B": "#4878CF",
    "Gemma-2-9B":     "#D65F5F",
    "Llama-3.1-8B":   "#6ACC65",
    "Qwen2.5-7B":     "#B47CC7",
    "Qwen3.5-9B":     "#C4AD66",
}


def first_flip_dist(df, model_col, qtype):
    """Returns (turn→rate, ever_flip_rate) where rate = fraction of all questions."""
    sub = df[(df["model"] == model_col) & (df["question_type"] == qtype)].copy()
    sub["flip"] = sub["judgement"].astype(str).str.lower() == "true"
    questions = sub["question"].unique()
    if len(questions) == 0:
        return {}, 0.0

    first_flips = {}
    for q in questions:
        q_df = sub[sub["question"] == q].sort_values("turn")
        flipped = q_df[q_df["flip"]]["turn"]
        first_flips[q] = int(flipped.min()) if len(flipped) > 0 else None

    total = len(first_flips)
    dist = {}
    for t in range(1, 6):
        dist[t] = sum(1 for v in first_flips.values() if v == t) / total
    ever = sum(1 for v in first_flips.values() if v is not None) / total
    return dist, ever


# ── Load all data ──────────────────────────────────────────────────────────────
models = list(JUDGE_CSVS.keys())
dfs = {m: pd.read_csv(p) for m, (p, _) in JUDGE_CSVS.items()}

# ── Figure ─────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(6.5, 3.0), sharey=True)
fig.subplots_adjust(wspace=0.06, right=0.88, bottom=0.18)

BUBBLE_SCALE = 2800   # area multiplier: rate=1.0 → s=2800 pts²
MIN_SHOW_PCT = 4      # show label only for bubbles ≥ 4%

for ax, qt, qt_label in zip(axes, QTYPES, QTYPE_LABELS):
    for i, model in enumerate(models):
        _, model_col = JUDGE_CSVS[model]
        color = MODEL_COLORS[model]
        dist, ever = first_flip_dist(dfs[model], model_col, qt)

        for t in range(1, 6):
            rate = dist.get(t, 0)
            if rate > 0.005:
                ax.scatter(t, i, s=max(rate * BUBBLE_SCALE, 20),
                           color=color, alpha=0.82, zorder=3,
                           edgecolors="white", linewidths=0.9)
                if rate * 100 >= MIN_SHOW_PCT:
                    ax.text(t, i, f"{rate*100:.0f}%",
                            ha="center", va="center",
                            fontsize=7, fontweight="bold", color="white", zorder=4)

    ax.set_xlim(0.4, 5.6)
    ax.set_ylim(-0.7, len(models) - 0.3)
    ax.set_xticks(range(1, 6))
    ax.set_xticklabels([f"T{t}" for t in range(1, 6)], fontsize=9)
    ax.set_title(qt_label, fontsize=10, pad=6)
    ax.yaxis.grid(True, linestyle="--", alpha=0.35, zorder=0)
    ax.xaxis.grid(True, linestyle="--", alpha=0.35, zorder=0)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

# Model names on leftmost panel, color-coded to match their bubbles
axes[0].set_yticks(range(len(models)))
axes[0].set_yticklabels(models, fontsize=9)
for tick, model in zip(axes[0].get_yticklabels(), models):
    tick.set_color(MODEL_COLORS[model])
    tick.set_fontweight("bold")

# Right-margin ever-flip rate annotations on the rightmost panel
for i, model in enumerate(models):
    _, model_col = JUDGE_CSVS[model]
    ever_vals = [first_flip_dist(dfs[model], model_col, qt)[1] for qt in QTYPES]
    avg_ever = np.mean(ever_vals)
    axes[-1].annotate(
        f"{avg_ever*100:.0f}%",
        xy=(5.6, i), xycoords="data",
        xytext=(6, 0), textcoords="offset points",
        ha="left", va="center",
        fontsize=9, fontweight="bold",
        color=MODEL_COLORS[model],
        annotation_clip=False,
    )

# ── Pressure-level note at bottom ──────────────────────────────────────────────
line1 = "  •  ".join([f"T{i+1}: {p}" for i, p in enumerate(PRESSURE[:3])])
line2 = "  •  ".join([f"T{i+4}: {p}" for i, p in enumerate(PRESSURE[3:])])
fig.text(0.5, 0.07, line1, ha="center", fontsize=8.5, color="#555555")
fig.text(0.5, 0.02, line2, ha="center", fontsize=8.5, color="#555555")

fig.suptitle(
    "First-Flip Turn Distribution by Model and Question Type\n"
    "(LLM-as-judge: Claude Haiku 4.5  ·  bubble area ∝ % of questions first-flipping at that pressure turn)",
    fontsize=10, y=1.03,
)

plt.savefig(OUT_PNG, dpi=300, bbox_inches="tight")
print(f"Saved → {OUT_PNG}")
