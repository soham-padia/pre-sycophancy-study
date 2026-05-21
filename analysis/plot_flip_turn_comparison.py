#!/usr/bin/env python3
"""Side-by-side bubble grid: keyword labels (top) vs LLM-as-judge labels (bottom).

Rows: labeling method (keyword / judge)
Columns: question type (base / critical / presupposition)
Y-axis: models
X-axis: pressure turn (T1–T5)
Bubble: % of questions first-flipping at that turn
Right annotation: ever-flip %
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

# ── Data sources ───────────────────────────────────────────────────────────────

MODELS = [
    "DeepSeek-R1-7B",
    "Gemma-2-9B",
    "Llama-3.1-8B",
    "Qwen2.5-7B",
    "Qwen3.5-9B",
]

# Multiturn CSVs for keyword labels
KEYWORD_DIRS = {
    "DeepSeek-R1-7B": REPO_ROOT / "data" / "DeepSeek-R1-Distill-Qwen-7B",
    "Gemma-2-9B":     REPO_ROOT / "data" / "Gemma-2-9B",
    "Llama-3.1-8B":   REPO_ROOT / "data" / "Llama-3.1-8B-Instruct",
    "Qwen2.5-7B":     REPO_ROOT / "data" / "Qwen2.5-7B-Instruct",
    "Qwen3.5-9B":     REPO_ROOT / "data" / "Qwen3.5-9B",
}

# Judge CSVs for LLM-as-judge labels
JUDGE_CSVS = {
    "DeepSeek-R1-7B": (REPO_ROOT / "analysis_claude" / "claude_judgements.csv",        "DeepSeek-R1-Distill-Qwen-7B"),
    "Gemma-2-9B":     (REPO_ROOT / "analysis_claude" / "gemma_judgements_haiku.csv",   "Gemma-2-9B"),
    "Llama-3.1-8B":   (REPO_ROOT / "analysis_claude" / "llama31_judgements_haiku.csv", "Llama-3.1-8B-Instruct"),
    "Qwen2.5-7B":     (REPO_ROOT / "analysis_claude" / "qwen25_judgements_haiku.csv",  "Qwen2.5-7B-Instruct"),
    "Qwen3.5-9B":     (REPO_ROOT / "analysis_claude" / "qwen35_judgements_haiku.csv",  "Qwen3.5-9B"),
}

QTYPES       = ["base", "critical", "presupposition"]
QTYPE_LABELS = ["Base", "Critical", "Presupposition"]

MODEL_COLORS = {
    "DeepSeek-R1-7B": "#4878CF",
    "Gemma-2-9B":     "#D65F5F",
    "Llama-3.1-8B":   "#6ACC65",
    "Qwen2.5-7B":     "#B47CC7",
    "Qwen3.5-9B":     "#C4AD66",
}

BUBBLE_SCALE = 2600
MIN_SHOW_PCT = 4


# ── Keyword: first-flip distribution from multiturn CSVs ──────────────────────

def keyword_first_flip_dist(model: str, qtype: str):
    csv_path = KEYWORD_DIRS[model] / f"{qtype}_multiturn.csv"
    if not csv_path.exists():
        return {t: 0.0 for t in range(1, 6)}, 0.0

    df = pd.read_csv(csv_path)
    turn_cols = [f"Response_Turn_{t}" for t in range(1, 6)]
    # Keep only columns that exist
    turn_cols = [c for c in turn_cols if c in df.columns]

    first_flips = []
    for _, row in df.iterrows():
        first = None
        for idx, col in enumerate(turn_cols, start=1):
            text = str(row.get(col, ""))
            if response_flipped(text):
                first = idx
                break
        first_flips.append(first)

    total = len(first_flips)
    dist = {t: sum(1 for v in first_flips if v == t) / total for t in range(1, 6)}
    ever = sum(1 for v in first_flips if v is not None) / total
    return dist, ever


# ── Judge: first-flip distribution from judge CSVs ────────────────────────────

def judge_first_flip_dist(model: str, qtype: str):
    csv_path, model_col = JUDGE_CSVS[model]
    if not csv_path.exists():
        return {t: 0.0 for t in range(1, 6)}, 0.0

    df = pd.read_csv(csv_path)
    sub = df[(df["model"] == model_col) & (df["question_type"] == qtype)].copy()
    sub["flip"] = sub["judgement"].astype(str).str.lower() == "true"

    questions = sub["question"].unique()
    first_flips = []
    for q in questions:
        q_df = sub[sub["question"] == q].sort_values("turn")
        flipped = q_df[q_df["flip"]]["turn"]
        first_flips.append(int(flipped.min()) if len(flipped) > 0 else None)

    total = len(first_flips)
    if total == 0:
        return {t: 0.0 for t in range(1, 6)}, 0.0
    dist = {t: sum(1 for v in first_flips if v == t) / total for t in range(1, 6)}
    ever = sum(1 for v in first_flips if v is not None) / total
    return dist, ever


# ── Draw one panel ─────────────────────────────────────────────────────────────

def draw_panel(ax, dist_fn, qtype, models, title=None):
    for i, model in enumerate(models):
        color = MODEL_COLORS[model]
        dist, ever = dist_fn(model, qtype)

        for t in range(1, 6):
            rate = dist.get(t, 0)
            if rate > 0.005:
                ax.scatter(t, i, s=max(rate * BUBBLE_SCALE, 18),
                           color=color, alpha=0.82, zorder=3,
                           edgecolors="white", linewidths=0.9)
                if rate * 100 >= MIN_SHOW_PCT:
                    ax.text(t, i, f"{rate*100:.0f}%",
                            ha="center", va="center",
                            fontsize=8.5, fontweight="bold", color="white", zorder=4)

    ax.set_xlim(0.4, 5.6)
    ax.set_ylim(-0.7, len(models) - 0.3)
    ax.set_xticks(range(1, 6))
    ax.set_xticklabels([f"T{t}" for t in range(1, 6)], fontsize=11)
    if title:
        ax.set_title(title, fontsize=13, pad=7)
    ax.yaxis.grid(True, linestyle="--", alpha=0.35, zorder=0)
    ax.xaxis.grid(True, linestyle="--", alpha=0.35, zorder=0)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


# ── Figure: 2 rows × 3 columns ────────────────────────────────────────────────

fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.subplots_adjust(wspace=0.06, hspace=0.38)

ROW_LABELS = ["Keyword\nLabels", "LLM-as-Judge\n(Claude Haiku 4.5)"]
dist_fns   = [keyword_first_flip_dist, judge_first_flip_dist]

for row, (dist_fn, row_label) in enumerate(zip(dist_fns, ROW_LABELS)):
    for col, (qt, qt_label) in enumerate(zip(QTYPES, QTYPE_LABELS)):
        ax = axes[row][col]
        title = qt_label if row == 0 else None
        draw_panel(ax, dist_fn, qt, MODELS, title=title)

        ax.set_yticks(range(len(MODELS)))
        if col == 0:
            ticklabels = ax.set_yticklabels(MODELS, fontsize=11)
            for label, model in zip(ticklabels, MODELS):
                label.set_color(MODEL_COLORS[model])
                label.set_fontweight("bold")
            ax.set_ylabel(row_label, fontsize=12, fontweight="bold",
                          labelpad=10, rotation=90, va="center")
        else:
            ax.tick_params(labelleft=False)

# ── Right-margin ever-flip annotations on rightmost column ───────────────────
for row, dist_fn in enumerate(dist_fns):
    ax = axes[row][2]
    for i, model in enumerate(MODELS):
        ever_vals = [dist_fn(model, qt)[1] for qt in QTYPES]
        avg_ever = sum(ever_vals) / len(ever_vals)
        ax.annotate(
            f"{avg_ever*100:.0f}%",
            xy=(5.6, i), xycoords="data",
            xytext=(6, 0), textcoords="offset points",
            ha="left", va="center",
            fontsize=10.5, fontweight="bold",
            color=MODEL_COLORS[model],
            annotation_clip=False,
        )

# ── Pressure-level note ───────────────────────────────────────────────────────
fig.text(
    0.5, -0.005,
    "T1: \"Are you sure?\"  •  T2: \"I think you're mistaken\"  •  T3: \"Pretty sure that's wrong\"",
    ha="center", fontsize=10.5, color="#555555",
)
fig.text(
    0.5, -0.03,
    "T4: \"I checked a textbook\"  •  T5: \"I'm confident you're wrong\"",
    ha="center", fontsize=10.5, color="#555555",
)

fig.suptitle(
    "First-Flip Turn Distribution: Keyword Labels vs LLM-as-Judge\n"
    "(bubble area ∝ % of questions first-flipping at that pressure turn)",
    fontsize=13, y=1.01,
)

plt.savefig(OUT_PNG, dpi=180, bbox_inches="tight")
print(f"Saved → {OUT_PNG}")
