#!/usr/bin/env python3
"""Bubble grid: Soham's LLM-as-judge labels vs Tomas's labels.

Only the 3 overlapping models are compared:
  Qwen3.5-9B (Soham) / Qwen3.5-9B-SYCO (Tomas)
  Gemma-2-9B / gemma-2-9b-it
  Qwen2.5-7B-Instruct (both)

Questions are restricted to the shared question set per model for fairness.
"""
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_PNG = REPO_ROOT / "analysis_claude" / "flip_turn_tomas_vs_soham.png"

QTYPES       = ["base", "critical", "presupposition"]
QTYPE_LABELS = ["Base", "Critical", "Presupposition"]

# ── Soham: LLM-as-judge CSVs ───────────────────────────────────────────────────
SOHAM_CSVS = {
    "Gemma-2-9B":  (REPO_ROOT / "analysis_claude" / "gemma_judgements_haiku.csv",   "Gemma-2-9B"),
    "Qwen2.5-7B":  (REPO_ROOT / "analysis_claude" / "qwen25_judgements_haiku.csv",  "Qwen2.5-7B-Instruct"),
    "Qwen3.5-9B":  (REPO_ROOT / "analysis_claude" / "qwen35_judgements_haiku.csv",  "Qwen3.5-9B"),
}

# ── Tomas: labels_all.csv ─────────────────────────────────────────────────────
TOMAS_CSV   = REPO_ROOT / "tomas" / "labels_all_new.csv"
TOMAS_NAMES = {
    "Gemma-2-9B":  "gemma-2-9b-it",
    "Qwen2.5-7B":  "Qwen2.5-7B-Instruct",
    "Qwen3.5-9B":  "Qwen3.5-9B-SYCO",
}

MODELS = ["Gemma-2-9B", "Qwen2.5-7B", "Qwen3.5-9B"]

MODEL_COLORS = {
    "Gemma-2-9B": "#D65F5F",
    "Qwen2.5-7B": "#B47CC7",
    "Qwen3.5-9B": "#C4AD66",
}

BUBBLE_SCALE = 2600
MIN_SHOW_PCT = 4

# ── Load data ──────────────────────────────────────────────────────────────────

def load_soham(model: str, qtype: str, common_q: set | None = None):
    csv_path, model_col = SOHAM_CSVS[model]
    df = pd.read_csv(csv_path)
    sub = df[(df["model"] == model_col) & (df["question_type"] == qtype)].copy()
    sub["flip"] = sub["judgement"].astype(str).str.lower() == "true"
    if common_q is not None:
        sub = sub[sub["question"].isin(common_q)]

    questions = sub["question"].unique()
    first_flips = []
    for q in questions:
        q_df = sub[sub["question"] == q].sort_values("turn")
        flipped = q_df[q_df["flip"]]["turn"]
        first_flips.append(int(flipped.min()) if len(flipped) > 0 else None)

    total = len(first_flips)
    if total == 0:
        return {t: 0.0 for t in range(1, 6)}, 0.0, 0
    dist = {t: sum(1 for v in first_flips if v == t) / total for t in range(1, 6)}
    ever = sum(1 for v in first_flips if v is not None) / total
    return dist, ever, total


_tomas_df = pd.read_csv(TOMAS_CSV)
_tomas_df["flip"] = _tomas_df["flip"].astype(str).str.lower() == "true"

def load_tomas(model: str, qtype: str, common_q: set | None = None):
    model_col = TOMAS_NAMES[model]
    sub = _tomas_df[(_tomas_df["model"] == model_col) &
                    (_tomas_df["question_type"] == qtype)].copy()
    if common_q is not None:
        sub = sub[sub["question"].isin(common_q)]

    total = len(sub)
    if total == 0:
        return {t: 0.0 for t in range(1, 6)}, 0.0, 0

    dist = {}
    for t in range(1, 6):
        dist[t] = (sub["flip_turn"] == t).sum() / total

    ever = sub["flip"].sum() / total
    return dist, ever, total


# Pre-compute common question sets per model/qtype
def common_questions(model: str, qtype: str) -> set:
    csv_path, model_col = SOHAM_CSVS[model]
    df_s = pd.read_csv(csv_path)
    soham_q = set(df_s[(df_s["model"] == model_col) &
                       (df_s["question_type"] == qtype)]["question"])
    model_col_t = TOMAS_NAMES[model]
    tomas_q = set(_tomas_df[(_tomas_df["model"] == model_col_t) &
                             (_tomas_df["question_type"] == qtype)]["question"])
    return soham_q & tomas_q


# ── Draw one panel ─────────────────────────────────────────────────────────────

def draw_panel(ax, load_fn, qtype, models, title=None, show_ylabels=False):
    common_qs = {m: common_questions(m, qtype) for m in models}

    for i, model in enumerate(models):
        color = MODEL_COLORS[model]
        dist, ever, total = load_fn(model, qtype, common_qs[model])

        for t in range(1, 6):
            rate = dist.get(t, 0)
            if rate > 0.005:
                ax.scatter(t, i, s=max(rate * BUBBLE_SCALE, 18),
                           color=color, alpha=0.82, zorder=3,
                           edgecolors="white", linewidths=0.9)
                if rate * 100 >= MIN_SHOW_PCT:
                    ax.text(t, i, f"{rate*100:.0f}%",
                            ha="center", va="center",
                            fontsize=7, fontweight="bold", color="white", zorder=4)

        ax.text(5.65, i, f"{ever*100:.0f}%",
                ha="left", va="center", fontsize=9,
                color=color, fontweight="bold")

        # n= annotation below model label
        ax.text(-0.1, i - 0.38, f"n={total}", ha="right", va="center",
                fontsize=6.5, color="#888888")

    ax.text(5.65, len(models) - 0.55, "ever\nflip",
            ha="left", va="bottom", fontsize=7, color="#666666", style="italic")

    ax.set_xlim(0.3, 6.3)
    ax.set_ylim(-0.75, len(models) - 0.25)
    ax.set_xticks(range(1, 6))
    ax.set_xticklabels([f"T{t}" for t in range(1, 6)], fontsize=8.5)
    if title:
        ax.set_title(title, fontsize=11, pad=7)
    if show_ylabels:
        ax.set_yticks(range(len(models)))
        ax.set_yticklabels(models, fontsize=9.5)
    else:
        ax.set_yticks(range(len(models)))
        ax.set_yticklabels([])
    ax.yaxis.grid(True, linestyle="--", alpha=0.35, zorder=0)
    ax.xaxis.grid(True, linestyle="--", alpha=0.35, zorder=0)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


# ── Figure ─────────────────────────────────────────────────────────────────────

fig, axes = plt.subplots(2, 3, figsize=(14, 7.5))
fig.subplots_adjust(wspace=0.06, hspace=0.42)

ROWS = [
    (load_soham, "Soham — LLM-as-Judge\n(Claude Haiku 4.5, first-flip)"),
    (load_tomas, "Tomas — LLM-as-Judge\n(holistic / final-position)"),
]

for row, (load_fn, row_label) in enumerate(ROWS):
    for col, (qt, qt_label) in enumerate(zip(QTYPES, QTYPE_LABELS)):
        ax = axes[row][col]
        title = qt_label if row == 0 else None
        draw_panel(ax, load_fn, qt, MODELS, title=title, show_ylabels=(col == 0))

        if col == 0:
            ax.set_ylabel(row_label, fontsize=10, fontweight="bold",
                          labelpad=12, rotation=90, va="center")

# ── Model-name note ───────────────────────────────────────────────────────────
fig.text(
    0.5, -0.02,
    "Tomas model names: Gemma-2-9B → gemma-2-9b-it  |  "
    "Qwen3.5-9B → Qwen3.5-9B-SYCO  |  Qwen2.5-7B → Qwen2.5-7B-Instruct\n"
    "Only shared questions used per model/qtype.  "
    "T1: \"Are you sure?\" → T5: \"I'm confident you're wrong\"",
    ha="center", fontsize=7.2, color="#555555",
)

# ── Bubble legend ─────────────────────────────────────────────────────────────
legend_ax = axes[1][2]
for pct, lbl in [(10, "10%"), (25, "25%"), (50, "50%")]:
    legend_ax.scatter([], [], s=(pct / 100) * BUBBLE_SCALE,
                      color="#888888", alpha=0.7, label=lbl,
                      edgecolors="white", linewidths=0.8)
legend_ax.legend(title="% first-flip\nat that turn",
                 title_fontsize=7.5, fontsize=8,
                 loc="lower right", framealpha=0.85,
                 handletextpad=0.4, labelspacing=0.5)

fig.suptitle(
    "Sycophantic Flip Rate: Soham's Judge Labels vs Tomas's Judge Labels\n"
    "(bubble area ∝ % of questions first-flipping at that pressure turn)",
    fontsize=12, y=1.02,
)

plt.savefig(OUT_PNG, dpi=180, bbox_inches="tight")
print(f"Saved → {OUT_PNG}")
