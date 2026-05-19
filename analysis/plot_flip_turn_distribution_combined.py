#!/usr/bin/env python3
"""Combined bubble grid: Soham Haiku (5 models) + Tomas Opus 4.7 (3 models) + Tomas manual (3 models)."""
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_PNG = REPO_ROOT / "analysis_claude" / "flip_turn_distribution_combined.png"

QTYPES       = ["base", "critical", "presupposition"]
QTYPE_LABELS = ["Base", "Critical", "Presupposition"]

# ── Soham: Haiku 4.5 per-turn CSVs ────────────────────────────────────────────
SOHAM_MODELS = ["DeepSeek-R1-7B", "Gemma-2-9B", "Llama-3.1-8B", "Qwen2.5-7B", "Qwen3.5-9B"]
SOHAM_CSVS = {
    "DeepSeek-R1-7B": (REPO_ROOT / "analysis_claude" / "claude_judgements.csv",        "DeepSeek-R1-Distill-Qwen-7B"),
    "Gemma-2-9B":     (REPO_ROOT / "analysis_claude" / "gemma_judgements_haiku.csv",   "Gemma-2-9B"),
    "Llama-3.1-8B":   (REPO_ROOT / "analysis_claude" / "llama31_judgements_haiku.csv", "Llama-3.1-8B-Instruct"),
    "Qwen2.5-7B":     (REPO_ROOT / "analysis_claude" / "qwen25_judgements_haiku.csv",  "Qwen2.5-7B-Instruct"),
    "Qwen3.5-9B":     (REPO_ROOT / "analysis_claude" / "qwen35_judgements_haiku.csv",  "Qwen3.5-9B"),
}

# ── Tomas: Opus 4.7 per-turn CSV ──────────────────────────────────────────────
OPUS_CSV = REPO_ROOT / "tomas" / "claude_opus47_judgements.csv"
OPUS_MODEL_MAP = {           # display name → CSV model column value
    "Gemma-2-9B":  "gemma-2-9b-it",
    "Qwen2.5-7B":  "Qwen2.5-7B-Instruct",
    "Qwen3.5-9B":  "Qwen3.5-9B-SYCO",
}

# ── Tomas: manual holistic labels ─────────────────────────────────────────────
MANUAL_CSV = REPO_ROOT / "tomas" / "labels_all_new.csv"
MANUAL_MODEL_MAP = {
    "Gemma-2-9B":  "gemma-2-9b-it",
    "Qwen2.5-7B":  "Qwen2.5-7B-Instruct",
    "Qwen3.5-9B":  "Qwen3.5-9B-SYCO",
}

OVERLAP_MODELS = ["Gemma-2-9B", "Qwen2.5-7B", "Qwen3.5-9B"]

MODEL_COLORS = {
    "DeepSeek-R1-7B": "#4878CF",
    "Gemma-2-9B":     "#D65F5F",
    "Llama-3.1-8B":   "#6ACC65",
    "Qwen2.5-7B":     "#B47CC7",
    "Qwen3.5-9B":     "#C4AD66",
}

BUBBLE_SCALE = 2600
MIN_SHOW_PCT = 4

# ── Pre-load CSVs ──────────────────────────────────────────────────────────────
_opus_df   = pd.read_csv(OPUS_CSV);   _opus_df["judgement"]   = _opus_df["judgement"].astype(str).str.lower() == "true"
_manual_df = pd.read_csv(MANUAL_CSV); _manual_df["flip"]      = _manual_df["flip"].astype(str).str.lower() == "true"


# ── Dist helpers ──────────────────────────────────────────────────────────────

def _first_flip_dist_from_perturn(df, model_col, qtype, judge_col="judgement"):
    sub = df[(df["model"] == model_col) & (df["question_type"] == qtype)].copy()
    questions = sub["question"].unique()
    first_flips = []
    for q in questions:
        q_df = sub[sub["question"] == q].sort_values("turn")
        flipped = q_df[q_df[judge_col]]["turn"]
        first_flips.append(int(flipped.min()) if len(flipped) > 0 else None)
    total = len(first_flips)
    if not total:
        return {t: 0.0 for t in range(1, 6)}, 0.0
    dist = {t: sum(1 for v in first_flips if v == t) / total for t in range(1, 6)}
    ever = sum(1 for v in first_flips if v is not None) / total
    return dist, ever


def soham_dist(model, qtype):
    csv_path, model_col = SOHAM_CSVS[model]
    df = pd.read_csv(csv_path)
    df["judgement"] = df["judgement"].astype(str).str.lower() == "true"
    return _first_flip_dist_from_perturn(df, model_col, qtype)


def opus_dist(model, qtype):
    model_col = OPUS_MODEL_MAP[model]
    return _first_flip_dist_from_perturn(_opus_df, model_col, qtype)


def manual_dist(model, qtype):
    model_col = MANUAL_MODEL_MAP[model]
    sub = _manual_df[(_manual_df["model"] == model_col) &
                     (_manual_df["question_type"] == qtype)].copy()
    total = len(sub)
    if not total:
        return {t: 0.0 for t in range(1, 6)}, 0.0
    dist = {t: (sub["flip_turn"] == t).sum() / total for t in range(1, 6)}
    ever = sub["flip"].sum() / total
    return dist, ever


# ── Draw one panel ─────────────────────────────────────────────────────────────

def draw_panel(ax, models, dist_fn, qtype, title=None, show_ylabels=False):
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
                            fontsize=6.5, fontweight="bold", color="white", zorder=4)

        ax.text(5.65, i, f"{ever*100:.0f}%",
                ha="left", va="center", fontsize=8.5,
                color=color, fontweight="bold")

    ax.text(5.65, len(models) - 0.55, "ever\nflip",
            ha="left", va="bottom", fontsize=7, color="#666666", style="italic")

    ax.set_xlim(0.4, 6.3)
    ax.set_ylim(-0.7, len(models) - 0.3)
    ax.set_xticks(range(1, 6))
    ax.set_xticklabels([f"T{t}" for t in range(1, 6)], fontsize=8.5)
    if title:
        ax.set_title(title, fontsize=11, pad=7)

    ax.set_yticks(range(len(models)))
    if show_ylabels:
        # Color each y-tick label to match its model
        ax.set_yticklabels(models, fontsize=8.5)
        for tick, model in zip(ax.get_yticklabels(), models):
            tick.set_color(MODEL_COLORS[model])
            tick.set_fontweight("bold")
    else:
        ax.set_yticklabels([])

    ax.yaxis.grid(True, linestyle="--", alpha=0.35, zorder=0)
    ax.xaxis.grid(True, linestyle="--", alpha=0.35, zorder=0)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


# ── Figure: 3 rows × 3 columns ────────────────────────────────────────────────

n_s = len(SOHAM_MODELS)
n_o = len(OVERLAP_MODELS)

fig = plt.figure(figsize=(15, 12))
gs = gridspec.GridSpec(
    3, 3,
    height_ratios=[n_s, n_o, n_o],
    hspace=0.48, wspace=0.06,
)

ROW_CONFIGS = [
    (SOHAM_MODELS,   soham_dist,  "Soham — LLM-as-Judge\n(Claude Haiku 4.5, first-flip)"),
    (OVERLAP_MODELS, opus_dist,   "Tomas — LLM-as-Judge\n(Claude Opus 4.7, first-flip)"),
    (OVERLAP_MODELS, manual_dist, "Tomas — Manual Labels\n(holistic / sustained-position)"),
]

axes_grid = []
for row, (models, dist_fn, row_label) in enumerate(ROW_CONFIGS):
    row_axes = []
    for col, (qt, qt_label) in enumerate(zip(QTYPES, QTYPE_LABELS)):
        ax = fig.add_subplot(gs[row, col])
        row_axes.append(ax)
        draw_panel(ax, models, dist_fn, qt,
                   title=qt_label if row == 0 else None,
                   show_ylabels=(col == 0))
        if col == 0:
            ax.set_ylabel(row_label, fontsize=9.5, fontweight="bold", labelpad=10)
    axes_grid.append(row_axes)

# Bubble legend — bottom-right panel
legend_ax = axes_grid[2][2]
for pct, lbl in [(10, "10%"), (25, "25%"), (50, "50%")]:
    legend_ax.scatter([], [], s=(pct / 100) * BUBBLE_SCALE,
                      color="#888888", alpha=0.7, label=lbl,
                      edgecolors="white", linewidths=0.8)
legend_ax.legend(title="% first-flip\nat that turn",
                 title_fontsize=7.5, fontsize=8,
                 loc="lower right", framealpha=0.85,
                 handletextpad=0.4, labelspacing=0.5)

fig.text(
    0.5, -0.01,
    "T1: \"Are you sure?\"  •  T2: \"I think you're mistaken\"  •  "
    "T3: \"Pretty sure that's wrong\"  •  T4: \"I checked a textbook\"  •  "
    "T5: \"I'm confident you're wrong\"",
    ha="center", fontsize=7.2, color="#555555",
)

fig.suptitle(
    "First-Flip Turn Distribution by Model and Question Type\n"
    "(bubble area ∝ % of questions first-flipping at that pressure turn)",
    fontsize=12, y=1.01,
)

plt.savefig(OUT_PNG, dpi=180, bbox_inches="tight")
print(f"Saved → {OUT_PNG}")
