#!/usr/bin/env python3
"""Teaser figure: 3-column layout with per-turn PCA thumbnails showing
pre-capitulation hidden-state drift.

Columns: Conversation | Hidden-State Space (4 PCA plots) | Behavior
Model:   Llama-3.1-8B, critical questions, layer 9 (best pre-flip layer)
"""
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch
from matplotlib.lines import Line2D
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

REPO_ROOT  = Path(__file__).resolve().parents[1]
OUT_PNG    = REPO_ROOT / "analysis_claude" / "teaser_pca_drift.png"
MODEL_DIR  = REPO_ROOT / "data" / "Llama-3.1-8B-Instruct"
JUDGE_CSV  = REPO_ROOT / "analysis_claude" / "llama31_judgements_haiku.csv"
QTYPE      = "critical"
MODEL_COL  = "Llama-3.1-8B-Instruct"
LAYER      = 9      # best pre-flip layer for Llama (highest F1 in sweep)

# ── Load hidden states ─────────────────────────────────────────────────────
print("Loading hidden states...")
hs = torch.load(MODEL_DIR / f"{QTYPE}_multiturn_hidden_states.pt", map_location="cpu")
questions = list(hs.keys())
print(f"  {len(questions)} questions loaded")

# ── Load judge labels → first-flip turn per question ──────────────────────
df = pd.read_csv(JUDGE_CSV)
df = df[(df["model"] == MODEL_COL) & (df["question_type"] == QTYPE)].copy()
df["flip"] = df["judgement"].astype(str).str.lower() == "true"

first_flip = {}
for q in questions:
    q_df = df[df["question"] == q].sort_values("turn")
    flipped_turns = q_df[q_df["flip"]]["turn"]
    first_flip[q] = int(flipped_turns.min()) if len(flipped_turns) > 0 else None


# ── Extract layer vector ───────────────────────────────────────────────────
def get_vec(q, turn):
    try:
        t = hs[q][turn]
        if t is None:
            return None
        return t.float()[LAYER].numpy()
    except Exception:
        return None


# ── Build valid question set (T0-T3 all present, in judge CSV) ────────────
valid_qs = []
for q in questions:
    if q not in first_flip:
        continue
    if all(get_vec(q, t) is not None for t in range(4)):
        valid_qs.append(q)
print(f"  {len(valid_qs)} questions with T0–T3 and judge labels")

# ── Fit PCA on all T0 vectors ─────────────────────────────────────────────
print("Fitting PCA on T0 vectors...")
X_t0 = np.array([get_vec(q, 0) for q in valid_qs])
scaler = StandardScaler()
X_t0_sc = scaler.fit_transform(X_t0)
pca = PCA(n_components=2, random_state=42)
pca.fit(X_t0_sc)

def project(q, turn):
    v = get_vec(q, turn)
    if v is None:
        return None
    return pca.transform(scaler.transform(v.reshape(1, -1)))[0]

# Project all valid questions at turns 0-3
turn_pts = {}
for t in range(4):
    turn_pts[t] = np.array([project(q, t) for q in valid_qs])  # (N, 2)

eventual_flip = np.array([first_flip[q] is not None for q in valid_qs])

# ── Pick example: flips at T3, maximum PC1 shift T0→T3 ───────────────────
candidates = [i for i, q in enumerate(valid_qs) if first_flip[q] == 3]
print(f"  {len(candidates)} questions first-flip at T3")
ex_idx = max(candidates,
             key=lambda i: abs(turn_pts[3][i, 0] - turn_pts[0][i, 0]))
ex_q = valid_qs[ex_idx]
print(f"  Example question (idx={ex_idx}): {ex_q[:80]!r}")

# ── Compute common axis limits (3–97th percentile across all turns) ────────
all_x = np.concatenate([turn_pts[t][:, 0] for t in range(4)])
all_y = np.concatenate([turn_pts[t][:, 1] for t in range(4)])
xlo, xhi = np.percentile(all_x, 3), np.percentile(all_x, 97)
ylo, yhi = np.percentile(all_y, 3), np.percentile(all_y, 97)
xpad = 0.18 * (xhi - xlo)
ypad = 0.18 * (yhi - ylo)
XLIM = (xlo - xpad, xhi + xpad)
YLIM = (ylo - ypad, yhi + ypad)

# ── Style constants ────────────────────────────────────────────────────────
ROW_FC   = ["#dceeff", "#dceeff", "#fff3cd", "#ffd6d6"]   # bg per turn
ROW_EC   = ["#1a3a6b", "#1a3a6b", "#8a6200", "#8b0000"]   # edge/text per turn
HOLD_C   = "#90b8f8"   # eventually-hold background dots
FLIP_C   = "#f4a0a0"   # eventually-flip background dots
EX_C     = ["#2255cc", "#2255cc", "#d46800", "#b00000"]   # example per turn
TRAJ_C   = "#333333"

CONV_LABELS = [
    "T0\nBaseline answer\ngiven",
    "T1\n\"Are you sure?\"",
    "T2\n\"I don't think\nthat's right.\"",
    "T3\nCapitulation",
]
BEH_LABELS = [
    "Stable\nOutput",
    "Stable\nOutput",
    "Hesitation\nVisible",
    "Behavioral\nFlip",
]

# ── Build figure ───────────────────────────────────────────────────────────
fig = plt.figure(figsize=(12, 10))
outer = gridspec.GridSpec(
    1, 3, figure=fig,
    left=0.04, right=0.96, top=0.87, bottom=0.07,
    wspace=0.10,
    width_ratios=[1, 2.4, 1],
)
left_gs  = gridspec.GridSpecFromSubplotSpec(4, 1, subplot_spec=outer[0],  hspace=0.22)
mid_gs   = gridspec.GridSpecFromSubplotSpec(4, 1, subplot_spec=outer[1],  hspace=0.22)
right_gs = gridspec.GridSpecFromSubplotSpec(4, 1, subplot_spec=outer[2],  hspace=0.22)


def draw_textbox(fig, gs_spec, text, fc, ec, fontsize=10):
    ax = fig.add_subplot(gs_spec)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    ax.add_patch(FancyBboxPatch(
        (0.06, 0.06), 0.88, 0.88,
        boxstyle="round,pad=0.05", linewidth=1.8,
        facecolor=fc, edgecolor=ec, zorder=1,
    ))
    ax.text(0.5, 0.5, text, ha="center", va="center",
            fontsize=fontsize, fontweight="bold", color=ec,
            multialignment="center", transform=ax.transAxes, zorder=2)
    return ax


# Draw left & right text boxes
for row in range(4):
    draw_textbox(fig, left_gs[row],  CONV_LABELS[row], ROW_FC[row], ROW_EC[row])
    draw_textbox(fig, right_gs[row], BEH_LABELS[row],  ROW_FC[row], ROW_EC[row])

# Draw PCA thumbnails
for row in range(4):
    ax = fig.add_subplot(mid_gs[row])

    xs = turn_pts[row][:, 0]
    ys = turn_pts[row][:, 1]

    # Background population
    ax.scatter(xs[~eventual_flip], ys[~eventual_flip],
               s=10, color=HOLD_C, alpha=0.50, linewidths=0, zorder=2)
    ax.scatter(xs[eventual_flip],  ys[eventual_flip],
               s=10, color=FLIP_C, alpha=0.50, linewidths=0, zorder=2)

    # Trajectory arrows T0 → current turn
    ex_trail = [turn_pts[t][ex_idx] for t in range(row + 1)]
    for i in range(len(ex_trail) - 1):
        ax.annotate(
            "", xy=ex_trail[i + 1], xytext=ex_trail[i],
            arrowprops=dict(
                arrowstyle="->,head_width=0.20,head_length=0.20",
                color=TRAJ_C, lw=1.6,
            ),
            zorder=5,
        )
    # Trajectory dots
    for t, pt in enumerate(ex_trail):
        is_current = (t == row)
        ax.scatter(
            pt[0], pt[1],
            s=120 if is_current else 50,
            color=EX_C[t],
            alpha=1.0 if is_current else 0.40,
            edgecolors="white" if is_current else "none",
            linewidths=1.5 if is_current else 0,
            zorder=6,
        )

    ax.set_xlim(*XLIM); ax.set_ylim(*YLIM)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlabel("PC 1", fontsize=7, color="#555", labelpad=2)
    ax.set_ylabel("PC 2", fontsize=7, color="#555", labelpad=2)
    for sp in ax.spines.values():
        sp.set_linewidth(1.4)
        sp.set_edgecolor(ROW_EC[row])

    # Turn tag inside the plot
    ax.text(0.03, 0.95, f"T{row}", transform=ax.transAxes,
            fontsize=9, fontweight="bold", color=ROW_EC[row],
            va="top", ha="left")

# ── Column headers ─────────────────────────────────────────────────────────
# Get the bounding boxes after layout is set
fig.canvas.draw()

def col_center(gs_spec):
    """Return figure-fraction x-center of a GridSpec column."""
    bbox = gs_spec.get_position(fig)
    return (bbox.x0 + bbox.x1) / 2

hdr_y = 0.895
fig.text(col_center(outer[0]), hdr_y, "Conversation",
         ha="center", fontsize=11, fontweight="bold", color="#222")
fig.text(col_center(outer[1]), hdr_y, "Hidden-State Space  (PCA)",
         ha="center", fontsize=11, fontweight="bold", color="#222")
fig.text(col_center(outer[2]), hdr_y, "Behavior",
         ha="center", fontsize=11, fontweight="bold", color="#222")

# ── Legend ─────────────────────────────────────────────────────────────────
legend_handles = [
    Line2D([0], [0], marker="o", color="w", markerfacecolor=HOLD_C,
           markersize=9, label="Population: Eventually Holds"),
    Line2D([0], [0], marker="o", color="w", markerfacecolor=FLIP_C,
           markersize=9, label="Population: Eventually Flips"),
    Line2D([0], [0], marker="o", color="w", markerfacecolor=EX_C[3],
           markersize=9, label="Example question trajectory  (flips at T3)"),
]
fig.legend(handles=legend_handles, loc="lower center", ncol=3,
           fontsize=8.5, frameon=True, framealpha=0.9,
           bbox_to_anchor=(0.5, 0.005), edgecolor="#aaa")

fig.suptitle(
    "Pre-Capitulation Drift in Hidden-State Space\n"
    r"$\it{Llama\text{-}3.1\text{-}8B}$"
    "  ·  critical questions  ·  layer 9  ·  LLM-as-judge labels",
    fontsize=11, y=0.99, color="#111",
)

plt.savefig(OUT_PNG, dpi=180, bbox_inches="tight")
print(f"Saved → {OUT_PNG}")
