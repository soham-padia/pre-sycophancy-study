#!/usr/bin/env python3
"""Teaser figure — all 4 models with hidden states.

Layout:
  Left column:   4 conversation text boxes  (T0–T3)
  Middle:        4-turn × 4-model grid of PCA thumbnails
  Right column:  4 behavior text boxes

Models: DeepSeek-R1-7B · Llama-3.1-8B · Qwen2.5-7B · Qwen3.5-9B
All use critical questions and each model's best pre-flip layer.
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

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_PNG   = REPO_ROOT / "analysis_claude" / "teaser_all_models.png"

MODELS = {
    "DeepSeek-R1-7B": {
        "dir":       REPO_ROOT / "data" / "DeepSeek-R1-Distill-Qwen-7B",
        "judge_csv": REPO_ROOT / "analysis_claude" / "claude_judgements.csv",
        "model_col": "DeepSeek-R1-Distill-Qwen-7B",
        "qtype":     "critical",
        "layer":     19,
        "color":     "#4878CF",
        "label":     "DeepSeek-R1-7B",
    },
    "Llama-3.1-8B": {
        "dir":       REPO_ROOT / "data" / "Llama-3.1-8B-Instruct",
        "judge_csv": REPO_ROOT / "analysis_claude" / "llama31_judgements_haiku.csv",
        "model_col": "Llama-3.1-8B-Instruct",
        "qtype":     "critical",
        "layer":     9,
        "color":     "#3aaa35",
        "label":     "Llama-3.1-8B",
    },
    "Qwen2.5-7B": {
        "dir":       REPO_ROOT / "data" / "Qwen2.5-7B-Instruct",
        "judge_csv": REPO_ROOT / "analysis_claude" / "qwen25_judgements_haiku.csv",
        "model_col": "Qwen2.5-7B-Instruct",
        "qtype":     "critical",
        "layer":     17,
        "color":     "#9b59b6",
        "label":     "Qwen2.5-7B",
    },
    "Qwen3.5-9B": {
        "dir":       REPO_ROOT / "data" / "Qwen3.5-9B",
        "judge_csv": REPO_ROOT / "analysis_claude" / "qwen35_judgements_haiku.csv",
        "model_col": "Qwen3.5-9B",
        "qtype":     "critical",
        "layer":     10,
        "color":     "#c0962a",
        "label":     "Qwen3.5-9B",
    },
}
MODEL_KEYS = list(MODELS.keys())

HOLD_C = "#90b8f8"
FLIP_C = "#f4a0a0"

ROW_FC = ["#dceeff", "#dceeff", "#fff3cd", "#ffd6d6"]
ROW_EC = ["#1a3a6b", "#1a3a6b", "#8a6200", "#8b0000"]

CONV_LABELS = [
    "T0\nBaseline\nanswer given",
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

# ── Per-model data loading & PCA ───────────────────────────────────────────

def load_model(key):
    cfg = MODELS[key]
    pt  = cfg["dir"] / f"{cfg['qtype']}_multiturn_hidden_states.pt"
    if not pt.exists():
        print(f"  [{key}] .pt not found — skipping")
        return None

    print(f"  Loading {key} ...")
    hs = torch.load(pt, map_location="cpu")
    questions = list(hs.keys())

    df = pd.read_csv(cfg["judge_csv"])
    df = df[(df["model"] == cfg["model_col"]) &
            (df["question_type"] == cfg["qtype"])].copy()
    df["flip"] = df["judgement"].astype(str).str.lower() == "true"

    first_flip = {}
    for q in questions:
        q_df = df[df["question"] == q].sort_values("turn")
        ff = q_df[q_df["flip"]]["turn"]
        first_flip[q] = int(ff.min()) if len(ff) > 0 else None

    layer = cfg["layer"]

    def get_vec(q, turn):
        try:
            t = hs[q][turn]
            return None if t is None else t.float()[layer].numpy()
        except Exception:
            return None

    # Valid questions: T0–T3 all present, in judge CSV
    valid = [q for q in questions
             if q in first_flip and
             all(get_vec(q, t) is not None for t in range(4))]

    X0     = np.array([get_vec(q, 0) for q in valid])
    scaler = StandardScaler()
    X0_sc  = scaler.fit_transform(X0)
    pca    = PCA(n_components=2, random_state=42)
    pca.fit(X0_sc)

    def project(q, turn):
        v = get_vec(q, turn)
        return None if v is None else pca.transform(scaler.transform(v.reshape(1, -1)))[0]

    turn_pts = {t: np.array([project(q, t) for q in valid]) for t in range(4)}
    eventual_flip = np.array([first_flip[q] is not None for q in valid])

    # Example: flips at T3 preferred, else nearest available flip turn
    for target_turn in [3, 2, 4, 1, 5]:
        cands = [i for i, q in enumerate(valid) if first_flip[q] == target_turn]
        if cands:
            ex_idx = max(cands, key=lambda i: abs(
                turn_pts[3][i, 0] - turn_pts[0][i, 0]))
            ex_flip_turn = target_turn
            break
    else:
        ex_idx = 0
        ex_flip_turn = first_flip[valid[0]] or 5

    # Common axis limits (3–97th pct)
    all_x = np.concatenate([turn_pts[t][:, 0] for t in range(4)])
    all_y = np.concatenate([turn_pts[t][:, 1] for t in range(4)])
    xlo, xhi = np.percentile(all_x, 3), np.percentile(all_x, 97)
    ylo, yhi = np.percentile(all_y, 3), np.percentile(all_y, 97)
    xp, yp = 0.18 * (xhi - xlo), 0.18 * (yhi - ylo)

    n_t3 = sum(1 for q in valid if first_flip[q] == 3)
    print(f"    {len(valid)} questions, {n_t3} flip at T3, "
          f"example idx={ex_idx} (flips at T{ex_flip_turn})")

    return dict(
        turn_pts=turn_pts,
        eventual_flip=eventual_flip,
        ex_idx=ex_idx,
        ex_flip_turn=ex_flip_turn,
        xlim=(xlo - xp, xhi + xp),
        ylim=(ylo - yp, yhi + yp),
        color=cfg["color"],
    )


print("Loading models...")
model_data = {k: load_model(k) for k in MODEL_KEYS}
model_data = {k: v for k, v in model_data.items() if v is not None}
active_keys = [k for k in MODEL_KEYS if k in model_data]
N_MODELS = len(active_keys)

# ── Figure ─────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(4 + N_MODELS * 3.1, 11))
outer = gridspec.GridSpec(
    1, 3, figure=fig,
    left=0.03, right=0.97, top=0.88, bottom=0.06,
    wspace=0.06,
    width_ratios=[0.9, N_MODELS * 2.8, 0.9],
)
left_gs  = gridspec.GridSpecFromSubplotSpec(4, 1, subplot_spec=outer[0], hspace=0.18)
mid_gs   = gridspec.GridSpecFromSubplotSpec(4, N_MODELS, subplot_spec=outer[1],
                                             hspace=0.12, wspace=0.08)
right_gs = gridspec.GridSpecFromSubplotSpec(4, 1, subplot_spec=outer[2], hspace=0.18)


def textbox(fig, spec, text, fc, ec, fontsize=9.5):
    ax = fig.add_subplot(spec)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    ax.add_patch(FancyBboxPatch(
        (0.06, 0.06), 0.88, 0.88,
        boxstyle="round,pad=0.05", linewidth=1.8,
        facecolor=fc, edgecolor=ec, zorder=1,
    ))
    ax.text(0.5, 0.5, text, ha="center", va="center",
            fontsize=fontsize, fontweight="bold", color=ec,
            multialignment="center", transform=ax.transAxes, zorder=2)


for row in range(4):
    textbox(fig, left_gs[row],  CONV_LABELS[row], ROW_FC[row], ROW_EC[row])
    textbox(fig, right_gs[row], BEH_LABELS[row],  ROW_FC[row], ROW_EC[row])

# ── PCA thumbnails ─────────────────────────────────────────────────────────
for col, key in enumerate(active_keys):
    d     = model_data[key]
    mcol  = d["color"]
    eidx  = d["ex_idx"]
    eflip = d["ex_flip_turn"]

    for row in range(4):
        ax = fig.add_subplot(mid_gs[row, col])

        xs = d["turn_pts"][row][:, 0]
        ys = d["turn_pts"][row][:, 1]

        # Background population
        ax.scatter(xs[~d["eventual_flip"]], ys[~d["eventual_flip"]],
                   s=7, color=HOLD_C, alpha=0.50, linewidths=0, zorder=2)
        ax.scatter(xs[d["eventual_flip"]],  ys[d["eventual_flip"]],
                   s=7, color=FLIP_C, alpha=0.50, linewidths=0, zorder=2)

        # Trajectory arrows T0 → current turn
        trail = [d["turn_pts"][t][eidx] for t in range(row + 1)]
        for i in range(len(trail) - 1):
            ax.annotate(
                "", xy=trail[i + 1], xytext=trail[i],
                arrowprops=dict(
                    arrowstyle="->,head_width=0.18,head_length=0.18",
                    color="#333", lw=1.4,
                ),
                zorder=5,
            )
        for t, pt in enumerate(trail):
            is_cur = (t == row)
            ax.scatter(
                pt[0], pt[1],
                s=100 if is_cur else 35,
                color=mcol,
                alpha=1.0 if is_cur else 0.35,
                edgecolors="white" if is_cur else "none",
                linewidths=1.2 if is_cur else 0,
                zorder=6,
            )

        ax.set_xlim(*d["xlim"]); ax.set_ylim(*d["ylim"])
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_linewidth(1.3)
            sp.set_edgecolor(ROW_EC[row])

        # T-label top-left (only first col to avoid clutter)
        if col == 0:
            ax.text(0.04, 0.94, f"T{row}", transform=ax.transAxes,
                    fontsize=8, fontweight="bold", color=ROW_EC[row],
                    va="top", ha="left")

        # Mark the flip turn on the example trajectory
        if row == eflip - 1:
            ax.text(0.96, 0.06, "flip", transform=ax.transAxes,
                    fontsize=6.5, color="#aa0000", va="bottom", ha="right",
                    fontstyle="italic")

# ── Column model headers ───────────────────────────────────────────────────
fig.canvas.draw()
for col, key in enumerate(active_keys):
    spec = mid_gs[0, col]
    bbox = spec.get_position(fig)
    cx   = (bbox.x0 + bbox.x1) / 2
    fig.text(cx, 0.895, MODELS[key]["label"],
             ha="center", fontsize=9.5, fontweight="bold",
             color=MODELS[key]["color"])

# ── Section headers ────────────────────────────────────────────────────────
left_bbox  = outer[0].get_position(fig)
mid_bbox   = outer[1].get_position(fig)
right_bbox = outer[2].get_position(fig)
hdr_y = 0.895
fig.text((left_bbox.x0 + left_bbox.x1) / 2,  hdr_y + 0.045,
         "Conversation", ha="center", fontsize=10.5, fontweight="bold", color="#222")
fig.text((mid_bbox.x0 + mid_bbox.x1) / 2,    hdr_y + 0.045,
         "Hidden-State Space  (PCA-2, T0-fitted)",
         ha="center", fontsize=10.5, fontweight="bold", color="#222")
fig.text((right_bbox.x0 + right_bbox.x1) / 2, hdr_y + 0.045,
         "Behavior", ha="center", fontsize=10.5, fontweight="bold", color="#222")

# ── Legend ─────────────────────────────────────────────────────────────────
handles = [
    Line2D([0], [0], marker="o", color="w", markerfacecolor=HOLD_C,
           markersize=8, label="Population: Eventually Holds"),
    Line2D([0], [0], marker="o", color="w", markerfacecolor=FLIP_C,
           markersize=8, label="Population: Eventually Flips"),
    Line2D([0], [0], marker="o", color="w", markerfacecolor="#555",
           markersize=8, label="Example trajectory (model-colored dot)"),
]
fig.legend(handles=handles, loc="lower center", ncol=3,
           fontsize=8.5, frameon=True, framealpha=0.9,
           bbox_to_anchor=(0.5, 0.005), edgecolor="#bbb")

fig.suptitle(
    "Pre-Capitulation Hidden-State Drift Across Models\n"
    "critical questions · LLM-as-judge labels · each model's best pre-flip layer",
    fontsize=11, y=0.995, color="#111",
)

plt.savefig(OUT_PNG, dpi=160, bbox_inches="tight")
print(f"\nSaved → {OUT_PNG}")
