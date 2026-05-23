#!/usr/bin/env python3
"""3D PCA maps and 2D projection triptychs — all 4 models.

Outputs:
  analysis_claude/pca_3d.png          — 1×4 grid of 3D scatter + trajectory
  analysis_claude/pca_projections.png — 4×3 grid of 2D projection pairs
"""
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from matplotlib.lines import Line2D
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_3D   = REPO_ROOT / "analysis_claude" / "pca_3d.png"
OUT_PROJ = REPO_ROOT / "analysis_claude" / "pca_projections.png"

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
        "color":     "#2eaa30",
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

# Turn colors: blue (T0) → red (T5)
TURN_COLORS  = ["#2c5282", "#4299e1", "#48bb78", "#f6ad55", "#fc8181", "#c53030"]
TURN_LABELS  = ["T0", "T1", "T2", "T3", "T4", "T5"]
TURN_SIZES   = [80, 80, 80, 120, 80, 80]  # T3 (flip) slightly larger


# ── Data loading ───────────────────────────────────────────────────────────

def load_model_3d(key):
    cfg   = MODELS[key]
    pt    = cfg["dir"] / f"{cfg['qtype']}_multiturn_hidden_states.pt"
    layer = cfg["layer"]

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
        ff   = q_df[q_df["flip"]]["turn"]
        first_flip[q] = int(ff.min()) if len(ff) > 0 else None

    def get_vec(q, turn):
        try:
            t = hs[q][turn]
            return None if t is None else t.float()[layer].numpy()
        except Exception:
            return None

    # Need T0 valid for all background questions
    valid_bg = [q for q in questions if q in first_flip and get_vec(q, 0) is not None]

    # Fit 3-component PCA on T0 vectors
    X0     = np.array([get_vec(q, 0) for q in valid_bg])
    scaler = StandardScaler()
    X0_sc  = scaler.fit_transform(X0)
    pca    = PCA(n_components=3, random_state=42)
    pca.fit(X0_sc)

    def project3(q, turn):
        v = get_vec(q, turn)
        return None if v is None else pca.transform(scaler.transform(v.reshape(1, -1)))[0]

    # Project T0 background (for scatter)
    pts_t0   = np.array([project3(q, 0) for q in valid_bg])
    eventual = np.array([first_flip[q] is not None for q in valid_bg])

    # Example: prefers T3-flip, picks median displacement (avoids outliers)
    valid_full = [q for q in valid_bg
                  if all(get_vec(q, t) is not None for t in range(4))]

    # Compute per-question displacement in 3D space from T0 to flip turn
    def displacement(q, flip_t):
        p0 = project3(q, 0)
        pf = project3(q, min(flip_t, 5))
        if p0 is None or pf is None:
            return 0.0
        return float(np.linalg.norm(pf - p0))

    for target in [3, 2, 4, 1, 5]:
        cands = [i for i, q in enumerate(valid_full) if first_flip[q] == target]
        if cands:
            disps = sorted(cands, key=lambda i: displacement(valid_full[i], target))
            # pick 75th-percentile displacement — clear drift, not an outlier
            ex_idx       = disps[int(len(disps) * 0.75)]
            ex_q         = valid_full[ex_idx]
            ex_flip_turn = target
            break

    # Full T0–T5 trajectory for example
    traj = []
    for t in range(6):
        p = project3(ex_q, t)
        traj.append(p)  # may be None for T4/T5 if missing

    pct_var = pca.explained_variance_ratio_ * 100

    print(f"    {len(valid_bg)} bg questions | example flips at T{ex_flip_turn} "
          f"| PCA var: {pct_var[0]:.1f}% + {pct_var[1]:.1f}% + {pct_var[2]:.1f}%")

    return dict(
        pts_t0       = pts_t0,
        eventual     = eventual,
        traj         = traj,
        ex_flip_turn = ex_flip_turn,
        pct_var      = pct_var,
        color        = cfg["color"],
        label        = cfg["label"],
        # also store all-turn projections for projection figure
        all_pts      = {t: np.array([project3(q, t) for q in valid_bg])
                        for t in range(6)},
    )


print("Loading models...")
model_data = {k: load_model_3d(k) for k in MODEL_KEYS}


# ══════════════════════════════════════════════════════════════════════════
# FIGURE 1 — 3D scatter
# ══════════════════════════════════════════════════════════════════════════

fig3d = plt.figure(figsize=(6.5, 6.2))
fig3d.subplots_adjust(left=0.02, right=0.98, top=0.90, bottom=0.11, wspace=0.05, hspace=0.15)

for col, key in enumerate(MODEL_KEYS):
    d   = model_data[key]
    ax  = fig3d.add_subplot(2, 2, col + 1, projection="3d")

    pts = d["pts_t0"]

    # Clip axes to 4–96th percentile of population, then expand to include trajectory
    lims = []
    traj_pts_arr = np.array([p for p in d["traj"] if p is not None])
    for dim in range(3):
        lo = np.percentile(pts[:, dim], 4)
        hi = np.percentile(pts[:, dim], 96)
        if len(traj_pts_arr):
            lo = min(lo, traj_pts_arr[:, dim].min())
            hi = max(hi, traj_pts_arr[:, dim].max())
        pad = 0.15 * (hi - lo)
        lims.append((lo - pad, hi + pad))

    mask_hold = ~d["eventual"]
    mask_flip =  d["eventual"]

    # Subsample background to reduce visual clutter (50% of each group)
    rng = np.random.default_rng(42)
    idx_hold = rng.choice(mask_hold.sum(), size=max(1, mask_hold.sum() // 2), replace=False)
    idx_flip = rng.choice(mask_flip.sum(), size=max(1, mask_flip.sum() // 2), replace=False)
    pts_hold = pts[mask_hold][idx_hold]
    pts_flip = pts[mask_flip][idx_flip]

    # Background — T0 population, colored by eventual outcome
    ax.scatter(pts_hold[:, 0], pts_hold[:, 1], pts_hold[:, 2],
               s=7, color=HOLD_C, alpha=0.35, linewidths=0, depthshade=True)
    ax.scatter(pts_flip[:, 0], pts_flip[:, 1], pts_flip[:, 2],
               s=7, color=FLIP_C, alpha=0.35, linewidths=0, depthshade=True)

    # Trajectory line T0–T5 (all non-None points)
    traj_valid = [(t, p) for t, p in enumerate(d["traj"]) if p is not None]
    if len(traj_valid) >= 2:
        txs = [p[0] for _, p in traj_valid]
        tys = [p[1] for _, p in traj_valid]
        tzs = [p[2] for _, p in traj_valid]
        ax.plot(txs, tys, tzs, color="#222", lw=2.0, zorder=4, alpha=0.85)

    # Turn markers
    for t, p in traj_valid:
        is_flip = (t == d["ex_flip_turn"])
        ax.scatter(p[0], p[1], p[2],
                   s=TURN_SIZES[t] * (2.2 if is_flip else 1.2),
                   color=TURN_COLORS[t],
                   marker="*" if is_flip else "o",
                   edgecolors="white",
                   linewidths=0.8,
                   depthshade=False, zorder=5)

    ax.set_xlim(*lims[0]); ax.set_ylim(*lims[1]); ax.set_zlim(*lims[2])

    v = d["pct_var"]
    ax.set_xlabel(f"PC1 ({v[0]:.0f}%)", fontsize=9, labelpad=1)
    ax.set_ylabel(f"PC2 ({v[1]:.0f}%)", fontsize=9, labelpad=1)
    ax.set_zlabel(f"PC3 ({v[2]:.0f}%)", fontsize=9, labelpad=1)
    ax.tick_params(labelsize=8)
    ax.set_title(d["label"], fontsize=10.5, fontweight="bold",
                 color=d["color"], pad=6)
    ax.view_init(elev=22, azim=225)
    ax.set_box_aspect([1, 1, 0.75])

# Legend
legend_handles = (
    [Line2D([0], [0], marker="o", color="w", markerfacecolor=HOLD_C,
            markersize=8, label="Population: Eventually Holds"),
     Line2D([0], [0], marker="o", color="w", markerfacecolor=FLIP_C,
            markersize=8, label="Population: Eventually Flips")]
    + [Line2D([0], [0],
              marker=("*" if t == 3 else "o"),
              color="w", markerfacecolor=TURN_COLORS[t],
              markersize=(9 if t == 3 else 7),
              label=TURN_LABELS[t] + (" ← flip" if t == 3 else ""))
       for t in range(6)]
)
fig3d.legend(handles=legend_handles, loc="lower center", ncol=4,
             fontsize=8.5, frameon=True, bbox_to_anchor=(0.5, 0.0),
             edgecolor="#bbb", framealpha=0.92)

fig3d.suptitle(
    "Hidden-State Trajectory in 3D PCA Space  "
    "(critical questions · T0-fitted PCA · ★ = first flip turn)",
    fontsize=10.5, y=0.97,
)
plt.savefig(OUT_3D, dpi=300, bbox_inches="tight")
print(f"Saved → {OUT_3D}")
plt.close(fig3d)


# ══════════════════════════════════════════════════════════════════════════
# FIGURE 2 — 2D projection triptychs  (4 models × 3 projection pairs)
# ══════════════════════════════════════════════════════════════════════════

PROJ_PAIRS  = [(0, 1), (0, 2), (1, 2)]
PROJ_LABELS = [("PC 1", "PC 2"), ("PC 1", "PC 3"), ("PC 2", "PC 3")]

fig2d = plt.figure(figsize=(13, 14))
fig2d.subplots_adjust(left=0.07, right=0.97, top=0.94, bottom=0.06,
                       wspace=0.22, hspace=0.28)
gs = gridspec.GridSpec(4, 3, figure=fig2d)

for row, key in enumerate(MODEL_KEYS):
    d = model_data[key]

    for col, (pi, pj) in enumerate(PROJ_PAIRS):
        ax = fig2d.add_subplot(gs[row, col])

        pts = d["pts_t0"]
        # Background
        ax.scatter(pts[~d["eventual"], pi], pts[~d["eventual"], pj],
                   s=8, color=HOLD_C, alpha=0.45, linewidths=0, zorder=2)
        ax.scatter(pts[d["eventual"],  pi], pts[d["eventual"],  pj],
                   s=8, color=FLIP_C, alpha=0.45, linewidths=0, zorder=2)

        # Trajectory line
        traj_valid = [(t, p) for t, p in enumerate(d["traj"]) if p is not None]
        if len(traj_valid) >= 2:
            txs = [p[pi] for _, p in traj_valid]
            tys = [p[pj] for _, p in traj_valid]
            ax.plot(txs, tys, color="#444", lw=1.4, zorder=4, alpha=0.75)

        # Turn markers
        for t, p in traj_valid:
            is_flip = (t == d["ex_flip_turn"])
            ax.scatter(p[pi], p[pj],
                       s=130 if is_flip else 55,
                       color=TURN_COLORS[t],
                       marker="*" if is_flip else "o",
                       edgecolors="white" if is_flip else "none",
                       linewidths=1.0 if is_flip else 0,
                       zorder=5)

        v = d["pct_var"]
        var_map = {0: v[0], 1: v[1], 2: v[2]}
        xl, yl = PROJ_LABELS[col]
        ax.set_xlabel(f"{xl} ({var_map[pi]:.0f}%)", fontsize=8)
        ax.set_ylabel(f"{yl} ({var_map[pj]:.0f}%)", fontsize=8)
        ax.tick_params(labelsize=6.5)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        # Model label on left-most panel only
        if col == 0:
            ax.set_ylabel(f"{yl} ({var_map[pj]:.0f}%)\n", fontsize=8)
            ax.annotate(d["label"],
                        xy=(-0.28, 0.5), xycoords="axes fraction",
                        ha="right", va="center", fontsize=9.5,
                        fontweight="bold", color=d["color"],
                        rotation=90)

        # Projection label on top row only
        if row == 0:
            xl_full = ["PC1 vs PC2", "PC1 vs PC3", "PC2 vs PC3"][col]
            ax.set_title(xl_full, fontsize=10, fontweight="bold", color="#333")

# Legend (turn colors)
leg_handles = (
    [Line2D([0], [0], marker="o", color="w", markerfacecolor=HOLD_C,
            markersize=8, label="Eventually Holds"),
     Line2D([0], [0], marker="o", color="w", markerfacecolor=FLIP_C,
            markersize=8, label="Eventually Flips")]
    + [Line2D([0], [0],
              marker=("*" if t == 3 else "o"),
              color="w", markerfacecolor=TURN_COLORS[t],
              markersize=(10 if t == 3 else 7),
              label=TURN_LABELS[t] + (" ← flip" if t == 3 else ""))
       for t in range(6)]
)
fig2d.legend(handles=leg_handles, loc="lower center", ncol=8,
             fontsize=8.5, frameon=True, bbox_to_anchor=(0.5, 0.00),
             edgecolor="#bbb", framealpha=0.92)

fig2d.suptitle(
    "Hidden-State Trajectory: 2D Projection Pairs  "
    "(critical questions · T0-fitted 3-component PCA · ★ = first flip turn)",
    fontsize=11, y=0.975,
)
plt.savefig(OUT_PROJ, dpi=160, bbox_inches="tight")
print(f"Saved → {OUT_PROJ}")
plt.close(fig2d)
