#!/usr/bin/env python3
"""Two-panel figure for Early Hidden-State Disruption section.

Panel A: AUC by layer (cosine similarity zero-shot predictor) — one line per model,
         best question type selected per model.
Panel B: Logistic regression accuracy by layer — one line per model, showing that
         linear probes remain at or below chance across ALL layers (not just best).
"""
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_PNG = REPO_ROOT / "analysis_claude" / "hidden_state_disruption.png"

COSINE_CSV  = REPO_ROOT / "analysis_claude" / "cosine_disruption.csv"
SWEEP_CSV   = REPO_ROOT / "analysis_claude" / "layer_sweep_accuracy.csv"

MODEL_COLORS = {
    "DeepSeek-R1-7B": "#4878CF",
    "Qwen2.5-7B":     "#B47CC7",
    "Llama-3.1-8B":   "#6ACC65",
    "Qwen3.5-9B":     "#C4AD66",
}
MODEL_ORDER = ["Gemma-2-9B", "DeepSeek-R1-7B", "Llama-3.1-8B", "Qwen3.5-9B", "Qwen2.5-7B"]

# Best qtype per model (highest peak AUC from the analysis)
BEST_QTYPE = {
    "DeepSeek-R1-7B": "critical",      # AUC=0.617 at layer 21
    "Qwen2.5-7B":     "base",          # AUC=0.630 at layer 26
    "Llama-3.1-8B":   "critical",      # AUC=0.636 at layer 32
    "Qwen3.5-9B":     "presupposition",# AUC=0.596 at layer 4
}

QTYPE_LABELS = {"base": "Base", "critical": "Critical", "presupposition": "Presup."}

# ── Load data ──────────────────────────────────────────────────────────────────
cos_df   = pd.read_csv(COSINE_CSV)
sweep_df = pd.read_csv(SWEEP_CSV)

# ── Figure layout ──────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(6.5, 3.6))
gs  = gridspec.GridSpec(1, 2, wspace=0.42, left=0.09, right=0.97, top=0.76, bottom=0.15)

ax_auc = fig.add_subplot(gs[0])
ax_lda = fig.add_subplot(gs[1])

# ── Panel A: AUC by layer ─────────────────────────────────────────────────────
for model, color in MODEL_COLORS.items():
    qt = BEST_QTYPE[model]
    sub = cos_df[(cos_df["model"] == model) & (cos_df["qtype"] == qt)].sort_values("layer")
    if sub.empty:
        continue
    ax_auc.plot(sub["layer"], sub["auc"], color=color, lw=1.8,
                label=f"{model} ({QTYPE_LABELS[qt]})")
    # Mark peak
    peak = sub.loc[sub["auc"].idxmax()]
    ax_auc.scatter(peak["layer"], peak["auc"], color=color, s=55, zorder=4)

ax_auc.axhline(0.5, color="#999999", lw=1.2, ls="--", label="Chance (AUC=0.5)")
ax_auc.set_xlabel("Layer", fontsize=9)
ax_auc.set_ylabel("ROC-AUC  (cos sim T0→T1 predicts flip)", fontsize=8.5)
ax_auc.set_title("A.  Zero-Shot Cosine Disruption AUC", fontsize=9, pad=5)
ax_auc.legend(fontsize=8, loc="lower right", framealpha=0.85)
ax_auc.tick_params(labelsize=8.5)
ax_auc.set_ylim(0.35, 0.70)
ax_auc.spines["top"].set_visible(False)
ax_auc.spines["right"].set_visible(False)
ax_auc.yaxis.grid(True, linestyle="--", alpha=0.3, zorder=0)
ax_auc.set_axisbelow(True)

# ── Panel B: Logistic regression accuracy by layer ────────────────────────────
for model, color in MODEL_COLORS.items():
    sub = sweep_df[sweep_df["model"] == model].sort_values("layer")
    if sub.empty:
        continue
    qt = BEST_QTYPE[model]
    chance = sub["chance"].iloc[0]
    ax_lda.plot(sub["layer"], sub["acc"], color=color, lw=1.8,
                label=f"{model} ({QTYPE_LABELS[qt]})")
    # Draw per-model chance as a faint horizontal line
    ax_lda.axhline(chance, color=color, lw=0.7, ls=":", alpha=0.45)

ax_lda.set_xlabel("Layer", fontsize=9)
ax_lda.set_ylabel("Logistic regression cross-val accuracy", fontsize=8.5)
ax_lda.set_title("B.  Linear Probe Accuracy by Layer", fontsize=9, pad=5)
ax_lda.legend(fontsize=8, loc="upper right", framealpha=0.85)
ax_lda.tick_params(labelsize=8.5)
ax_lda.set_ylim(0.35, 0.95)
ax_lda.spines["top"].set_visible(False)
ax_lda.spines["right"].set_visible(False)
ax_lda.yaxis.grid(True, linestyle="--", alpha=0.3, zorder=0)
ax_lda.set_axisbelow(True)

fig.suptitle(
    "Hidden-State Disruption: Zero-Shot Cosine Signal vs. Linear Probe Failure",
    fontsize=10, y=0.995,
)

plt.savefig(OUT_PNG, dpi=300, bbox_inches="tight")
print(f"Saved → {OUT_PNG}")
