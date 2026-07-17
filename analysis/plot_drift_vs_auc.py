#!/usr/bin/env python3
"""Drift magnitude vs. predictive AUC (paper Appendix: drift dissociation).

One point per layer (best prompt condition per model): mean representational
drift 1 - cos(h_T0, h_T1) against that layer's cosine ROC-AUC.

Data source: analysis_claude/cosine_auc_ci_bands.csv
(regenerate with python analysis/cosine_auc_ci_bands.py).
"""
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_PNG   = REPO_ROOT / "analysis_claude" / "drift_vs_auc.png"
BANDS_CSV = REPO_ROOT / "analysis_claude" / "cosine_auc_ci_bands.csv"

MODEL_COLORS = {
    "DeepSeek-R1-7B": "#4878CF",
    "Qwen2.5-7B":     "#B47CC7",
    "Llama-3.1-8B":   "#6ACC65",
    "Qwen3.5-9B":     "#C4AD66",
}
LEGEND_NAMES = {
    "DeepSeek-R1-7B": "DeepSeek",
    "Qwen2.5-7B":     "Qwen2.5",
    "Llama-3.1-8B":   "Llama-3.1",
    "Qwen3.5-9B":     "Qwen3.5",
}

bands = pd.read_csv(BANDS_CSV)
bands = bands[bands["variant"] == "all"]

fig, ax = plt.subplots(figsize=(4.4, 3.6))
fig.subplots_adjust(left=0.15, right=0.97, top=0.97, bottom=0.16)

for model, color in MODEL_COLORS.items():
    sub = bands[bands["model"] == model]
    ax.scatter(sub["mean_drift"], sub["auc"], color=color, s=42, alpha=0.75,
               edgecolors="none", label=LEGEND_NAMES[model])

ax.axhline(0.5, color="#888888", lw=1.4, ls="--")
ax.set_xlabel(r"Representational drift  $1-\cos(\mathbf{h}_{T0},\mathbf{h}_{T1})$",
              fontsize=12)
ax.set_ylabel("ROC-AUC", fontsize=12)
ax.tick_params(labelsize=11)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.grid(True, linestyle="--", alpha=0.3)
ax.set_axisbelow(True)
ax.legend(fontsize=10, loc="lower right", framealpha=0.9)

plt.savefig(OUT_PNG, dpi=300, bbox_inches="tight")
print(f"Saved -> {OUT_PNG}")
