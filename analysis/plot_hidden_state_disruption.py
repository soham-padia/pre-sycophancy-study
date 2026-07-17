#!/usr/bin/env python3
"""Two-panel central evidence figure for the cosine disruption signal.

Panel A: T0->T1 cosine AUC by layer with 95% bootstrap CI bands
         (best prompt condition per model).
Panel B: peak AUC per model with 95% bootstrap CIs, computed on all
         conversations (filled) and after excluding conversations whose first
         flip occurs at T1 (open) — the pre-behavioral robustness check.

The drift-vs-AUC dissociation panel lives in analysis/plot_drift_vs_auc.py
(paper appendix).

Data sources (all committed; regenerate with):
    python analysis/cosine_auc_ci_bands.py      -> cosine_auc_ci_bands.csv
    python analysis/cosine_disruption_checks.py -> cosine_disruption_checks.csv
"""
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_PNG   = REPO_ROOT / "analysis_claude" / "hidden_state_disruption.png"

BANDS_CSV  = REPO_ROOT / "analysis_claude" / "cosine_auc_ci_bands.csv"
CHECKS_CSV = REPO_ROOT / "analysis_claude" / "cosine_disruption_checks.csv"

MODEL_COLORS = {
    "DeepSeek-R1-7B": "#4878CF",
    "Qwen2.5-7B":     "#B47CC7",
    "Llama-3.1-8B":   "#6ACC65",
    "Qwen3.5-9B":     "#C4AD66",
}

# Best condition per model (matches the cosine analysis / paper Sec. 6.1)
BEST_QTYPE = {
    "DeepSeek-R1-7B": "critical",
    "Qwen2.5-7B":     "base",
    "Llama-3.1-8B":   "critical",
    "Qwen3.5-9B":     "presupposition",
}

QTYPE_LABELS = {"base": "Base", "critical": "Critical", "presupposition": "Presup."}
SHORT = {"DeepSeek-R1-7B": "DS", "Qwen2.5-7B": "Q2.5",
         "Llama-3.1-8B": "L3.1", "Qwen3.5-9B": "Q3.5"}

bands  = pd.read_csv(BANDS_CSV)
checks = pd.read_csv(CHECKS_CSV)

fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(6.3, 2.9),
                                 gridspec_kw={"width_ratios": [1.35, 1.0]})
fig.subplots_adjust(left=0.09, right=0.98, top=0.80, bottom=0.17, wspace=0.28)

# ── Panel A: AUC by layer with CI bands ───────────────────────────────────────
for model, color in MODEL_COLORS.items():
    sub = bands[(bands["model"] == model) & (bands["variant"] == "all")].sort_values("layer")
    if sub.empty:
        continue
    ax_a.plot(sub["layer"], sub["auc"], color=color, lw=1.8,
              label=f"{SHORT[model]} ({QTYPE_LABELS[BEST_QTYPE[model]]})")
    ax_a.fill_between(sub["layer"], sub["lo"], sub["hi"], color=color, alpha=0.13, lw=0)
    peak = sub.loc[sub["auc"].idxmax()]
    ax_a.scatter(peak["layer"], peak["auc"], color=color, s=42, zorder=4,
                 edgecolors="white", linewidths=0.8)
ax_a.axhline(0.5, color="#888888", lw=1.2, ls="--")
ax_a.set_xlabel("Layer", fontsize=11)
ax_a.set_ylabel("ROC-AUC", fontsize=11)
ax_a.set_title("(A)  Cosine AUC across layers", fontsize=11, pad=5)
ax_a.set_ylim(0.33, 0.78)
ax_a.tick_params(labelsize=10)
ax_a.spines["top"].set_visible(False)
ax_a.spines["right"].set_visible(False)
ax_a.yaxis.grid(True, linestyle="--", alpha=0.3)
ax_a.set_axisbelow(True)

# ── Panel B: peak AUC per model, all vs. excluding-T1-flip, with 95% CIs ─────
OFFSET = 0.16
for i, (model, qt) in enumerate(BEST_QTYPE.items()):
    row = checks[(checks["model"] == model) & (checks["qtype"] == qt)].iloc[0]
    color = MODEL_COLORS[model]
    ax_b.errorbar(i - OFFSET, row["auc_original"],
                  yerr=[[row["auc_original"] - row["ci_orig_lo"]],
                        [row["ci_orig_hi"] - row["auc_original"]]],
                  fmt="o", color=color, markersize=6.5, capsize=3.5, lw=1.4)
    ax_b.errorbar(i + OFFSET, row["auc_excl_t1"],
                  yerr=[[row["auc_excl_t1"] - row["ci_excl_lo"]],
                        [row["ci_excl_hi"] - row["auc_excl_t1"]]],
                  fmt="o", color=color, markersize=6.5, capsize=3.5, lw=1.4,
                  markerfacecolor="white")
ax_b.axhline(0.5, color="#888888", lw=1.2, ls="--")
ax_b.set_xticks(range(len(BEST_QTYPE)))
ax_b.set_xticklabels([SHORT[m] for m in BEST_QTYPE], fontsize=10)
ax_b.set_ylabel("Peak ROC-AUC", fontsize=11)
ax_b.set_title("(B)  Peak AUC: all vs. excl. T1-flip", fontsize=11, pad=5)
ax_b.set_ylim(0.40, 0.80)
ax_b.set_xlim(-0.6, len(BEST_QTYPE) - 0.4)
ax_b.tick_params(labelsize=10)
ax_b.spines["top"].set_visible(False)
ax_b.spines["right"].set_visible(False)
ax_b.yaxis.grid(True, linestyle="--", alpha=0.3)
ax_b.set_axisbelow(True)
marker_all  = plt.Line2D([], [], marker="o", color="#555555", ls="", markersize=6.5, label="All")
marker_excl = plt.Line2D([], [], marker="o", color="#555555", ls="", markersize=6.5,
                         markerfacecolor="white", label="Excl. T1-flip")
ax_b.legend(handles=[marker_all, marker_excl], fontsize=9, loc="lower right",
            framealpha=0.9, borderpad=0.4)

handles, labels = ax_a.get_legend_handles_labels()
fig.legend(handles, labels, fontsize=10, ncol=4, loc="upper center",
           bbox_to_anchor=(0.5, 1.02), framealpha=0.9, columnspacing=1.2,
           handlelength=1.6)

plt.savefig(OUT_PNG, dpi=300, bbox_inches="tight")
print(f"Saved -> {OUT_PNG}")
