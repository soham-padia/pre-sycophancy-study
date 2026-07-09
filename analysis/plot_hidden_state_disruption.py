#!/usr/bin/env python3
"""Four-panel central evidence figure for the cosine disruption signal.

Panel A: T0->T1 cosine AUC by layer with 95% bootstrap CI bands
         (best prompt condition per model).
Panel B: the same after excluding questions whose first flip occurs at T1
         (pre-behavioral robustness check).
Panel C: peak AUC per model with 95% bootstrap CIs (from
         cosine_disruption_checks.py, 2000 resamples, seeded).
Panel D: per-layer representational drift (1 - mean cosine) vs. predictive
         AUC, showing the dissociation between movement and predictiveness.

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

fig, axes = plt.subplots(1, 4, figsize=(13.0, 3.1))
fig.subplots_adjust(left=0.05, right=0.99, top=0.82, bottom=0.16, wspace=0.30)
ax_a, ax_b, ax_c, ax_d = axes

# ── Panels A and B: AUC by layer with CI bands ────────────────────────────────
for ax, variant, title in ((ax_a, "all", "A.  Zero-shot cosine AUC by layer"),
                           (ax_b, "excl_t1", "B.  Excluding T1-flip conversations")):
    for model, color in MODEL_COLORS.items():
        sub = bands[(bands["model"] == model) & (bands["variant"] == variant)].sort_values("layer")
        if sub.empty:
            continue
        label = f"{model} ({QTYPE_LABELS[BEST_QTYPE[model]]})" if variant == "all" else None
        ax.plot(sub["layer"], sub["auc"], color=color, lw=1.6, label=label)
        ax.fill_between(sub["layer"], sub["lo"], sub["hi"], color=color, alpha=0.13, lw=0)
        peak = sub.loc[sub["auc"].idxmax()]
        ax.scatter(peak["layer"], peak["auc"], color=color, s=38, zorder=4)
    ax.axhline(0.5, color="#999999", lw=1.0, ls="--")
    ax.set_xlabel("Layer", fontsize=9)
    ax.set_ylabel("ROC-AUC", fontsize=9)
    ax.set_title(title, fontsize=9, pad=5)
    ax.set_ylim(0.30, 0.80)
    ax.tick_params(labelsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.yaxis.grid(True, linestyle="--", alpha=0.3)
    ax.set_axisbelow(True)

# ── Panel C: peak AUC per model with 95% CIs (checks CSV, best condition) ─────
xs, labels_c = [], []
for i, (model, qt) in enumerate(BEST_QTYPE.items()):
    row = checks[(checks["model"] == model) & (checks["qtype"] == qt)].iloc[0]
    color = MODEL_COLORS[model]
    ax_c.errorbar(i, row["auc_original"],
                  yerr=[[row["auc_original"] - row["ci_orig_lo"]],
                        [row["ci_orig_hi"] - row["auc_original"]]],
                  fmt="o", color=color, markersize=6, capsize=4, lw=1.4)
    xs.append(i)
    labels_c.append(f"{SHORT[model]}\n({QTYPE_LABELS[qt]})")
ax_c.axhline(0.5, color="#999999", lw=1.0, ls="--")
ax_c.set_xticks(xs)
ax_c.set_xticklabels(labels_c, fontsize=8)
ax_c.set_ylabel("Peak ROC-AUC", fontsize=9)
ax_c.set_title("C.  Peak AUC with 95% CI", fontsize=9, pad=5)
ax_c.set_ylim(0.40, 0.80)
ax_c.set_xlim(-0.6, len(xs) - 0.4)
ax_c.tick_params(labelsize=8)
ax_c.spines["top"].set_visible(False)
ax_c.spines["right"].set_visible(False)
ax_c.yaxis.grid(True, linestyle="--", alpha=0.3)
ax_c.set_axisbelow(True)

# ── Panel D: drift magnitude vs predictive AUC (per layer) ────────────────────
for model, color in MODEL_COLORS.items():
    sub = bands[(bands["model"] == model) & (bands["variant"] == "all")]
    ax_d.scatter(sub["mean_drift"], sub["auc"], color=color, s=14, alpha=0.65,
                 edgecolors="none")
ax_d.axhline(0.5, color="#999999", lw=1.0, ls="--")
ax_d.set_xlabel(r"Drift magnitude  $1-\overline{\cos}(\mathbf{h}_{T0},\mathbf{h}_{T1})$", fontsize=9)
ax_d.set_ylabel("ROC-AUC", fontsize=9)
ax_d.set_title("D.  Drift vs. predictiveness", fontsize=9, pad=5)
ax_d.set_ylim(0.30, 0.80)
ax_d.tick_params(labelsize=8)
ax_d.spines["top"].set_visible(False)
ax_d.spines["right"].set_visible(False)
ax_d.yaxis.grid(True, linestyle="--", alpha=0.3)
ax_d.set_axisbelow(True)

handles, labels = ax_a.get_legend_handles_labels()
fig.legend(handles, labels, fontsize=8, ncol=4, loc="upper center",
           bbox_to_anchor=(0.5, 1.00), framealpha=0.9, borderaxespad=0.2)

plt.savefig(OUT_PNG, dpi=300, bbox_inches="tight")
print(f"Saved -> {OUT_PNG}")
