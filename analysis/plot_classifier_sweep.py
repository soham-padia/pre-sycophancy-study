#!/usr/bin/env python3
"""11-classifier sweep bar chart for Pre-Flip Signals are Nonlinearly Encoded section.

Numbers sourced from final report (Qwen2.5-7B presupposition pre-flip experiment).
"""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_PNG = REPO_ROOT / "analysis_claude" / "classifier_sweep.png"

# Results from 11-classifier sweep (Qwen presupposition pre-flip, chance = 63.8%)
CHANCE = 63.8
CLASSIFIERS = [
    ("RBF SVM",           75.5, "nonlinear"),
    ("Random Forest",     75.5, "nonlinear"),
    ("KNN (k=10)",        75.5, "nonlinear"),
    ("MLP (128+64)",      73.6, "nonlinear"),
    ("Naive Bayes",       72.5, "nonlinear"),
    ("Logistic Reg.",     64.2, "linear"),
    ("Linear SVM",        60.4, "linear"),
]

COLORS = {"nonlinear": "#4878CF", "linear": "#D65F5F"}
LABELS = {"nonlinear": "Nonlinear / ensemble", "linear": "Linear"}

fig, ax = plt.subplots(figsize=(8, 4.2))
fig.subplots_adjust(left=0.22, right=0.97, top=0.88, bottom=0.12)

names  = [c[0] for c in CLASSIFIERS]
accs   = [c[1] for c in CLASSIFIERS]
colors = [COLORS[c[2]] for c in CLASSIFIERS]
y      = np.arange(len(CLASSIFIERS))

bars = ax.barh(y, accs, color=colors, alpha=0.85, height=0.6, zorder=3)

# Accuracy labels
for bar, acc in zip(bars, accs):
    ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
            f"{acc:.1f}%", va="center", ha="left", fontsize=8.5, fontweight="bold")

# Chance line
ax.axvline(CHANCE, color="#333333", lw=1.5, ls="--", zorder=4,
           label=f"Majority-class chance ({CHANCE}%)")

ax.set_yticks(y)
ax.set_yticklabels(names, fontsize=9)
ax.set_xlabel("Cross-validation accuracy (%)", fontsize=9.5)
ax.set_xlim(55, 82)
ax.set_title(
    "11-Classifier Sweep: Pre-Flip Prediction Accuracy\n"
    "(Qwen2.5-7B, presupposition questions, pre-flip hidden states)",
    fontsize=10, pad=8,
)

# Custom legend for classifier type
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor=COLORS["nonlinear"], alpha=0.85, label=LABELS["nonlinear"]),
    Patch(facecolor=COLORS["linear"],    alpha=0.85, label=LABELS["linear"]),
    plt.Line2D([0], [0], color="#333333", lw=1.5, ls="--",
               label=f"Majority-class chance ({CHANCE}%)"),
]
ax.legend(handles=legend_elements, fontsize=8, loc="lower right", framealpha=0.88)

ax.yaxis.grid(False)
ax.xaxis.grid(True, linestyle="--", alpha=0.3, zorder=0)
ax.set_axisbelow(True)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.invert_yaxis()

plt.savefig(OUT_PNG, dpi=180, bbox_inches="tight")
print(f"Saved → {OUT_PNG}")
