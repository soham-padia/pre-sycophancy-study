#!/usr/bin/env python3
"""Bar chart: ever-flip rate by model and question type (LLM-as-judge labels)."""
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_PNG = REPO_ROOT / "analysis_claude" / "fliprate_by_model_qtype.png"

JUDGE_CSVS = {
    "DeepSeek-R1-7B":  (REPO_ROOT / "analysis_claude" / "claude_judgements.csv",        "DeepSeek-R1-Distill-Qwen-7B"),
    "Gemma-2-9B":      (REPO_ROOT / "analysis_claude" / "gemma_judgements_haiku.csv",   "Gemma-2-9B"),
    "Llama-3.1-8B":    (REPO_ROOT / "analysis_claude" / "llama31_judgements_haiku.csv", "Llama-3.1-8B-Instruct"),
    "Qwen2.5-7B":      (REPO_ROOT / "analysis_claude" / "qwen25_judgements_haiku.csv",  "Qwen2.5-7B-Instruct"),
    "Qwen3.5-9B":      (REPO_ROOT / "analysis_claude" / "qwen35_judgements_haiku.csv",  "Qwen3.5-9B"),
}

QTYPES = ["base", "critical", "presupposition"]
QTYPE_LABELS = ["Base", "Critical", "Presupposition"]
COLORS = ["#4878CF", "#D65F5F", "#6ACC65"]  # blue, red, green

def ever_flip_rate(df, model_col, qtype):
    sub = df[(df["model"] == model_col) & (df["question_type"] == qtype)].copy()
    sub["flip"] = sub["judgement"].astype(str).str.lower() == "true"
    rates = sub.groupby("question")["flip"].any()
    return rates.mean() if len(rates) else float("nan")

rows = []
for model_label, (csv_path, model_col) in JUDGE_CSVS.items():
    df = pd.read_csv(csv_path)
    for qt in QTYPES:
        rate = ever_flip_rate(df, model_col, qt)
        rows.append({"model": model_label, "qtype": qt, "rate": rate})

data = pd.DataFrame(rows)

models = list(JUDGE_CSVS.keys())
x = np.arange(len(models))
width = 0.25

fig, ax = plt.subplots(figsize=(9, 5))

for i, (qt, qt_label, color) in enumerate(zip(QTYPES, QTYPE_LABELS, COLORS)):
    vals = [data[(data["model"] == m) & (data["qtype"] == qt)]["rate"].values[0] * 100
            for m in models]
    bars = ax.bar(x + (i - 1) * width, vals, width, label=qt_label, color=color,
                  edgecolor="white", linewidth=0.6)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.8,
                f"{v:.0f}%", ha="center", va="bottom", fontsize=8, color="#333333")

ax.set_xticks(x)
ax.set_xticklabels(models, fontsize=10)
ax.set_ylabel("Ever-Flip Rate (%)", fontsize=11)
ax.set_title("Sycophantic Flip Rate by Model and Question Type\n(Judge: Claude Haiku 4.5, first-flip methodology)",
             fontsize=11, pad=10)
ax.set_ylim(0, 105)
ax.yaxis.grid(True, linestyle="--", alpha=0.5, zorder=0)
ax.set_axisbelow(True)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.legend(title="Question Type", fontsize=9, title_fontsize=9,
          loc="upper right", framealpha=0.8)

plt.tight_layout()
plt.savefig(OUT_PNG, dpi=180, bbox_inches="tight")
print(f"Saved → {OUT_PNG}")
