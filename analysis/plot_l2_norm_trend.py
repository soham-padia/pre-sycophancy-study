#!/usr/bin/env python3
"""L2 norm activation magnitude over pressure turns: flip vs. hold groups.

For each model, computes the mean L2 norm of the last-token hidden state
(averaged across all layers) at each turn T0–T5. Questions are split into
ever-flip (will eventually capitulate) and hold (never capitulate) groups
using LLM-as-judge labels. Post-flip turns are censored so only pre-flip
states contribute to the flip-group average.

Output:
  analysis_claude/l2_norm_trend.png
  analysis_claude/l2_norm_trend.csv
"""
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_PNG = REPO_ROOT / "analysis_claude" / "l2_norm_trend.png"
OUT_CSV = REPO_ROOT / "analysis_claude" / "l2_norm_trend.csv"

MODELS = {
    "DeepSeek-R1-7B": {
        "dir":       REPO_ROOT / "data" / "DeepSeek-R1-Distill-Qwen-7B",
        "judge_csv": REPO_ROOT / "analysis_claude" / "claude_judgements.csv",
        "model_col": "DeepSeek-R1-Distill-Qwen-7B",
    },
    "Qwen2.5-7B": {
        "dir":       REPO_ROOT / "data" / "Qwen2.5-7B-Instruct",
        "judge_csv": REPO_ROOT / "analysis_claude" / "qwen25_judgements_haiku.csv",
        "model_col": "Qwen2.5-7B-Instruct",
    },
    "Llama-3.1-8B": {
        "dir":       REPO_ROOT / "data" / "Llama-3.1-8B-Instruct",
        "judge_csv": REPO_ROOT / "analysis_claude" / "llama31_judgements_haiku.csv",
        "model_col": "Llama-3.1-8B-Instruct",
    },
    "Qwen3.5-9B": {
        "dir":       REPO_ROOT / "data" / "Qwen3.5-9B",
        "judge_csv": REPO_ROOT / "analysis_claude" / "qwen35_judgements_haiku.csv",
        "model_col": "Qwen3.5-9B",
    },
}

QTYPES = ["base", "critical", "presupposition"]

MODEL_COLORS = {
    "DeepSeek-R1-7B": "#4878CF",
    "Qwen2.5-7B":     "#B47CC7",
    "Llama-3.1-8B":   "#6ACC65",
    "Qwen3.5-9B":     "#C4AD66",
}


def load_judge_labels(cfg):
    """Returns {question: first_flip_turn_or_None} aggregated across all qtypes."""
    labels = {}
    df = pd.read_csv(cfg["judge_csv"])
    df = df[df["model"] == cfg["model_col"]].copy()
    df["flip"] = df["judgement"].astype(str).str.lower() == "true"
    for q, grp in df.groupby("question"):
        flipped = grp[grp["flip"]]["turn"]
        labels[q] = int(flipped.min()) if len(flipped) > 0 else None
    return labels


def l2_norm_over_turns(hs, labels):
    """
    Returns two dicts: flip_norms[turn] and hold_norms[turn],
    each a list of per-question mean-layer L2 norms at that turn.
    Censors post-flip turns for flip questions.
    """
    flip_norms = {t: [] for t in range(6)}
    hold_norms = {t: [] for t in range(6)}

    for q, first_flip in labels.items():
        if q not in hs:
            continue
        turns = hs[q]
        ever_flip = first_flip is not None

        for t, tensor in enumerate(turns):
            if tensor is None:
                continue
            # censor post-flip turns
            if ever_flip and first_flip is not None and t > first_flip:
                continue
            # mean L2 norm across all layers
            norm = float(tensor.float().norm(dim=1).mean())
            if ever_flip:
                flip_norms[t].append(norm)
            else:
                hold_norms[t].append(norm)

    return flip_norms, hold_norms


def run():
    rows = []
    fig, axes = plt.subplots(1, 4, figsize=(6.5, 2.4), sharey=False)
    fig.subplots_adjust(wspace=0.38, left=0.09, right=0.97, top=0.78, bottom=0.18)

    for ax, (model_name, cfg) in zip(axes, MODELS.items()):
        color = MODEL_COLORS[model_name]
        labels = load_judge_labels(cfg)

        # aggregate across all three question types
        agg_flip = {t: [] for t in range(6)}
        agg_hold = {t: [] for t in range(6)}

        for qtype in QTYPES:
            pt = cfg["dir"] / f"{qtype}_multiturn_hidden_states.pt"
            if not pt.exists():
                continue
            hs = torch.load(str(pt), map_location="cpu")
            fn, hn = l2_norm_over_turns(hs, labels)
            for t in range(6):
                agg_flip[t].extend(fn[t])
                agg_hold[t].extend(hn[t])

        turns = list(range(6))
        flip_means = [np.mean(agg_flip[t]) if agg_flip[t] else np.nan for t in turns]
        hold_means = [np.mean(agg_hold[t]) if agg_hold[t] else np.nan for t in turns]
        flip_se   = [np.std(agg_flip[t]) / np.sqrt(len(agg_flip[t])) if agg_flip[t] else np.nan for t in turns]
        hold_se   = [np.std(agg_hold[t]) / np.sqrt(len(agg_hold[t])) if agg_hold[t] else np.nan for t in turns]

        for t in turns:
            for group, val, se in [("flip", flip_means[t], flip_se[t]),
                                   ("hold", hold_means[t], hold_se[t])]:
                rows.append({"model": model_name, "turn": t, "group": group,
                             "mean_l2": val, "se": se,
                             "n": len(agg_flip[t]) if group == "flip" else len(agg_hold[t])})

        t_arr = np.array(turns, dtype=float)

        ax.plot(t_arr, flip_means, color=color, lw=1.8, marker="o",
                markersize=4, label="Flip", zorder=3)
        ax.fill_between(t_arr,
                        [m - s for m, s in zip(flip_means, flip_se)],
                        [m + s for m, s in zip(flip_means, flip_se)],
                        color=color, alpha=0.18, zorder=2)

        ax.plot(t_arr, hold_means, color=color, lw=1.8, marker="s",
                markersize=4, linestyle="--", alpha=0.55, label="Hold", zorder=3)

        ax.set_title(model_name, fontsize=8, fontweight="bold", pad=3)
        ax.set_xlabel("Turn", fontsize=7.5)
        ax.set_xticks(range(6))
        ax.set_xticklabels([f"T{t}" for t in range(6)], fontsize=7)
        ax.tick_params(labelsize=7)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(True, alpha=0.25)

        if ax is axes[0]:
            ax.set_ylabel("Mean L2 norm (all layers)", fontsize=7.5)
            ax.legend(fontsize=7, loc="upper left", handlelength=1.2,
                      borderpad=0.4, labelspacing=0.2)

    fig.suptitle(
        "Activation Magnitude (L2 norm) over Pressure Turns: Flip vs. Hold",
        fontsize=9, y=0.97,
    )
    fig.savefig(str(OUT_PNG), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure saved to {OUT_PNG}")

    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False)
    print(f"Data saved to {OUT_CSV}")
    return df


if __name__ == "__main__":
    df = run()
    # Print summary: is flip L2 norm monotonically increasing pre-cap?
    print("\nFlip group mean L2 norm by turn (averaged across models):")
    summary = df[df["group"] == "flip"].groupby("turn")["mean_l2"].mean()
    print(summary.to_string())
    print("\nHold group mean L2 norm by turn:")
    summary_h = df[df["group"] == "hold"].groupby("turn")["mean_l2"].mean()
    print(summary_h.to_string())
