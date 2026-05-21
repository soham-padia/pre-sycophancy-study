#!/usr/bin/env python3
"""PCA visualization of pre-flip vs hold hidden states.

For each model at its best probe layer, projects hidden states to 2D via PCA
and plots pre-flip (will flip) vs hold (will not flip) states side by side.
"""
from pathlib import Path
import torch, numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings("ignore")

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_PNG = REPO_ROOT / "analysis_claude" / "preflip_pca.png"

MODELS = {
    "DeepSeek-R1-7B": {
        "dir":        "data/DeepSeek-R1-Distill-Qwen-7B",
        "best_layer": 19,
        "best_qtype": "critical",
        "judge_csv":  REPO_ROOT / "analysis_claude" / "claude_judgements.csv",
        "model_col":  "DeepSeek-R1-Distill-Qwen-7B",
    },
    "Qwen2.5-7B": {
        "dir":        "data/Qwen2.5-7B-Instruct",
        "best_layer": 17,
        "best_qtype": "presupposition",
        "judge_csv":  REPO_ROOT / "analysis_claude" / "qwen25_judgements_haiku.csv",
        "model_col":  "Qwen2.5-7B-Instruct",
    },
    "Llama-3.1-8B": {
        "dir":        "data/Llama-3.1-8B-Instruct",
        "best_layer": 9,
        "best_qtype": "presupposition",
        "judge_csv":  REPO_ROOT / "analysis_claude" / "llama31_judgements_haiku.csv",
        "model_col":  "Llama-3.1-8B-Instruct",
    },
    "Qwen3.5-9B": {
        "dir":        "data/Qwen3.5-9B",
        "best_layer": 10,
        "best_qtype": "presupposition",
        "judge_csv":  REPO_ROOT / "analysis_claude" / "qwen35_judgements_haiku.csv",
        "model_col":  "Qwen3.5-9B",
    },
}

COLORS = {"flip": "#D65F5F", "hold": "#4878CF"}


def load_judge_labels(cfg, qtype):
    if not cfg["judge_csv"].exists():
        return None
    df = pd.read_csv(cfg["judge_csv"])
    df = df[(df["model"] == cfg["model_col"]) & (df["question_type"] == qtype)].copy()
    df["flip"] = df["judgement"].astype(str).str.lower() == "true"
    labels = {}
    for q, grp in df.groupby("question"):
        grp = grp.sort_values("turn")
        first_flip = grp[grp["flip"]]["turn"].min() if grp["flip"].any() else None
        turn_labels = {}
        for _, row in grp.iterrows():
            t = int(row["turn"])
            if first_flip is not None and t > first_flip:
                continue
            turn_labels[t] = int(row["flip"])
        labels[q] = turn_labels
    return labels


def load_hs(cfg, qtype):
    pt = Path(cfg["dir"]) / f"{qtype}_multiturn_hidden_states.pt"
    if not pt.exists():
        return None
    return torch.load(str(pt), map_location="cpu")


def build_features(hs, judge_labels, layer):
    X, y = [], []
    for q, turn_labels in judge_labels.items():
        if q not in hs:
            continue
        for t, label in turn_labels.items():
            tensor = hs[q][t]
            if tensor is None:
                continue
            X.append(tensor.float()[layer].numpy())
            y.append(label)
    return np.array(X), np.array(y)


def main():
    model_names = list(MODELS.keys())
    n = len(model_names)

    fig = plt.figure(figsize=(13, 10))
    gs = gridspec.GridSpec(2, 2, hspace=0.45, wspace=0.35,
                           left=0.07, right=0.97, top=0.90, bottom=0.07)

    for idx, model_name in enumerate(model_names):
        cfg = MODELS[model_name]
        qtype = cfg["best_qtype"]
        layer = cfg["best_layer"]
        ax = fig.add_subplot(gs[idx // 2, idx % 2])

        judge_labels = load_judge_labels(cfg, qtype)
        hs = load_hs(cfg, qtype)
        if judge_labels is None or hs is None:
            ax.set_title(f"{model_name}\n(data unavailable)")
            continue

        X, y = build_features(hs, judge_labels, layer)
        if len(X) == 0 or len(np.unique(y)) < 2:
            ax.set_title(f"{model_name}\n(insufficient data)")
            continue

        sc = StandardScaler()
        X_sc = sc.fit_transform(X)
        pca = PCA(n_components=2, random_state=42)
        X_2d = pca.fit_transform(X_sc)
        var = pca.explained_variance_ratio_

        flip_mask = y == 1
        hold_mask = y == 0

        ax.scatter(X_2d[hold_mask, 0], X_2d[hold_mask, 1],
                   c=COLORS["hold"], alpha=0.35, s=12, label="Hold",
                   edgecolors="none", zorder=2)
        ax.scatter(X_2d[flip_mask, 0], X_2d[flip_mask, 1],
                   c=COLORS["flip"], alpha=0.45, s=12, label="Pre-flip",
                   edgecolors="none", zorder=3)

        ax.set_title(
            f"{model_name}  (layer {layer}, {qtype})\n"
            f"PC1={var[0]:.1%}, PC2={var[1]:.1%}  "
            f"n_flip={flip_mask.sum()}, n_hold={hold_mask.sum()}",
            fontsize=10.5)
        ax.set_xlabel("PC 1", fontsize=10)
        ax.set_ylabel("PC 2", fontsize=10)
        ax.tick_params(labelsize=9)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        if idx == 0:
            ax.legend(fontsize=9.5, loc="upper right", framealpha=0.85,
                      markerscale=1.5)

    fig.suptitle(
        "PCA Projection of Pre-Flip and Hold Hidden States\n"
        "(best probe layer per model, LLM-as-judge first-flip labels)",
        fontsize=13, y=0.97)

    plt.savefig(OUT_PNG, dpi=180, bbox_inches="tight")
    print(f"Saved → {OUT_PNG}")


if __name__ == "__main__":
    main()
