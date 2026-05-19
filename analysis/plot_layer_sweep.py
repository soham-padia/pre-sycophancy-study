#!/usr/bin/env python3
"""Layer-wise probe accuracy curve.

Sweeps all layers for each model using logistic regression on pre-flip hidden
states with LLM-as-judge first-flip labels. Saves per-layer accuracy CSV and
plots one line per model.
"""
from pathlib import Path
import torch, numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
import warnings
warnings.filterwarnings("ignore")

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_CSV = REPO_ROOT / "analysis_claude" / "layer_sweep_accuracy.csv"
OUT_PNG = REPO_ROOT / "analysis_claude" / "layer_sweep.png"

MODELS = {
    "DeepSeek-R1-7B": {
        "dir":       "data/DeepSeek-R1-Distill-Qwen-7B",
        "n_layers":  29,
        "judge_csv": REPO_ROOT / "analysis_claude" / "claude_judgements.csv",
        "model_col": "DeepSeek-R1-Distill-Qwen-7B",
        "qtypes":    ["base", "critical", "presupposition"],
    },
    "Qwen2.5-7B": {
        "dir":       "data/Qwen2.5-7B-Instruct",
        "n_layers":  29,
        "judge_csv": REPO_ROOT / "analysis_claude" / "qwen25_judgements_haiku.csv",
        "model_col": "Qwen2.5-7B-Instruct",
        "qtypes":    ["base", "critical", "presupposition"],
    },
    "Llama-3.1-8B": {
        "dir":       "data/Llama-3.1-8B-Instruct",
        "n_layers":  33,
        "judge_csv": REPO_ROOT / "analysis_claude" / "llama31_judgements_haiku.csv",
        "model_col": "Llama-3.1-8B-Instruct",
        "qtypes":    ["base", "critical", "presupposition"],
    },
    "Qwen3.5-9B": {
        "dir":       "data/Qwen3.5-9B",
        "n_layers":  33,
        "judge_csv": REPO_ROOT / "analysis_claude" / "qwen35_judgements_haiku.csv",
        "model_col": "Qwen3.5-9B",
        "qtypes":    ["base", "critical", "presupposition"],
    },
}

MODEL_COLORS = {
    "DeepSeek-R1-7B": "#4878CF",
    "Qwen2.5-7B":     "#B47CC7",
    "Llama-3.1-8B":   "#6ACC65",
    "Qwen3.5-9B":     "#C4AD66",
}

BEST_QTYPE = {
    "DeepSeek-R1-7B": "critical",
    "Qwen2.5-7B":     "presupposition",
    "Llama-3.1-8B":   "presupposition",
    "Qwen3.5-9B":     "presupposition",
}


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
    X, y, qids = [], [], []
    for qi, (q, turn_labels) in enumerate(judge_labels.items()):
        if q not in hs:
            continue
        for t, label in turn_labels.items():
            tensor = hs[q][t]
            if tensor is None:
                continue
            X.append(tensor.float()[layer].numpy())
            y.append(label)
            qids.append(qi)
    return np.array(X), np.array(y), np.array(qids)


def run_cv(X, y, qids, n_splits=5):
    uq = np.unique(qids)
    ql = np.array([y[qids == qi].max() for qi in uq])
    if len(np.unique(ql)) < 2:
        return None
    n_splits = min(n_splits, min(np.sum(ql == 0), np.sum(ql == 1)))
    if n_splits < 2:
        return None
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    all_p, all_t = [], []
    for tr_idx, te_idx in skf.split(uq, ql):
        tr_qs = set(uq[tr_idx]); te_qs = set(uq[te_idx])
        tr_m = np.array([qi in tr_qs for qi in qids])
        te_m = np.array([qi in te_qs for qi in qids])
        X_tr, y_tr = X[tr_m], y[tr_m]
        X_te, y_te = X[te_m], y[te_m]
        if len(np.unique(y_tr)) < 2:
            continue
        sc = StandardScaler()
        X_tr = sc.fit_transform(X_tr)
        X_te = sc.transform(X_te)
        clf = LogisticRegression(max_iter=500, C=0.1, class_weight="balanced",
                                 solver="lbfgs", n_jobs=-1)
        clf.fit(X_tr, y_tr)
        all_p.extend(clf.predict(X_te))
        all_t.extend(y_te)
    if not all_p:
        return None
    return accuracy_score(all_t, all_p)


def main():
    rows = []
    print("Running layer sweep (logistic regression, pre-flip task)...")

    for model_name, cfg in MODELS.items():
        qtype = BEST_QTYPE[model_name]
        judge_labels = load_judge_labels(cfg, qtype)
        hs = load_hs(cfg, qtype)
        if judge_labels is None or hs is None:
            print(f"  {model_name}: missing data, skipping")
            continue

        # chance for this dataset
        all_y = [label for tl in judge_labels.values() for label in tl.values()]
        chance = max(np.mean(all_y), 1 - np.mean(all_y))

        print(f"  {model_name} | {qtype} | {cfg['n_layers']} layers | chance={chance:.3f}")
        for layer in range(cfg["n_layers"]):
            X, y, qids = build_features(hs, judge_labels, layer)
            if len(X) == 0 or len(np.unique(y)) < 2:
                continue
            acc = run_cv(X, y, qids)
            if acc is None:
                continue
            rows.append({"model": model_name, "qtype": qtype, "layer": layer,
                         "acc": round(acc, 4), "chance": round(chance, 4)})
            if layer % 5 == 0:
                print(f"    layer {layer:2d}: acc={acc:.3f}")

    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False)
    print(f"\nSaved → {OUT_CSV}")

    # Plot
    fig, ax = plt.subplots(figsize=(8, 4.5))
    fig.subplots_adjust(left=0.10, right=0.97, top=0.88, bottom=0.12)

    for model_name in MODELS:
        sub = df[df["model"] == model_name].sort_values("layer")
        if sub.empty:
            continue
        color = MODEL_COLORS[model_name]
        qtype = BEST_QTYPE[model_name]
        ax.plot(sub["layer"], sub["acc"], color=color, lw=1.8,
                label=f"{model_name} ({qtype})")
        peak = sub.loc[sub["acc"].idxmax()]
        ax.scatter(peak["layer"], peak["acc"], color=color, s=60, zorder=4)
        # chance line (per model, just use first value)
        chance = sub["chance"].iloc[0]

    # Draw chance lines per model (use a single grey dashed line at ~mean chance)
    mean_chance = df.groupby("model")["chance"].first().mean()
    ax.axhline(mean_chance, color="#999999", lw=1.2, ls="--",
               label=f"Mean chance ({mean_chance:.2f})")

    ax.set_xlabel("Layer", fontsize=10)
    ax.set_ylabel("Cross-validation accuracy", fontsize=10)
    ax.set_title(
        "Layer-wise Logistic Regression Accuracy on Pre-Flip Hidden States\n"
        "(best question type per model, LLM-as-judge labels)",
        fontsize=10, pad=8)
    ax.legend(fontsize=8, loc="upper left", framealpha=0.88)
    ax.set_ylim(0.45, 0.95)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.yaxis.grid(True, linestyle="--", alpha=0.3, zorder=0)
    ax.set_axisbelow(True)

    plt.savefig(OUT_PNG, dpi=180, bbox_inches="tight")
    print(f"Saved → {OUT_PNG}")


if __name__ == "__main__":
    main()
