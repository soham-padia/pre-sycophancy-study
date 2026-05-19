"""
Probe Classifier Sweep — LLM-as-Judge Labels
=============================================
Re-runs the 11-classifier sweep using Claude Haiku 4.5 first-flip labels
instead of keyword labels. Also evaluates cross-model probe transfer:
train probe on model A, test on model B.

Cross-model transfer is possible when both models share the same hidden
dimension (3584: DeepSeek/Qwen2.5; 4096: Llama/Qwen3.5). For cross-family
pairs we use PCA to project to a 256-dim common space.

Outputs:
  analysis_claude/probe_judge_sweep.csv      — per-experiment results
  analysis_claude/probe_judge_transfer.csv   — cross-model transfer results
"""

import os, json, torch, numpy as np, pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings("ignore")

REPO_ROOT = Path(__file__).resolve().parent
OUT_SWEEP    = REPO_ROOT / "analysis_claude" / "probe_judge_sweep.csv"
OUT_TRANSFER = REPO_ROOT / "analysis_claude" / "probe_judge_transfer.csv"

# ── Model config ──────────────────────────────────────────────────────────────
MODELS = {
    "DeepSeek-R1-7B": {
        "dir":        "data/DeepSeek-R1-Distill-Qwen-7B",
        "n_layers":   29,
        "hidden_dim": 3584,
        "best_layer": 19,
        "judge_csv":  REPO_ROOT / "analysis_claude" / "claude_judgements.csv",
        "model_col":  "DeepSeek-R1-Distill-Qwen-7B",
    },
    "Qwen2.5-7B": {
        "dir":        "data/Qwen2.5-7B-Instruct",
        "n_layers":   29,
        "hidden_dim": 3584,
        "best_layer": 17,
        "judge_csv":  REPO_ROOT / "analysis_claude" / "qwen25_judgements_haiku.csv",
        "model_col":  "Qwen2.5-7B-Instruct",
    },
    "Llama-3.1-8B": {
        "dir":        "data/Llama-3.1-8B-Instruct",
        "n_layers":   33,
        "hidden_dim": 4096,
        "best_layer": 9,
        "judge_csv":  REPO_ROOT / "analysis_claude" / "llama31_judgements_haiku.csv",
        "model_col":  "Llama-3.1-8B-Instruct",
    },
    "Qwen3.5-9B": {
        "dir":        "data/Qwen3.5-9B",
        "n_layers":   33,
        "hidden_dim": 4096,
        "best_layer": 10,
        "judge_csv":  REPO_ROOT / "analysis_claude" / "qwen35_judgements_haiku.csv",
        "model_col":  "Qwen3.5-9B",
    },
}

QUESTION_TYPES = ["base", "critical", "presupposition"]

CLASSIFIERS = {
    "LogisticReg":   lambda: LogisticRegression(max_iter=2000, C=0.1, class_weight="balanced", solver="lbfgs", n_jobs=-1),
    "LinearSVM":     lambda: LinearSVC(max_iter=2000, C=0.1, class_weight="balanced"),
    "Ridge":         lambda: RidgeClassifier(alpha=1.0, class_weight="balanced"),
    "MLP_small":     lambda: MLPClassifier(hidden_layer_sizes=(64,),     max_iter=500, early_stopping=True, random_state=42),
    "MLP_medium":    lambda: MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=500, early_stopping=True, random_state=42),
    "RBF_SVM":       lambda: SVC(kernel="rbf", C=1.0, class_weight="balanced", gamma="scale"),
    "RandomForest":  lambda: RandomForestClassifier(n_estimators=100, max_depth=8, class_weight="balanced", random_state=42, n_jobs=-1),
    "KNN_10":        lambda: KNeighborsClassifier(n_neighbors=10, n_jobs=-1),
    "NaiveBayes":    lambda: GaussianNB(),
}

LINEAR_CLFS = {"LogisticReg", "LinearSVM", "Ridge"}


# ── Load judge labels ─────────────────────────────────────────────────────────
def load_judge_labels(model_name, qtype):
    cfg = MODELS[model_name]
    if not cfg["judge_csv"].exists():
        return None
    df = pd.read_csv(cfg["judge_csv"])
    df = df[(df["model"] == cfg["model_col"]) & (df["question_type"] == qtype)].copy()
    df["flip"] = df["judgement"].astype(str).str.lower() == "true"
    # first-flip per question
    labels = {}
    for q, grp in df.groupby("question"):
        grp = grp.sort_values("turn")
        first_flip = grp[grp["flip"]]["turn"].min() if grp["flip"].any() else None
        turn_labels = {}
        for _, row in grp.iterrows():
            t = int(row["turn"])
            if first_flip is not None and t > first_flip:
                continue  # censor post-flip
            turn_labels[t] = int(row["flip"])
        labels[q] = turn_labels
    return labels


# ── Load hidden states ────────────────────────────────────────────────────────
def load_hs(model_name, qtype):
    cfg = MODELS[model_name]
    pt = Path(cfg["dir"]) / f"{qtype}_multiturn_hidden_states.pt"
    if not pt.exists():
        return None
    return torch.load(str(pt), map_location="cpu")


# ── Build feature matrix ──────────────────────────────────────────────────────
def build_preflip_features(hs, judge_labels, layer):
    """Returns X (n_turns, hidden_dim), y (n_turns,), qids (n_turns,)."""
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


def build_everflip_features(hs, judge_labels, layer):
    """One sample per question at turn 0 (baseline), label = ever flipped."""
    X, y, qids = [], [], []
    for qi, (q, turn_labels) in enumerate(judge_labels.items()):
        if q not in hs:
            continue
        ever = int(any(turn_labels.values()))
        tensor = hs[q][0]
        if tensor is None:
            continue
        X.append(tensor.float()[layer].numpy())
        y.append(ever)
        qids.append(qi)
    return np.array(X), np.array(y), np.array(qids)


# ── Cross-validation ──────────────────────────────────────────────────────────
def run_cv(X, y, qids, clf_fn, n_splits=5, pca_dims=None):
    uq  = np.unique(qids)
    ql  = np.array([y[qids == qi].max() for qi in uq])
    if len(np.unique(ql)) < 2:
        return None, None
    n_splits = min(n_splits, min(np.sum(ql == 0), np.sum(ql == 1)))
    if n_splits < 2:
        return None, None
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
        if pca_dims:
            pca = PCA(n_components=min(pca_dims, X_tr.shape[1], X_tr.shape[0] - 1), random_state=42)
            X_tr = pca.fit_transform(X_tr)
            X_te = pca.transform(X_te)
        clf = clf_fn(); clf.fit(X_tr, y_tr)
        all_p.extend(clf.predict(X_te)); all_t.extend(y_te)
    if not all_p:
        return None, None
    acc = accuracy_score(all_t, all_p)
    f1  = f1_score(all_t, all_p, zero_division=0)
    return acc, f1


# ── Cross-model transfer ──────────────────────────────────────────────────────
def run_transfer(X_train, y_train, X_test, y_test, clf_fn, pca_dims=256):
    if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
        return None, None
    if X_train.shape[1] != X_test.shape[1]:
        # Cross-family (different hidden dims): skip — no shared coordinate space
        return None, None
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test  = sc.transform(X_test)
    if pca_dims:
        max_pca = min(X_train.shape[0] - 1, X_train.shape[1])
        n = min(pca_dims, max_pca)
        pca = PCA(n_components=n, random_state=42)
        X_train = pca.fit_transform(X_train)
        X_test  = pca.transform(X_test)
    clf = clf_fn(); clf.fit(X_train, y_train)
    preds = clf.predict(X_test)
    return accuracy_score(y_test, preds), f1_score(y_test, preds, zero_division=0)


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    sweep_rows    = []
    transfer_rows = []

    # ── Per-model classifier sweep ─────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("  CLASSIFIER SWEEP  (LLM-as-Judge labels, first-flip)")
    print("=" * 72)

    model_data = {}  # cache for transfer

    for model_name in MODELS:
        cfg = MODELS[model_name]
        layer = cfg["best_layer"]
        for qtype in QUESTION_TYPES:
            judge_labels = load_judge_labels(model_name, qtype)
            hs            = load_hs(model_name, qtype)
            if judge_labels is None or hs is None:
                continue

            X_pre, y_pre, qids_pre = build_preflip_features(hs, judge_labels, layer)
            X_evr, y_evr, qids_evr = build_everflip_features(hs, judge_labels, layer)

            # cache ever-flip features for transfer
            model_data[(model_name, qtype)] = (X_evr, y_evr)

            n_q   = len(judge_labels)
            n_f   = int(any(v for vd in judge_labels.values() for v in vd.values()))
            chance_pre = max(np.mean(y_pre), 1 - np.mean(y_pre)) if len(y_pre) > 0 else 0
            chance_evr = max(np.mean(y_evr), 1 - np.mean(y_evr)) if len(y_evr) > 0 else 0

            print(f"\n  {model_name} | {qtype} | layer {layer} | {n_q}q")

            for task, X, y, qids, chance in [
                ("pre_flip",  X_pre, y_pre, qids_pre, chance_pre),
                ("ever_flip", X_evr, y_evr, qids_evr, chance_evr),
            ]:
                if len(X) == 0 or len(np.unique(y)) < 2:
                    continue
                print(f"    {task} | chance={chance:.3f}")
                for clf_name, clf_fn in CLASSIFIERS.items():
                    pca = 100 if clf_name in {"KNN_10", "RBF_SVM"} else None
                    acc, f1 = run_cv(X, y, qids, clf_fn, pca_dims=pca)
                    if acc is None:
                        continue
                    delta = acc - chance
                    tag   = "NONLINEAR" if clf_name not in LINEAR_CLFS else "LINEAR"
                    note  = "✓" if delta > 0.03 else ("~" if delta > 0 else "✗")
                    print(f"      {clf_name:<16} acc={acc:.3f} Δ={delta:+.3f} {note}  [{tag}]")
                    sweep_rows.append({
                        "model": model_name, "qtype": qtype, "task": task,
                        "clf": clf_name, "clf_type": tag,
                        "acc": round(acc, 4), "f1_flip": round(f1, 4),
                        "chance": round(chance, 4), "delta": round(delta, 4),
                    })

    # Save sweep results now before transfer (in case transfer errors)
    pd.DataFrame(sweep_rows).to_csv(OUT_SWEEP, index=False)

    # ── Cross-model transfer ───────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("  CROSS-MODEL PROBE TRANSFER  (best nonlinear: RBF SVM)")
    print("  Train on source model, test on target model (ever_flip task)")
    print("=" * 72)

    model_names = list(MODELS.keys())
    for src in model_names:
        for tgt in model_names:
            if src == tgt:
                continue
            for qtype in QUESTION_TYPES:
                if (src, qtype) not in model_data or (tgt, qtype) not in model_data:
                    continue
                X_tr, y_tr = model_data[(src, qtype)]
                X_te, y_te = model_data[(tgt, qtype)]
                if len(X_tr) == 0 or len(X_te) == 0:
                    continue
                if len(np.unique(y_tr)) < 2 or len(np.unique(y_te)) < 2:
                    continue
                chance_te = max(np.mean(y_te), 1 - np.mean(y_te))
                acc, f1 = run_transfer(X_tr, y_tr, X_te, y_te,
                                       lambda: SVC(kernel="rbf", C=1.0,
                                                   class_weight="balanced", gamma="scale"))
                if acc is None:
                    continue
                delta = acc - chance_te
                note  = "✓ TRANSFERS" if delta > 0.03 else ("~ marginal" if delta > 0 else "✗ fails")
                print(f"  {src:<16} → {tgt:<16} | {qtype:<16} | "
                      f"acc={acc:.3f} chance={chance_te:.3f} Δ={delta:+.3f}  {note}")
                transfer_rows.append({
                    "source": src, "target": tgt, "qtype": qtype,
                    "acc": round(acc, 4), "f1": round(f1, 4),
                    "chance": round(chance_te, 4), "delta": round(delta, 4),
                })

    # ── Save ───────────────────────────────────────────────────────────────────
    pd.DataFrame(sweep_rows).to_csv(OUT_SWEEP, index=False)
    pd.DataFrame(transfer_rows).to_csv(OUT_TRANSFER, index=False)
    print(f"\nSaved → {OUT_SWEEP}")
    print(f"Saved → {OUT_TRANSFER}")

    # ── Summary: best nonlinear vs best linear ─────────────────────────────────
    df = pd.DataFrame(sweep_rows)
    print("\n" + "=" * 72)
    print("  SUMMARY: Best nonlinear vs best linear per model/qtype (pre_flip)")
    print("=" * 72)
    pre = df[df["task"] == "pre_flip"]
    for (model, qtype), grp in pre.groupby(["model", "qtype"]):
        lin  = grp[grp["clf_type"] == "LINEAR"]["acc"].max()
        nlin = grp[grp["clf_type"] == "NONLINEAR"]["acc"].max()
        ch   = grp["chance"].iloc[0]
        print(f"  {model:<16} {qtype:<16} linear={lin:.3f} nonlinear={nlin:.3f} "
              f"chance={ch:.3f}  gap={nlin-lin:+.3f}")


if __name__ == "__main__":
    main()
