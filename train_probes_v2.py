"""
Sycophancy Probe Training Pipeline v2
======================================
Labels: Claude Haiku 4.5 LLM-as-judge (first-flip-only methodology)
        Replaces unreliable keyword matching with judge CSV labels.

Models: Llama-3.1-8B (152q) + Qwen3.5-9B (112q) — both 33 layers × 4096 dims

Usage:
    python train_probes_v2.py
"""

import os
import json
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict, Counter
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC, SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.model_selection import StratifiedKFold
import warnings
warnings.filterwarnings("ignore")

REPO_ROOT = Path(__file__).resolve().parent


# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
MODELS = {
    "DeepSeek-R1-7B": {
        "dir": "data/DeepSeek-R1-Distill-Qwen-7B",
        "n_layers": 29,
        "hidden_dim": 3584,
        "probe_layers": [25, 26, 27, 28],
    },
    "Qwen2.5-7B": {
        "dir": "data/Qwen2.5-7B-Instruct",
        "n_layers": 29,
        "hidden_dim": 3584,
        "probe_layers": [15, 16, 17, 18, 19, 25, 26, 27, 28],  # paper found 17-19 best
    },
    "Llama-3.1-8B": {
        "dir": "data/Llama-3.1-8B-Instruct",
        "n_layers": 33,
        "hidden_dim": 4096,
        "probe_layers": [29, 30, 31, 32],
    },
    "Qwen3.5-9B": {
        "dir": "data/Qwen3.5-9B",
        "n_layers": 33,
        "hidden_dim": 4096,
        "probe_layers": [19, 20, 21, 30, 31, 32],
    },
}

# Map probe model name → (judge CSV path, model column value in CSV)
JUDGE_CSVS = {
    "DeepSeek-R1-7B": (REPO_ROOT / "analysis_claude" / "claude_judgements.csv",        "DeepSeek-R1-Distill-Qwen-7B"),
    "Qwen2.5-7B":     (REPO_ROOT / "analysis_claude" / "qwen25_judgements_haiku.csv",  "Qwen2.5-7B-Instruct"),
    "Llama-3.1-8B":   (REPO_ROOT / "analysis_claude" / "llama31_judgements_haiku.csv", "Llama-3.1-8B-Instruct"),
    "Qwen3.5-9B":     (REPO_ROOT / "analysis_claude" / "qwen35_judgements_haiku.csv",  "Qwen3.5-9B"),
}

QUESTION_TYPES = ["base", "critical", "presupposition"]

# Best pre-flip layers by model (highest F1 from base layer sweep)
BEST_LAYERS = {
    "DeepSeek-R1-7B": 19,   # F1=0.392 in sweep; 19/29 = 66%
    "Qwen2.5-7B":     17,   # paper's finding: layers 17-19 most predictive for this model
    "Llama-3.1-8B":   9,    # F1=0.439 in sweep; early-mid signal
    "Qwen3.5-9B":     10,   # F1=0.413 in sweep
}

# 11 classifiers matching the paper's sweep (factories so each CV fold gets a fresh instance)
CLASSIFIERS = {
    "Logistic Regression": lambda: LogisticRegression(max_iter=2000, C=0.1, class_weight="balanced", solver="lbfgs"),
    "Linear SVM":          lambda: LinearSVC(max_iter=2000, C=0.1, class_weight="balanced"),
    "LDA":                 lambda: LinearDiscriminantAnalysis(),
    "RBF SVM":             lambda: SVC(kernel="rbf", C=1.0, class_weight="balanced"),
    "Random Forest":       lambda: RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=42),
    "Extra Trees":         lambda: ExtraTreesClassifier(n_estimators=100, class_weight="balanced", random_state=42),
    "Gradient Boosting":   lambda: GradientBoostingClassifier(n_estimators=100, random_state=42),
    "AdaBoost":            lambda: AdaBoostClassifier(n_estimators=100, random_state=42),
    "KNN (k=10)":          lambda: KNeighborsClassifier(n_neighbors=10),
    "Decision Tree":       lambda: DecisionTreeClassifier(class_weight="balanced", random_state=42),
    "Naive Bayes":         lambda: GaussianNB(),
    "MLP (128+64)":        lambda: MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=1000, early_stopping=True),
}

CLF_TYPE = {
    "Logistic Regression": "linear",
    "Linear SVM":          "linear",
    "LDA":                 "linear",
    "RBF SVM":             "nonlinear",
    "Random Forest":       "ensemble",
    "Extra Trees":         "ensemble",
    "Gradient Boosting":   "ensemble",
    "AdaBoost":            "ensemble",
    "KNN (k=10)":          "non-param",
    "Decision Tree":       "nonlinear",
    "Naive Bayes":         "probab.",
    "MLP (128+64)":        "nonlinear",
}

# ─────────────────────────────────────────────
# DATA LOADING (from Claude Haiku judge CSV)
# ─────────────────────────────────────────────
def load_model_data(model_name, question_type="base"):
    """Load hidden states and flip labels from Claude Haiku judge CSV."""
    cfg = MODELS[model_name]
    model_dir = cfg["dir"]

    pt_path = os.path.join(model_dir, f"{question_type}_multiturn_hidden_states.pt")
    if not os.path.exists(pt_path):
        return None, None, None, None

    judge_csv, model_col_name = JUDGE_CSVS[model_name]
    if not judge_csv.exists():
        print(f"    [{question_type}] judge CSV not found: {judge_csv}")
        return None, None, None, None

    # Load hidden states
    hs = torch.load(pt_path, map_location="cpu")
    questions = list(hs.keys())

    # Load judge labels, filter to this model + question type
    df = pd.read_csv(judge_csv)
    df = df[(df["model"] == model_col_name) & (df["question_type"] == question_type)].copy()
    df["judgement_bool"] = df["judgement"].astype(str).str.lower() == "true"

    # Build flip_data: {question: [label_t1, label_t2, ..., label_t5]}
    flip_data = {}
    for q in questions:
        q_df = df[df["question"] == q].sort_values("turn")
        if len(q_df) == 0:
            continue
        labels = [0] * 5
        for _, row in q_df.iterrows():
            t = int(row["turn"])
            if 1 <= t <= 5:
                labels[t - 1] = 1 if row["judgement_bool"] else 0
        flip_data[q] = labels

    # Stats
    total = len(flip_data)
    if total == 0:
        print(f"    [{question_type}] no matching rows in judge CSV")
        return None, None, None, None

    flipped_qs = sum(1 for v in flip_data.values() if any(v))
    total_flips = sum(sum(v) for v in flip_data.values())

    first_flip_turns = []
    for v in flip_data.values():
        for i, f in enumerate(v):
            if f:
                first_flip_turns.append(i + 1)
                break

    print(f"    [{question_type}] {total} questions, "
          f"{flipped_qs} flipped ({flipped_qs/total*100:.1f}%), "
          f"{total_flips} total flip events"
          f"{f', mean first flip: {np.mean(first_flip_turns):.2f}' if first_flip_turns else ''}")

    return hs, questions, flip_data, cfg


def get_layer_vec(hs, q, turn, layer):
    """Extract hidden state vector."""
    return hs[q][turn].float()[layer].numpy()


def get_engineered_features(hs, q, turn):
    """Architecture-agnostic engineered features."""
    feats = []
    last = hs[q][turn].float()[-1].numpy()

    feats.append(np.linalg.norm(last))
    feats.append(np.var(last))
    feats.append(np.mean(last))
    feats.append(np.max(np.abs(last)))

    if turn > 0:
        last_t0 = hs[q][0].float()[-1].numpy()
        last_prev = hs[q][turn - 1].float()[-1].numpy()

        cos_anchor = np.dot(last_t0, last) / (np.linalg.norm(last_t0) * np.linalg.norm(last) + 1e-10)
        cos_consec = np.dot(last_prev, last) / (np.linalg.norm(last_prev) * np.linalg.norm(last) + 1e-10)
        l2_delta = np.linalg.norm(last) - np.linalg.norm(last_t0)

        feats.extend([cos_anchor, l2_delta, cos_consec])

        # Multi-layer drift
        drifts = []
        for li in [-1, -2, -3]:
            v0 = hs[q][0].float()[li].numpy()
            vt = hs[q][turn].float()[li].numpy()
            c = np.dot(v0, vt) / (np.linalg.norm(v0) * np.linalg.norm(vt) + 1e-10)
            drifts.append(1 - c)
        feats.append(np.mean(drifts))
    else:
        feats.extend([1.0, 0.0, 1.0, 0.0])

    feats.append(turn / 5.0)
    return np.array(feats, dtype=np.float32)


# ─────────────────────────────────────────────
# DATASET BUILDERS
# ─────────────────────────────────────────────
def build_preflip_dataset(hs, questions, flip_data, layer, feat_type="raw",
                          exclude_post_flip=True):
    """Task A: Predict flip at Turn N+1 from Turn N hidden state."""
    X, y, qids = [], [], []
    for qi, q in enumerate(questions):
        if q not in flip_data:
            continue
        flips = flip_data[q]
        first_flip = None
        for i, f in enumerate(flips):
            if f:
                first_flip = i + 1
                break

        for t in range(5):
            if exclude_post_flip and first_flip is not None and t >= first_flip:
                continue

            if feat_type == "raw":
                vec = get_layer_vec(hs, q, t, layer)
            elif feat_type == "engineered":
                vec = get_engineered_features(hs, q, t)
            elif feat_type == "combined":
                vec = np.concatenate([get_layer_vec(hs, q, t, layer),
                                      get_engineered_features(hs, q, t)])
            X.append(vec)
            y.append(flips[t])
            qids.append(qi)

    return np.array(X), np.array(y), np.array(qids)


def build_everflip_dataset(hs, questions, flip_data, layer, feat_type="raw",
                            use_turns=(0, 1)):
    """Task B: Predict if question ever flips, using early turn hidden states."""
    X, y, qids = [], [], []
    for qi, q in enumerate(questions):
        if q not in flip_data:
            continue
        ever_flip = 1 if any(flip_data[q]) else 0

        for t in use_turns:
            if feat_type == "raw":
                vec = get_layer_vec(hs, q, t, layer)
            elif feat_type == "engineered":
                vec = get_engineered_features(hs, q, t)
            elif feat_type == "combined":
                vec = np.concatenate([get_layer_vec(hs, q, t, layer),
                                      get_engineered_features(hs, q, t)])
            X.append(vec)
            y.append(ever_flip)
            qids.append(qi)

    return np.array(X), np.array(y), np.array(qids)


# ─────────────────────────────────────────────
# TRAINING & EVALUATION
# ─────────────────────────────────────────────
def train_eval(X_tr, y_tr, X_te, y_te, model_type="logistic"):
    scaler = StandardScaler()
    Xtr = scaler.fit_transform(X_tr)
    Xte = scaler.transform(X_te)

    if model_type == "logistic":
        clf = LogisticRegression(max_iter=2000, C=0.1, class_weight="balanced", solver="lbfgs")
    elif model_type == "svm":
        clf = LinearSVC(max_iter=2000, C=0.1, class_weight="balanced")
    elif model_type == "mlp":
        clf = MLPClassifier(hidden_layer_sizes=(128,), max_iter=1000, early_stopping=True)

    clf.fit(Xtr, y_tr)
    preds = clf.predict(Xte)
    return preds, accuracy_score(y_te, preds), f1_score(y_te, preds, zero_division=0)


def run_cv(X, y, qids, n_splits=5, model_type="logistic"):
    """Stratified CV grouped by question."""
    uq = np.unique(qids)
    ql = np.array([y[qids == qi].max() for qi in uq])

    if len(np.unique(ql)) < 2:
        return None, None
    actual_splits = min(n_splits, min(np.sum(ql == 0), np.sum(ql == 1)))
    if actual_splits < 2:
        return None, None

    skf = StratifiedKFold(n_splits=actual_splits, shuffle=True, random_state=42)
    all_p, all_t = [], []

    for tr_idx, te_idx in skf.split(uq, ql):
        tr_qs = set(uq[tr_idx])
        te_qs = set(uq[te_idx])
        tr_mask = np.array([qi in tr_qs for qi in qids])
        te_mask = np.array([qi in te_qs for qi in qids])

        if len(np.unique(y[tr_mask])) < 2:
            continue

        p, _, _ = train_eval(X[tr_mask], y[tr_mask], X[te_mask], y[te_mask], model_type)
        all_p.extend(p)
        all_t.extend(y[te_mask])

    if not all_p:
        return None, None
    return accuracy_score(all_t, all_p), f1_score(all_t, all_p, zero_division=0)


def run_cv_any_clf(X, y, qids, clf_factory, n_splits=5):
    """Stratified CV grouped by question using any sklearn classifier factory."""
    uq = np.unique(qids)
    ql = np.array([y[qids == qi].max() for qi in uq])

    if len(np.unique(ql)) < 2:
        return None, None
    actual_splits = min(n_splits, min(np.sum(ql == 0), np.sum(ql == 1)))
    if actual_splits < 2:
        return None, None

    skf = StratifiedKFold(n_splits=actual_splits, shuffle=True, random_state=42)
    all_p, all_t = [], []

    for tr_idx, te_idx in skf.split(uq, ql):
        tr_qs = set(uq[tr_idx])
        te_qs = set(uq[te_idx])
        tr_mask = np.array([qi in tr_qs for qi in qids])
        te_mask = np.array([qi in te_qs for qi in qids])

        if len(np.unique(y[tr_mask])) < 2:
            continue

        scaler = StandardScaler()
        Xtr = scaler.fit_transform(X[tr_mask])
        Xte = scaler.transform(X[te_mask])

        clf = clf_factory()
        clf.fit(Xtr, y[tr_mask])
        preds = clf.predict(Xte)
        all_p.extend(preds)
        all_t.extend(y[te_mask])

    if not all_p:
        return None, None
    return accuracy_score(all_t, all_p), f1_score(all_t, all_p, zero_division=0)


# ─────────────────────────────────────────────
# EXPERIMENTS
# ─────────────────────────────────────────────
def experiment_per_model(all_data):
    print("\n" + "=" * 70)
    print("EXPERIMENT 1: Per-Model Probe (Stratified CV)")
    print("=" * 70)

    results_summary = []

    for model_name in all_data:
        for qtype in all_data[model_name]:
            hs, questions, flip_data, cfg = all_data[model_name][qtype]
            if hs is None:
                continue

            n_flipped = sum(1 for v in flip_data.values() if any(v))
            n_hold = len(flip_data) - n_flipped

            # Skip if too imbalanced for ever_flip
            ever_flip_viable = min(n_flipped, n_hold) >= 3

            print(f"\n  {model_name} | {qtype} | {len(flip_data)}q ({n_flipped} flip, {n_hold} hold)")
            print(f"  {'─' * 60}")

            for layer in cfg["probe_layers"]:
                # Task A: pre-flip
                X, y, qids = build_preflip_dataset(hs, questions, flip_data, layer, "raw")
                if len(X) > 0 and len(np.unique(y)) >= 2:
                    chance = max(np.mean(y), 1 - np.mean(y))
                    acc_lr, f1_lr = run_cv(X, y, qids, model_type="logistic")
                    acc_mlp, f1_mlp = run_cv(X, y, qids, model_type="mlp")

                    if acc_lr is not None:
                        tag = "✓" if acc_lr > chance else "✗"
                        print(f"    L{layer:>2} pre_flip  | LR: {acc_lr:.3f} F1={f1_lr:.3f} "
                              f"MLP: {acc_mlp:.3f} F1={f1_mlp:.3f} "
                              f"(chance={chance:.3f}) {tag} | n={len(y)} ({sum(y)}+/{len(y)-sum(y)}-)")
                        results_summary.append({
                            "model": model_name, "qtype": qtype, "layer": layer,
                            "task": "pre_flip", "acc": acc_lr, "f1": f1_lr,
                            "chance": chance, "n": len(y),
                        })

                # Task B: ever-flip (only if viable)
                if ever_flip_viable:
                    X, y, qids = build_everflip_dataset(hs, questions, flip_data, layer, "raw", use_turns=(0, 1))
                    if len(X) > 0 and len(np.unique(y)) >= 2:
                        chance = max(np.mean(y), 1 - np.mean(y))
                        acc_lr, f1_lr = run_cv(X, y, qids, model_type="logistic")
                        acc_mlp, f1_mlp = run_cv(X, y, qids, model_type="mlp")

                        if acc_lr is not None:
                            tag = "✓" if acc_lr > chance else "✗"
                            print(f"    L{layer:>2} ever_flip | LR: {acc_lr:.3f} F1={f1_lr:.3f} "
                                  f"MLP: {acc_mlp:.3f} F1={f1_mlp:.3f} "
                                  f"(chance={chance:.3f}) {tag} | n={len(y)} ({sum(y)}+/{len(y)-sum(y)}-)")
                            results_summary.append({
                                "model": model_name, "qtype": qtype, "layer": layer,
                                "task": "ever_flip", "acc": acc_lr, "f1": f1_lr,
                                "chance": chance, "n": len(y),
                            })

    return results_summary


def experiment_cross_model(all_data):
    print("\n" + "=" * 70)
    print("EXPERIMENT 2: Cross-Model Probe Transfer")
    print("=" * 70)

    model_names = list(all_data.keys())
    results = []

    for qtype in QUESTION_TYPES:
        # Check both models have this qtype
        available = [m for m in model_names if qtype in all_data[m] and all_data[m][qtype][0] is not None]
        if len(available) < 2:
            continue

        print(f"\n  Question type: {qtype}")

        for train_m in available:
            for test_m in available:
                if train_m == test_m:
                    continue

                hs_tr, q_tr, fd_tr, cfg_tr = all_data[train_m][qtype]
                hs_te, q_te, fd_te, cfg_te = all_data[test_m][qtype]

                # Skip pairs with incompatible hidden dims (can't transfer raw vectors)
                if cfg_tr["hidden_dim"] != cfg_te["hidden_dim"]:
                    print(f"    {train_m:>15} → {test_m:<15} | [skip] hidden_dim mismatch "
                          f"({cfg_tr['hidden_dim']} vs {cfg_te['hidden_dim']})")
                    continue

                # Use penultimate layer of the source model
                layer_tr = cfg_tr["n_layers"] - 2
                layer_te = cfg_te["n_layers"] - 2

                for task_name, builder in [("pre_flip", build_preflip_dataset),
                                            ("ever_flip", build_everflip_dataset)]:
                    if task_name == "ever_flip":
                        X_tr, y_tr, _ = build_everflip_dataset(hs_tr, q_tr, fd_tr, layer_tr, "raw", (0, 1))
                        X_te, y_te, _ = build_everflip_dataset(hs_te, q_te, fd_te, layer_te, "raw", (0, 1))
                    else:
                        X_tr, y_tr, _ = build_preflip_dataset(hs_tr, q_tr, fd_tr, layer_tr, "raw")
                        X_te, y_te, _ = build_preflip_dataset(hs_te, q_te, fd_te, layer_te, "raw")

                    if (len(X_tr) == 0 or len(X_te) == 0 or
                            len(np.unique(y_tr)) < 2 or len(np.unique(y_te)) < 2):
                        continue

                    chance = max(np.mean(y_te), 1 - np.mean(y_te))
                    preds, acc, f1 = train_eval(X_tr, y_tr, X_te, y_te, "logistic")

                    # Check if probe is just predicting majority class
                    pred_dist = Counter(preds)
                    majority_only = len(pred_dist) == 1

                    tag = "✓ REAL" if (acc > chance and not majority_only) else ("⚠ MAJORITY" if majority_only else "✗")
                    print(f"    {train_m:>15} → {test_m:<15} | {task_name:>10} | "
                          f"Acc={acc:.3f} F1={f1:.3f} (chance={chance:.3f}) {tag} "
                          f"[L{layer_tr}→L{layer_te}]")

                    if not majority_only and acc > 0.60:
                        print(f"    {'':>15}   {'':>15}   {classification_report(y_te, preds, target_names=['Hold','Flip'], zero_division=0)}")

                    results.append({
                        "train": train_m, "test": test_m, "qtype": qtype,
                        "task": task_name, "acc": acc, "f1": f1,
                        "chance": chance, "majority_only": majority_only,
                    })

    return results


def experiment_layer_sweep(all_data, qtype="base"):
    print("\n" + "=" * 70)
    print(f"EXPERIMENT 3: Layer Sweep ({qtype})")
    print("=" * 70)

    for model_name in all_data:
        if qtype not in all_data[model_name] or all_data[model_name][qtype][0] is None:
            continue

        hs, questions, flip_data, cfg = all_data[model_name][qtype]
        print(f"\n  {model_name} ({qtype})")
        print(f"  {'Layer':<7} {'PreFlip Acc':>12} {'F1':>6} {'EverFlip Acc':>13} {'F1':>6}")
        print(f"  {'─'*7} {'─'*12} {'─'*6} {'─'*13} {'─'*6}")

        best_layer, best_acc = -1, 0
        n_flipped = sum(1 for v in flip_data.values() if any(v))
        n_hold = len(flip_data) - n_flipped
        ever_flip_viable = min(n_flipped, n_hold) >= 3

        for layer in range(cfg["n_layers"]):
            # Pre-flip
            X, y, qids = build_preflip_dataset(hs, questions, flip_data, layer, "raw")
            pf_acc, pf_f1 = (None, None)
            if len(X) > 0 and len(np.unique(y)) >= 2:
                pf_acc, pf_f1 = run_cv(X, y, qids, model_type="logistic")

            # Ever-flip
            ef_acc, ef_f1 = (None, None)
            if ever_flip_viable:
                X2, y2, qids2 = build_everflip_dataset(hs, questions, flip_data, layer, "raw", (0, 1))
                if len(X2) > 0 and len(np.unique(y2)) >= 2:
                    ef_acc, ef_f1 = run_cv(X2, y2, qids2, model_type="logistic")

            pf_s = f"{pf_acc:.3f}" if pf_acc is not None else "  N/A"
            pf_f = f"{pf_f1:.3f}" if pf_f1 is not None else " N/A"
            ef_s = f"{ef_acc:.3f}" if ef_acc is not None else "   N/A"
            ef_f = f"{ef_f1:.3f}" if ef_f1 is not None else " N/A"

            # Track best (ignoring degenerate F1=0 cases)
            if pf_acc is not None and pf_f1 is not None and pf_f1 > 0 and pf_acc > best_acc:
                best_acc = pf_acc
                best_layer = layer

            print(f"  {layer:<7} {pf_s:>12} {pf_f:>6} {ef_s:>13} {ef_f:>6}")

        print(f"\n  Best pre_flip layer: {best_layer} (Acc={best_acc:.3f})")


def experiment_classifier_sweep(all_data):
    """Sweep all 12 classifiers on pre-flip task at best layer per model.
    Mirrors the paper's 11-classifier sweep that found 75.5% with KNN/RF/SVM."""
    print("\n" + "=" * 70)
    print("EXPERIMENT 4: Full Classifier Sweep (pre-flip, best layer per model)")
    print("=" * 70)
    print("  Replicating paper's nonlinear probe sweep with LLM-judge labels.\n")

    all_results = {}

    for model_name in all_data:
        layer = BEST_LAYERS.get(model_name, 17)

        for qtype in QUESTION_TYPES:
            if qtype not in all_data[model_name] or all_data[model_name][qtype][0] is None:
                continue

            hs, questions, flip_data, cfg = all_data[model_name][qtype]
            X, y, qids = build_preflip_dataset(hs, questions, flip_data, layer, "raw",
                                               exclude_post_flip=True)

            if len(X) == 0 or len(np.unique(y)) < 2:
                continue

            chance = max(np.mean(y), 1 - np.mean(y))
            n_flip = int(y.sum())
            print(f"\n  {model_name} | {qtype} | Layer {layer} | "
                  f"n={len(X)} ({n_flip}+ / {len(y)-n_flip}-) | chance={chance:.3f}")
            print(f"  {'Classifier':<22} {'Type':<11} {'Acc':>7} {'Δchance':>8} {'F1':>7}")
            print(f"  {'─'*22} {'─'*11} {'─'*7} {'─'*8} {'─'*7}")

            rows = []
            for clf_name, clf_factory in CLASSIFIERS.items():
                acc, f1 = run_cv_any_clf(X, y, qids, clf_factory)
                if acc is None:
                    continue
                delta = acc - chance
                flag = " ✓" if delta > 0 else ""
                print(f"  {clf_name:<22} {CLF_TYPE[clf_name]:<11} {acc:>7.3f} {delta:>+8.3f} {f1:>7.3f}{flag}")
                rows.append({"clf": clf_name, "type": CLF_TYPE[clf_name],
                              "acc": acc, "f1": f1, "delta": delta})

            if rows:
                best = max(rows, key=lambda r: r["acc"])
                best_nl = max((r for r in rows if r["type"] != "linear"), key=lambda r: r["acc"])
                print(f"\n  → Best overall : {best['clf']} Acc={best['acc']:.3f} "
                      f"({best['delta']:+.3f} vs chance)  F1={best['f1']:.3f}")
                print(f"  → Best nonlinear: {best_nl['clf']} Acc={best_nl['acc']:.3f} "
                      f"({best_nl['delta']:+.3f} vs chance)  F1={best_nl['f1']:.3f}")
            all_results[(model_name, qtype)] = rows

    # Cross-experiment summary
    print("\n" + "=" * 70)
    print("CLASSIFIER SWEEP SUMMARY — results beating chance baseline")
    print("=" * 70)
    print(f"  {'Model':<18} {'QType':<16} {'Classifier':<22} {'Acc':>7} {'Δchance':>8} {'F1':>7}")
    print(f"  {'─'*18} {'─'*16} {'─'*22} {'─'*7} {'─'*8} {'─'*7}")
    for (model_name, qtype), rows in all_results.items():
        for r in rows:
            if r["delta"] > 0:
                print(f"  {model_name:<18} {qtype:<16} {r['clf']:<22} "
                      f"{r['acc']:>7.3f} {r['delta']:>+8.3f} {r['f1']:>7.3f}")

    return all_results


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    print("╔" + "═" * 68 + "╗")
    print("║  SYCON-Bench Probe Training v2                                     ║")
    print("║  Models: DeepSeek + Qwen2.5-7B + Llama-3.1-8B + Qwen3.5-9B         ║")
    print("║  Labels: Claude Haiku 4.5 judge CSV (first-flip-only)              ║")
    print("║  Question types: base, critical, presupposition                    ║")
    print("╚" + "═" * 68 + "╝")

    # Load all data
    print("\n" + "=" * 70)
    print("LOADING DATA")
    print("=" * 70)

    all_data = {}
    for model_name in MODELS:
        all_data[model_name] = {}
        print(f"\n  {model_name}:")
        for qtype in QUESTION_TYPES:
            result = load_model_data(model_name, qtype)
            all_data[model_name][qtype] = result

    # Run experiments
    per_model_results = experiment_per_model(all_data)
    cross_model_results = experiment_cross_model(all_data)
    experiment_layer_sweep(all_data, "base")
    experiment_classifier_sweep(all_data)

    # Print final summary
    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)

    print("\n  Per-model results beating chance baseline:")
    for r in per_model_results:
        if r["acc"] > r["chance"]:
            print(f"    {r['model']:>15} | {r['qtype']:>15} | L{r['layer']:>2} | "
                  f"{r['task']:>10} | Acc={r['acc']:.3f} > chance={r['chance']:.3f} | F1={r['f1']:.3f}")

    print("\n  Cross-model results (non-majority-class):")
    for r in cross_model_results:
        if not r["majority_only"] and r["acc"] > 0.55:
            print(f"    {r['train']:>15} → {r['test']:<15} | {r['qtype']:>15} | "
                  f"{r['task']:>10} | Acc={r['acc']:.3f} F1={r['f1']:.3f}")

    print("""
    KEY FINDINGS TO REPORT:
    1. Per-model pre-flip: Does LR beat chance? Which layers? Which qtypes?
    2. Per-model ever-flip: Only meaningful for Llama (balanced classes)
    3. Cross-model: Any genuine transfer (not majority-class prediction)?
    4. Layer sweep: Where does accuracy peak? Consistent with H1/H2 findings?
    5. Question type effect: Are presupposition/critical easier or harder?
    """)


if __name__ == "__main__":
    main()
