#!/usr/bin/env python3
"""Geometric analysis of pre-flip vs hold hidden states.

Tests whether pre-flip and hold states form non-linearly separable clusters —
i.e., that sycophancy is encoded in curved representational geometry rather than
linearly separable directions (motivation for why nonlinear probes outperform linear).

Metrics computed (no probe training):
  - PCA explained variance (first 5 components)
  - Silhouette score (how well-clustered pre-flip vs hold are in PCA space)
  - LDA accuracy (linear separability upper-bound)
  - Between-class / within-class distance ratio (Fisher criterion proxy)

Also runs t-SNE (2D) and saves coordinates for plotting if matplotlib available.

Outputs:
  analysis_claude/preflip_geometry.txt   — report
  analysis_claude/preflip_geometry.csv   — per-model/layer metrics
  analysis_claude/tsne_coords/           — t-SNE coordinates per model (for plotting)

Usage:
    python analysis/preflip_geometry.py
"""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import silhouette_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_TXT  = REPO_ROOT / "analysis_claude" / "preflip_geometry.txt"
OUTPUT_CSV  = REPO_ROOT / "analysis_claude" / "preflip_geometry.csv"
TSNE_DIR    = REPO_ROOT / "analysis_claude" / "tsne_coords"

MODELS = {
    "DeepSeek-R1-7B": {"dir": "data/DeepSeek-R1-Distill-Qwen-7B", "n_layers": 29, "best_layer": 19},
    "Qwen2.5-7B":     {"dir": "data/Qwen2.5-7B-Instruct",          "n_layers": 29, "best_layer": 17},
    "Llama-3.1-8B":   {"dir": "data/Llama-3.1-8B-Instruct",        "n_layers": 33, "best_layer": 9},
    "Qwen3.5-9B":     {"dir": "data/Qwen3.5-9B",                   "n_layers": 33, "best_layer": 10},
}

JUDGE_CSVS = {
    "DeepSeek-R1-7B": (REPO_ROOT / "analysis_claude" / "claude_judgements.csv",        "DeepSeek-R1-Distill-Qwen-7B"),
    "Qwen2.5-7B":     (REPO_ROOT / "analysis_claude" / "qwen25_judgements_haiku.csv",  "Qwen2.5-7B-Instruct"),
    "Llama-3.1-8B":   (REPO_ROOT / "analysis_claude" / "llama31_judgements_haiku.csv", "Llama-3.1-8B-Instruct"),
    "Qwen3.5-9B":     (REPO_ROOT / "analysis_claude" / "qwen35_judgements_haiku.csv",  "Qwen3.5-9B"),
}

QUESTION_TYPES = ["base", "critical", "presupposition"]


# ─────────────────────────────────────────────────────────────────────────────
# Data loading (mirrors train_probes_v2.py)
# ─────────────────────────────────────────────────────────────────────────────

def load_data(model_name: str, qtype: str):
    cfg = MODELS[model_name]
    pt_path = REPO_ROOT / cfg["dir"] / f"{qtype}_multiturn_hidden_states.pt"
    if not pt_path.exists():
        return None, None, None

    judge_csv, model_col = JUDGE_CSVS[model_name]
    if not judge_csv.exists():
        return None, None, None

    hs = torch.load(str(pt_path), map_location="cpu")
    questions = list(hs.keys())

    df = pd.read_csv(judge_csv)
    df = df[(df["model"] == model_col) & (df["question_type"] == qtype)].copy()
    df["judgement_bool"] = df["judgement"].astype(str).str.lower() == "true"

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

    return hs, questions, flip_data


# ─────────────────────────────────────────────────────────────────────────────
# Build pre-flip dataset (mirrors train_probes_v2.py build_preflip_dataset)
# ─────────────────────────────────────────────────────────────────────────────

def build_preflip_dataset(hs, flip_data: dict, layer: int):
    """
    For each question, for each turn t (0..4):
      label = 1 if flip_data[q][t] == 1, else 0
      feature = hs[q][t][layer]
    Only includes turns up to and including first flip (exclude_post_flip=True).
    Returns X (n, hidden_dim), y (n,), questions (n,).
    """
    X, y, qs = [], [], []
    for q, labels in flip_data.items():
        first_flip = next((i for i, v in enumerate(labels) if v), None)
        for t, label in enumerate(labels):
            if first_flip is not None and t > first_flip:
                break
            try:
                vec = hs[q][t].float()[layer].numpy()
            except (IndexError, KeyError, TypeError):
                continue
            X.append(vec)
            y.append(label)
            qs.append(q)

    if not X:
        return None, None, None
    return np.array(X), np.array(y), np.array(qs)


# ─────────────────────────────────────────────────────────────────────────────
# Geometric metrics
# ─────────────────────────────────────────────────────────────────────────────

def pca_variance(X: np.ndarray, n_components: int = 10) -> np.ndarray:
    pca = PCA(n_components=min(n_components, X.shape[1], X.shape[0] - 1))
    pca.fit(X)
    return pca.explained_variance_ratio_


def silhouette(X: np.ndarray, y: np.ndarray, n_pca: int = 50) -> float:
    if len(np.unique(y)) < 2 or len(X) < 10:
        return np.nan
    pca = PCA(n_components=min(n_pca, X.shape[1], X.shape[0] - 1))
    Xr = pca.fit_transform(X)
    try:
        return float(silhouette_score(Xr, y, sample_size=min(1000, len(y)), random_state=42))
    except Exception:
        return np.nan


def lda_accuracy(X: np.ndarray, y: np.ndarray, n_splits: int = 5) -> float:
    """Cross-validated LDA accuracy (proxy for linear separability)."""
    if len(np.unique(y)) < 2 or len(X) < 20:
        return np.nan
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    # Reduce to 50 dims first for numerical stability
    n_pca = min(50, X.shape[1], X.shape[0] - 1)
    pca = PCA(n_components=n_pca)
    X = pca.fit_transform(X)

    skf = StratifiedKFold(n_splits=min(n_splits, min(np.bincount(y.astype(int)))),
                          shuffle=True, random_state=42)
    preds, truths = [], []
    for tr, te in skf.split(X, y):
        clf = LinearDiscriminantAnalysis()
        clf.fit(X[tr], y[tr])
        preds.extend(clf.predict(X[te]))
        truths.extend(y[te])
    from sklearn.metrics import accuracy_score
    return float(accuracy_score(truths, preds))


def fisher_ratio(X: np.ndarray, y: np.ndarray) -> float:
    """Between-class / within-class variance ratio (Fisher criterion, averaged over dims)."""
    if len(np.unique(y)) < 2:
        return np.nan
    mu = X.mean(axis=0)
    classes = np.unique(y)
    sb = np.zeros(X.shape[1])
    sw = np.zeros(X.shape[1])
    for c in classes:
        Xc = X[y == c]
        mc = Xc.mean(axis=0)
        sb += len(Xc) * (mc - mu) ** 2
        sw += ((Xc - mc) ** 2).sum(axis=0)
    sw = np.where(sw < 1e-10, 1e-10, sw)
    return float(np.mean(sb / sw))


def tsne_coords(X: np.ndarray, y: np.ndarray, n_pca: int = 50):
    """Run PCA → t-SNE and return 2D coordinates."""
    try:
        from sklearn.manifold import TSNE
        pca = PCA(n_components=min(n_pca, X.shape[1], X.shape[0] - 1))
        Xr = pca.fit_transform(StandardScaler().fit_transform(X))
        tsne = TSNE(n_components=2, perplexity=min(30, len(X) // 4),
                    random_state=42, n_iter=1000)
        return tsne.fit_transform(Xr), y
    except Exception:
        return None, None


# ─────────────────────────────────────────────────────────────────────────────
# Report
# ─────────────────────────────────────────────────────────────────────────────

def section(title: str) -> str:
    return f"\n{'=' * 70}\n{title}\n{'=' * 70}"


def run_analysis() -> tuple[str, list[dict]]:
    lines = [
        "PRE-FLIP STATE GEOMETRY ANALYSIS",
        "Geometric structure of pre-flip vs hold hidden states",
        "",
        "Motivation: nonlinear classifiers consistently outperform linear probes.",
        "This analysis tests whether the geometry of pre-flip representations",
        "explains why — i.e., whether flip/hold states are linearly vs non-linearly",
        "separable in the representation space.",
        "",
        "Metrics:",
        "  Silhouette  : cluster cohesion in PCA-50 space (higher = more separated)",
        "  LDA acc     : cross-val accuracy with linear classifier (linear separability)",
        "  Fisher ratio: between/within class variance (linear discrimination power)",
        "  PCA var[0]  : variance explained by first PC (low = distributed encoding)",
    ]
    rows = []
    TSNE_DIR.mkdir(parents=True, exist_ok=True)

    for model_name, cfg in MODELS.items():
        lines.append(section(f"MODEL: {model_name}"))
        best_layer = cfg["best_layer"]

        for qtype in QUESTION_TYPES:
            hs, _, flip_data = load_data(model_name, qtype)
            if hs is None:
                lines.append(f"\n  [{qtype}] data not available")
                continue

            X, y, _ = build_preflip_dataset(hs, flip_data, best_layer)
            if X is None or len(np.unique(y)) < 2:
                lines.append(f"\n  [{qtype}] insufficient data")
                continue

            n_pos = int(y.sum())
            n_neg = len(y) - n_pos
            chance = max(np.mean(y), 1 - np.mean(y))

            # Compute metrics
            sil   = silhouette(X, y)
            lda   = lda_accuracy(X, y)
            fr    = fisher_ratio(X, y)
            ev    = pca_variance(X, n_components=5)

            lines.append(
                f"\n  {qtype} | Layer {best_layer} | n={len(X)} "
                f"({n_pos} flip / {n_neg} hold) | chance={chance:.3f}"
            )
            lines.append(f"  Silhouette (PCA-50)  : {sil:+.4f}  (>0 = flip/hold separated)")
            lines.append(f"  LDA accuracy (5-CV)  : {lda:.4f}  (chance={chance:.3f})")
            lines.append(f"  Fisher ratio         : {fr:.6f}")
            lines.append(f"  PCA var[PC1..PC5]    : {' '.join(f'{v:.3f}' for v in ev)}")
            lines.append(f"  PCA cumvar[5]        : {ev.sum():.3f}")

            rows.append({
                "model": model_name, "qtype": qtype, "layer": best_layer,
                "n": len(X), "n_flip": n_pos, "n_hold": n_neg,
                "chance": chance, "silhouette": sil, "lda_acc": lda,
                "fisher_ratio": fr, "pca_var_pc1": ev[0], "pca_cumvar5": ev.sum(),
            })

        # t-SNE at best layer for base qtype (save coords for plotting)
        lines.append(f"\n  Computing t-SNE (base, layer {best_layer}) ...")
        hs, _, flip_data = load_data(model_name, "base")
        if hs is not None:
            X, y, _ = build_preflip_dataset(hs, flip_data, best_layer)
            if X is not None and len(np.unique(y)) >= 2:
                coords, labels = tsne_coords(X, y)
                if coords is not None:
                    tsne_df = pd.DataFrame({
                        "x": coords[:, 0], "y": coords[:, 1], "label": labels
                    })
                    out_path = TSNE_DIR / f"{model_name.replace('/', '_')}_base_L{best_layer}_tsne.csv"
                    tsne_df.to_csv(out_path, index=False)
                    lines.append(f"  t-SNE coordinates saved → {out_path.name}")

                    # Quick separation metric from t-SNE coords
                    flip_c = coords[labels == 1]
                    hold_c = coords[labels == 0]
                    if len(flip_c) > 0 and len(hold_c) > 0:
                        mu_flip = flip_c.mean(axis=0)
                        mu_hold = hold_c.mean(axis=0)
                        centroid_dist = float(np.linalg.norm(mu_flip - mu_hold))
                        lines.append(f"  t-SNE centroid distance (flip vs hold): {centroid_dist:.2f}")
                else:
                    lines.append("  t-SNE failed (sklearn TSNE unavailable or error)")

    # ── Summary table ────────────────────────────────────────────────────────
    lines.append(section("SUMMARY — geometric separability at best probe layer"))
    lines.append(
        f"  {'Model':<20} {'QType':<16} {'Silhouette':>11} "
        f"{'LDA acc':>9} {'Chance':>8} {'Fisher':>10}"
    )
    lines.append(f"  {'─'*20} {'─'*16} {'─'*11} {'─'*9} {'─'*8} {'─'*10}")
    for r in rows:
        lines.append(
            f"  {r['model']:<20} {r['qtype']:<16} {r['silhouette']:>+11.4f} "
            f"{r['lda_acc']:>9.4f} {r['chance']:>8.3f} {r['fisher_ratio']:>10.6f}"
        )

    lines.append(
        "\n  Interpretation: Low Fisher ratio + low LDA accuracy (near chance) despite"
        "\n  nonlinear classifiers beating chance → signal is in curved geometry, not"
        "\n  linear separable directions. Negative silhouette = overlapping distributions."
    )

    return "\n".join(lines), rows


def main():
    report, rows = run_analysis()
    print(report)

    OUTPUT_TXT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_TXT.write_text(report + "\n", encoding="utf-8")
    print(f"\nReport saved to {OUTPUT_TXT}")

    if rows:
        df = pd.DataFrame(rows)
        df.to_csv(OUTPUT_CSV, index=False)
        print(f"Data saved to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
