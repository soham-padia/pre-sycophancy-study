#!/usr/bin/env python3
"""Confidence intervals for the pre-flip geometry table (paper Table 4).

For each (model, condition) at the model's best probe layer:
  - Fisher ratio with a 95% bootstrap CI (resampling questions, so all turns of
    a question stay together; 1000 resamples)
  - LDA 5-fold CV accuracy with a 95% Wilson CI on the pooled predictions

Reuses the exact dataset construction from analysis/preflip_geometry.py so the
point estimates reproduce the published table before CIs are added.

Outputs:
    analysis_claude/geometry_cis.csv
    analysis_claude/geometry_cis_table.tex
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "analysis"))

from preflip_geometry import (MODELS, QUESTION_TYPES, load_data, fisher_ratio)  # noqa: E402
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis            # noqa: E402
from sklearn.decomposition import PCA                                           # noqa: E402
from sklearn.model_selection import StratifiedKFold                             # noqa: E402
from sklearn.preprocessing import StandardScaler                                # noqa: E402
from sklearn.metrics import accuracy_score                                      # noqa: E402

OUT_CSV = REPO_ROOT / "analysis_claude" / "geometry_cis.csv"
OUT_TEX = REPO_ROOT / "analysis_claude" / "geometry_cis_table.tex"

N_BOOT = 1000
Z = 1.959964


def build_preflip_dataset_with_qids(hs, flip_data: dict, layer: int):
    """Mirrors preflip_geometry.build_preflip_dataset but also returns qids."""
    X, y, qids = [], [], []
    for qi, (q, labels) in enumerate(flip_data.items()):
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
            qids.append(qi)
    if not X:
        return None, None, None
    return np.array(X), np.array(y), np.array(qids)


def fisher_bootstrap_ci(X, y, qids, n_boot=N_BOOT, seed=42):
    rng = np.random.default_rng(seed)
    uq = np.unique(qids)
    per_q = {qi: np.where(qids == qi)[0] for qi in uq}
    boots = []
    for _ in range(n_boot):
        chosen = rng.choice(uq, size=len(uq), replace=True)
        idx = np.concatenate([per_q[qi] for qi in chosen])
        yb = y[idx]
        if yb.min() == yb.max():
            continue
        boots.append(fisher_ratio(X[idx], yb))
    if len(boots) < 10:
        return np.nan, np.nan
    return float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def lda_accuracy_pooled(X, y, n_splits=5):
    """Same protocol as preflip_geometry.lda_accuracy; returns (acc, n_pooled)."""
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    n_pca = min(50, X.shape[1], X.shape[0] - 1)
    X = PCA(n_components=n_pca).fit_transform(X)
    skf = StratifiedKFold(n_splits=min(n_splits, min(np.bincount(y.astype(int)))),
                          shuffle=True, random_state=42)
    preds, truths = [], []
    for tr, te in skf.split(X, y):
        clf = LinearDiscriminantAnalysis()
        clf.fit(X[tr], y[tr])
        preds.extend(clf.predict(X[te]))
        truths.extend(y[te])
    return float(accuracy_score(truths, preds)), len(truths)


def wilson_ci(p: float, n: int) -> tuple[float, float]:
    denom = 1 + Z**2 / n
    center = (p + Z**2 / (2 * n)) / denom
    half = (Z / denom) * np.sqrt(p * (1 - p) / n + Z**2 / (4 * n**2))
    return max(0.0, center - half), min(1.0, center + half)


SHORT = {"DeepSeek-R1-7B": "DS", "Qwen2.5-7B": "Q2.5",
         "Llama-3.1-8B": "L3.1", "Qwen3.5-9B": "Q3.5"}
COND_SHORT = {"base": "base", "critical": "critical", "presupposition": "presup."}


def main():
    rows, tex = [], []
    for model_name, cfg in MODELS.items():
        layer = cfg["best_layer"]
        model_tex = []
        for qtype in QUESTION_TYPES:
            hs, _, flip_data = load_data(model_name, qtype)
            if hs is None:
                continue
            X, y, qids = build_preflip_dataset_with_qids(hs, flip_data, layer)
            if X is None or len(np.unique(y)) < 2:
                continue

            chance = max(np.mean(y), 1 - np.mean(y))
            fr = fisher_ratio(X, y)
            fr_lo, fr_hi = fisher_bootstrap_ci(X, y, qids)
            lda, n_pooled = lda_accuracy_pooled(X, y)
            lda_lo, lda_hi = wilson_ci(lda, n_pooled)

            print(f"{model_name} {qtype}: Fisher={fr:.4f} [{fr_lo:.4f},{fr_hi:.4f}]  "
                  f"LDA={lda:.4f} [{lda_lo:.4f},{lda_hi:.4f}]  chance={chance:.3f}  n={len(y)}")
            rows.append({
                "model": model_name, "condition": qtype, "layer": layer, "n": len(y),
                "fisher": round(fr, 4), "fisher_lo": round(fr_lo, 4), "fisher_hi": round(fr_hi, 4),
                "lda_acc": round(lda, 4), "lda_lo": round(lda_lo, 4), "lda_hi": round(lda_hi, 4),
                "chance": round(chance, 4),
            })
            model_tex.append(
                f"    & {COND_SHORT[qtype]:<10} & "
                f"{fr:.3f} {{\\scriptsize[{fr_lo:.3f}, {fr_hi:.3f}]}} & "
                f"{lda:.3f}$\\pm${(lda_hi - lda_lo) / 2:.3f} & {chance:.3f} \\\\"
            )
        if model_tex:
            tex.append(f"  \\multirow{{3}}{{*}}{{{SHORT[model_name]}}}")
            tex.extend(model_tex)
            tex.append("  \\addlinespace")

    pd.DataFrame(rows).to_csv(OUT_CSV, index=False)
    OUT_TEX.write_text("\n".join(tex) + "\n", encoding="utf-8")
    print(f"\nSaved -> {OUT_CSV}\nSaved -> {OUT_TEX}")


if __name__ == "__main__":
    main()
