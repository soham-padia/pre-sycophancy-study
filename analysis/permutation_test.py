#!/usr/bin/env python3
"""Sweep-corrected permutation test for the peak T0->T1 cosine AUC.

Addresses the layer-sweeping concern: we search 29-33 layers and report the
peak AUC, so the peak is inflated relative to a pre-registered layer. The
correct null repeats the FULL layer sweep under shuffled labels:

    p = P( max_layer AUC(shuffled labels) >= observed peak AUC )

Implementation note: with scores fixed per layer, AUC under a permuted label
vector is a linear function of the rank-sum of the positive class, so each
permutation's full layer sweep is a single matrix-vector product. This makes
10,000 permutations per (model, condition) cell effectively free.

Outputs:
    analysis_claude/permutation_test.csv
    analysis_claude/permutation_test.txt
"""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import rankdata

warnings.filterwarnings("ignore")

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_CSV = REPO_ROOT / "analysis_claude" / "permutation_test.csv"
OUT_TXT = REPO_ROOT / "analysis_claude" / "permutation_test.txt"

MODELS = {
    "DeepSeek-R1-7B": {"dir": "data/DeepSeek-R1-Distill-Qwen-7B", "n_layers": 29},
    "Qwen2.5-7B":     {"dir": "data/Qwen2.5-7B-Instruct",          "n_layers": 29},
    "Llama-3.1-8B":   {"dir": "data/Llama-3.1-8B-Instruct",        "n_layers": 33},
    "Qwen3.5-9B":     {"dir": "data/Qwen3.5-9B",                   "n_layers": 33},
}

JUDGE_CSVS = {
    "DeepSeek-R1-7B": (REPO_ROOT / "analysis_claude" / "claude_judgements.csv",        "DeepSeek-R1-Distill-Qwen-7B"),
    "Qwen2.5-7B":     (REPO_ROOT / "analysis_claude" / "qwen25_judgements_haiku.csv",  "Qwen2.5-7B-Instruct"),
    "Llama-3.1-8B":   (REPO_ROOT / "analysis_claude" / "llama31_judgements_haiku.csv", "Llama-3.1-8B-Instruct"),
    "Qwen3.5-9B":     (REPO_ROOT / "analysis_claude" / "qwen35_judgements_haiku.csv",  "Qwen3.5-9B"),
}

CONDITIONS = ["base", "critical", "presupposition"]
N_PERM = 10_000


def load_ever_flip(model_name: str, cond: str) -> dict[str, bool]:
    judge_csv, model_col = JUDGE_CSVS[model_name]
    df = pd.read_csv(judge_csv)
    df = df[(df["model"] == model_col) & (df["question_type"] == cond)].copy()
    df["flip"] = df["judgement"].astype(str).str.lower() == "true"
    return {q: bool(g["flip"].any()) for q, g in df.groupby("question")}


def cos_sims_all_layers(hs, ever_flip, n_layers):
    sims, labels = [], []
    for q, ever in ever_flip.items():
        try:
            v0 = hs[q][0].float().numpy()
            v1 = hs[q][1].float().numpy()
        except (KeyError, IndexError, TypeError, AttributeError):
            continue
        a = v0[:n_layers] / (np.linalg.norm(v0[:n_layers], axis=1, keepdims=True) + 1e-10)
        b = v1[:n_layers] / (np.linalg.norm(v1[:n_layers], axis=1, keepdims=True) + 1e-10)
        sims.append(np.sum(a * b, axis=1))
        labels.append(int(ever))
    return np.array(sims), np.array(labels)


def sweep_auc_from_ranks(ranks: np.ndarray, y: np.ndarray) -> np.ndarray:
    """AUC per layer for label vector y given per-layer ranks of the scores.

    ranks: (n, L) ranks of (-cos_sim) per layer (average ranks for ties)
    y:     (n,) binary labels
    AUC_l = (sum_{i: y_i=1} rank_il - n1(n1+1)/2) / (n1 * n0)
    """
    n1 = int(y.sum())
    n0 = len(y) - n1
    rank_sums = y @ ranks              # (L,)
    return (rank_sums - n1 * (n1 + 1) / 2) / (n1 * n0)


def main():
    rng = np.random.default_rng(42)
    rows, lines = [], [
        "SWEEP-CORRECTED PERMUTATION TEST",
        f"Null: max-over-layers AUC with shuffled ever-flip labels ({N_PERM:,} permutations)",
        "p = P(shuffled peak >= observed peak)",
        "=" * 78,
        f"  {'Model':<16} {'Cond.':<14} {'Obs peak':>8} {'Layer':>5} "
        f"{'Null median':>11} {'Null 95th':>9} {'p':>8}",
        "  " + "-" * 76,
    ]

    for model_name, cfg in MODELS.items():
        for cond in CONDITIONS:
            pt = REPO_ROOT / cfg["dir"] / f"{cond}_multiturn_hidden_states.pt"
            hs = torch.load(str(pt), map_location="cpu")
            ever = load_ever_flip(model_name, cond)
            sims, y = cos_sims_all_layers(hs, ever, cfg["n_layers"])
            del hs
            if len(np.unique(y)) < 2:
                continue

            ranks = np.column_stack([rankdata(-sims[:, l]) for l in range(sims.shape[1])])
            obs_aucs = sweep_auc_from_ranks(ranks, y.astype(float))
            obs_peak = float(obs_aucs.max())
            obs_layer = int(obs_aucs.argmax())

            null_peaks = np.empty(N_PERM)
            yf = y.astype(float)
            for i in range(N_PERM):
                yp = rng.permutation(yf)
                null_peaks[i] = sweep_auc_from_ranks(ranks, yp).max()

            p = float((null_peaks >= obs_peak).mean())
            med, q95 = float(np.median(null_peaks)), float(np.percentile(null_peaks, 95))

            lines.append(
                f"  {model_name:<16} {cond:<14} {obs_peak:>8.3f} {obs_layer:>5} "
                f"{med:>11.3f} {q95:>9.3f} {p:>8.4f}"
            )
            rows.append({
                "model": model_name, "condition": cond, "n": len(y),
                "n_flip": int(y.sum()), "observed_peak_auc": round(obs_peak, 4),
                "observed_layer": obs_layer, "null_median_peak": round(med, 4),
                "null_p95_peak": round(q95, 4), "p_value": round(p, 4),
            })
            print(lines[-1])

    lines.append("")
    lines.append("Interpretation: the null median shows how high a peak AUC the layer")
    lines.append("sweep alone produces under label-shuffled data. Cells with p < 0.05")
    lines.append("have peaks that the sweep cannot explain.")

    pd.DataFrame(rows).to_csv(OUT_CSV, index=False)
    OUT_TXT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nSaved -> {OUT_CSV}\nSaved -> {OUT_TXT}")


if __name__ == "__main__":
    main()
