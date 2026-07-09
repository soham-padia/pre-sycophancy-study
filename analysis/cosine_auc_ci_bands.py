#!/usr/bin/env python3
"""Per-layer bootstrap CI bands for the T0->T1 cosine AUC (Figure 3A).

For each model at its best prompt condition (matching Panel A of
hidden_state_disruption.png), computes the cosine-disruption AUC at every
layer plus a 95% bootstrap CI (resampling questions, 1000 resamples).

Output:
    analysis_claude/cosine_auc_ci_bands.csv  (model, qtype, layer, auc, lo, hi, n)
"""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_CSV   = REPO_ROOT / "analysis_claude" / "cosine_auc_ci_bands.csv"

MODELS = {
    "DeepSeek-R1-7B": {"dir": "data/DeepSeek-R1-Distill-Qwen-7B", "n_layers": 29, "qtype": "critical"},
    "Qwen2.5-7B":     {"dir": "data/Qwen2.5-7B-Instruct",          "n_layers": 29, "qtype": "base"},
    "Llama-3.1-8B":   {"dir": "data/Llama-3.1-8B-Instruct",        "n_layers": 33, "qtype": "critical"},
    "Qwen3.5-9B":     {"dir": "data/Qwen3.5-9B",                   "n_layers": 33, "qtype": "presupposition"},
}

JUDGE_CSVS = {
    "DeepSeek-R1-7B": (REPO_ROOT / "analysis_claude" / "claude_judgements.csv",        "DeepSeek-R1-Distill-Qwen-7B"),
    "Qwen2.5-7B":     (REPO_ROOT / "analysis_claude" / "qwen25_judgements_haiku.csv",  "Qwen2.5-7B-Instruct"),
    "Llama-3.1-8B":   (REPO_ROOT / "analysis_claude" / "llama31_judgements_haiku.csv", "Llama-3.1-8B-Instruct"),
    "Qwen3.5-9B":     (REPO_ROOT / "analysis_claude" / "qwen35_judgements_haiku.csv",  "Qwen3.5-9B"),
}

N_BOOT = 1000


def load_flip_info(model_name: str, qtype: str) -> dict[str, dict]:
    """{question: {ever_flip, first_flip_turn}}"""
    judge_csv, model_col = JUDGE_CSVS[model_name]
    df = pd.read_csv(judge_csv)
    df = df[(df["model"] == model_col) & (df["question_type"] == qtype)].copy()
    df["flip"] = df["judgement"].astype(str).str.lower() == "true"
    info = {}
    for q, grp in df.groupby("question"):
        first = grp[grp["flip"]]["turn"].min() if grp["flip"].any() else None
        info[q] = {"ever_flip": bool(grp["flip"].any()),
                   "first_flip_turn": int(first) if first is not None and not pd.isna(first) else None}
    return info


def cos_sims_all_layers(hs, flip_info: dict[str, dict], n_layers: int):
    """Return (sims (n_q, n_layers), labels (n_q,), t1flip (n_q,)) for T0 vs T1."""
    sims, labels, t1flip = [], [], []
    for q, info in flip_info.items():
        try:
            v0 = hs[q][0].float().numpy()
            v1 = hs[q][1].float().numpy()
        except (KeyError, IndexError, TypeError, AttributeError):
            continue
        if v0 is None or v1 is None or v0.shape[0] < n_layers:
            continue
        a = v0[:n_layers] / (np.linalg.norm(v0[:n_layers], axis=1, keepdims=True) + 1e-10)
        b = v1[:n_layers] / (np.linalg.norm(v1[:n_layers], axis=1, keepdims=True) + 1e-10)
        sims.append(np.sum(a * b, axis=1))
        labels.append(int(info["ever_flip"]))
        t1flip.append(int(info["first_flip_turn"] == 1) if info["first_flip_turn"] else 0)
    return np.array(sims), np.array(labels), np.array(t1flip)


def band_rows(model_name, qtype, variant, sims, labels, rng):
    """Per-layer AUC + bootstrap CI + mean drift rows for one variant."""
    n = len(labels)
    boot_idx = rng.integers(0, n, size=(N_BOOT, n))
    out = []
    for layer in range(sims.shape[1]):
        s = sims[:, layer]
        if len(np.unique(labels)) < 2:
            continue
        auc = roc_auc_score(labels, -s)
        boots = []
        for bi in boot_idx:
            yb = labels[bi]
            if yb.min() == yb.max():
                continue
            boots.append(roc_auc_score(yb, -s[bi]))
        lo, hi = np.percentile(boots, [2.5, 97.5]) if len(boots) > 10 else (np.nan, np.nan)
        out.append({"model": model_name, "qtype": qtype, "variant": variant,
                    "layer": layer, "auc": round(auc, 4), "lo": round(float(lo), 4),
                    "hi": round(float(hi), 4), "mean_drift": round(float(1 - s.mean()), 4),
                    "n": n})
    return out


def main():
    rng = np.random.default_rng(42)
    rows = []
    for model_name, cfg in MODELS.items():
        qtype = cfg["qtype"]
        pt_path = REPO_ROOT / cfg["dir"] / f"{qtype}_multiturn_hidden_states.pt"
        hs = torch.load(str(pt_path), map_location="cpu")
        flip_info = load_flip_info(model_name, qtype)

        sims, labels, t1flip = cos_sims_all_layers(hs, flip_info, cfg["n_layers"])
        keep = t1flip == 0
        print(f"{model_name} ({qtype}): n={len(labels)}, {labels.sum()} flip, "
              f"{int(t1flip.sum())} flip-at-T1 (excl variant n={int(keep.sum())})")

        rows += band_rows(model_name, qtype, "all", sims, labels, rng)
        rows += band_rows(model_name, qtype, "excl_t1", sims[keep], labels[keep], rng)

    pd.DataFrame(rows).to_csv(OUT_CSV, index=False)
    print(f"Saved -> {OUT_CSV}")


if __name__ == "__main__":
    main()
