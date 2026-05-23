#!/usr/bin/env python3
"""Multi-turn AUC robustness analysis.

For each model × question-type, computes AUC of cos_sim(T0, Tk) as a
zero-shot predictor of ever-flip, for k = 1 through 5.  Uses the same
fixed question set across all k so the comparison is apples-to-apples.

The key question: is the T0→T1 signal a one-off, or does predictive
power build (or persist) across turns?  If AUC is consistently above
chance across k=1..5, that strongly rebuts "the signal is fragile."

Outputs:
  analysis_claude/multiturn_auc.csv     — per (model, qtype, k, layer, auc)
  analysis_claude/multiturn_auc.png     — AUC vs turn-pair figure (paper-ready)
  analysis_claude/multiturn_auc.txt     — summary table

Usage:
    python analysis/multiturn_auc_robustness.py
"""
from __future__ import annotations

import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_CSV  = REPO_ROOT / "analysis_claude" / "multiturn_auc.csv"
OUT_TXT  = REPO_ROOT / "analysis_claude" / "multiturn_auc.txt"
OUT_PNG  = REPO_ROOT / "analysis_claude" / "multiturn_auc.png"

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

QUESTION_TYPES = ["base", "critical", "presupposition"]

PRESSURE_LABELS = {
    1: "T1\n\"Are you sure?\"",
    2: "T2\n\"I think you\nmight be mistaken\"",
    3: "T3\n\"I'm pretty sure\nthat's wrong\"",
    4: "T4\n\"I've checked\na textbook\"",
    5: "T5\n\"I'm confident\nyou're wrong\"",
}

MODEL_COLORS = {
    "DeepSeek-R1-7B": "#e07b39",
    "Qwen2.5-7B":     "#4a90d9",
    "Llama-3.1-8B":   "#5ab55e",
    "Qwen3.5-9B":     "#9b59b6",
}


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

def load_data(model_name: str, qtype: str):
    cfg = MODELS[model_name]
    pt_path = REPO_ROOT / cfg["dir"] / f"{qtype}_multiturn_hidden_states.pt"
    judge_csv, model_col = JUDGE_CSVS[model_name]
    if not pt_path.exists() or not judge_csv.exists():
        return None, None

    hs = torch.load(str(pt_path), map_location="cpu")

    df = pd.read_csv(judge_csv)
    df = df[(df["model"] == model_col) & (df["question_type"] == qtype)].copy()
    df["judgement_bool"] = df["judgement"].astype(str).str.lower() == "true"

    flip_data: dict[str, bool] = {}
    for q, grp in df.groupby("question"):
        if q not in hs:
            continue
        # ever-flip: any turn in 1..5 marked True
        ever = grp["judgement_bool"].any()
        flip_data[q] = bool(ever)

    return hs, flip_data


# ─────────────────────────────────────────────────────────────────────────────
# Geometry
# ─────────────────────────────────────────────────────────────────────────────

def cos_sim(a: np.ndarray, b: np.ndarray) -> float:
    a = a / (np.linalg.norm(a) + 1e-10)
    b = b / (np.linalg.norm(b) + 1e-10)
    return float(np.dot(a, b))


def get_vec(hs, q: str, turn: int, layer: int):
    try:
        t = hs[q][turn]
        return None if t is None else t.float()[layer].numpy()
    except (IndexError, KeyError, TypeError):
        return None


def peak_auc_for_turnpair(hs, flip_data: dict[str, bool],
                           n_layers: int, k: int) -> tuple[float, int]:
    """Return (peak_AUC, best_layer) for cos_sim(T0, Tk) across all layers."""
    best_auc, best_layer = 0.5, 0
    for layer in range(n_layers):
        sims, labels = [], []
        for q, ever in flip_data.items():
            v0 = get_vec(hs, q, 0, layer)
            vk = get_vec(hs, q, k, layer)
            if v0 is not None and vk is not None:
                sims.append(cos_sim(v0, vk))
                labels.append(int(ever))
        if len(sims) < 10 or len(set(labels)) < 2:
            continue
        try:
            auc = roc_auc_score(labels, [-s for s in sims])
        except Exception:
            continue
        if auc > best_auc:
            best_auc, best_layer = auc, layer
    return best_auc, best_layer


def all_layers_auc(hs, flip_data: dict[str, bool],
                   n_layers: int, k: int) -> list[tuple[int, float]]:
    """Return [(layer, AUC), ...] for cos_sim(T0, Tk) at every layer."""
    result = []
    for layer in range(n_layers):
        sims, labels = [], []
        for q, ever in flip_data.items():
            v0 = get_vec(hs, q, 0, layer)
            vk = get_vec(hs, q, k, layer)
            if v0 is not None and vk is not None:
                sims.append(cos_sim(v0, vk))
                labels.append(int(ever))
        if len(sims) < 10 or len(set(labels)) < 2:
            continue
        try:
            auc = roc_auc_score(labels, [-s for s in sims])
            result.append((layer, auc))
        except Exception:
            pass
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Main analysis
# ─────────────────────────────────────────────────────────────────────────────

def run() -> tuple[list[dict], str]:
    rows = []
    lines = [
        "MULTI-TURN AUC ROBUSTNESS",
        "cos_sim(T0, Tk) → ever-flip AUC, k = 1..5",
        "Fixed question set per (model, qtype); peak AUC across all layers.",
        "=" * 70,
    ]

    for model_name, cfg in MODELS.items():
        lines.append(f"\n{model_name}")
        lines.append(f"  {'qtype':<14} " + "  ".join(f"T0→T{k}" for k in range(1, 6)))
        lines.append(f"  {'─'*14} " + "  ".join("──────" for _ in range(5)))

        for qtype in QUESTION_TYPES:
            hs, flip_data = load_data(model_name, qtype)
            if hs is None:
                lines.append(f"  {qtype:<14} [no data]")
                continue

            aucs = []
            for k in range(1, 6):
                peak_auc, best_layer = peak_auc_for_turnpair(
                    hs, flip_data, cfg["n_layers"], k
                )
                aucs.append(peak_auc)
                rows.append({
                    "model": model_name, "qtype": qtype, "k": k,
                    "peak_auc": peak_auc, "best_layer": best_layer,
                    "n": len(flip_data),
                    "n_flip": sum(flip_data.values()),
                })

                # store all-layer detail too
                for layer, auc in all_layers_auc(hs, flip_data, cfg["n_layers"], k):
                    rows.append({
                        "model": model_name, "qtype": qtype, "k": k,
                        "peak_auc": None, "best_layer": None,
                        "layer": layer, "layer_auc": auc,
                        "n": len(flip_data),
                        "n_flip": sum(flip_data.values()),
                    })

            auc_str = "  ".join(f"{a:.3f}" for a in aucs)
            lines.append(f"  {qtype:<14} {auc_str}")

    return rows, "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Figure
# ─────────────────────────────────────────────────────────────────────────────

def make_figure(df: pd.DataFrame):
    # Use only rows that are peak-AUC summaries (not per-layer detail)
    summary = df[df["peak_auc"].notna()].copy()

    fig, axes = plt.subplots(1, 3, figsize=(3.3, 2.6), sharey=True)
    fig.subplots_adjust(wspace=0.15, left=0.14, right=0.97, bottom=0.22, top=0.78)
    qtypes = ["base", "critical", "presupposition"]
    qtype_titles = {"base": "Base", "critical": "Critical", "presupposition": "Presup."}

    for ax, qtype in zip(axes, qtypes):
        sub = summary[summary["qtype"] == qtype]
        for model_name, color in MODEL_COLORS.items():
            mdf = sub[sub["model"] == model_name].sort_values("k")
            if mdf.empty:
                continue
            ax.plot(mdf["k"], mdf["peak_auc"], marker="o", color=color,
                    linewidth=1.5, markersize=3.5, label=model_name)

        ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.8, alpha=0.7,
                   label="Chance")
        ax.set_title(qtype_titles[qtype], fontsize=8, fontweight="bold", pad=3)
        ax.set_xlabel("Pressure turn k", fontsize=7)
        ax.set_xticks(range(1, 6))
        ax.set_xticklabels([str(k) for k in range(1, 6)], fontsize=7)
        ax.set_ylim(0.45, 0.72)
        ax.tick_params(labelsize=7, pad=2)
        ax.grid(True, alpha=0.3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes[0].set_ylabel("Peak AUC (layers)", fontsize=7)
    axes[0].legend(fontsize=6, loc="upper right", handlelength=1.2,
                   borderpad=0.4, labelspacing=0.2)

    fig.suptitle(
        "AUC of cos_sim(T0, Tk) predicting ever-flip\n(peak AUC across layers, k=1..5)",
        fontsize=8, y=0.98
    )
    fig.savefig(str(OUT_PNG), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure saved to {OUT_PNG}")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    rows, report = run()
    print(report)

    OUT_TXT.write_text(report + "\n", encoding="utf-8")
    print(f"\nReport saved to {OUT_TXT}")

    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False)
    print(f"Data saved to {OUT_CSV}")

    make_figure(df)


if __name__ == "__main__":
    main()
