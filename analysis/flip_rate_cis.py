#!/usr/bin/env python3
"""95% binomial confidence intervals for ever-flip rates (paper Table 2).

For each (model, prompt condition) and both labeling methods (keyword, judge),
computes the ever-flip rate with a 95% CI. Reports both the normal
approximation (p +/- 1.96*sqrt(p(1-p)/n)) used in the paper table for
compactness and the Wilson interval in the CSV.

Outputs:
    analysis_claude/flip_rate_cis.csv
    analysis_claude/flip_rate_cis_table.tex   (LaTeX rows for Table 2)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
from flip_labeling import response_flipped

OUT_CSV = REPO_ROOT / "analysis_claude" / "flip_rate_cis.csv"
OUT_TEX = REPO_ROOT / "analysis_claude" / "flip_rate_cis_table.tex"

KEYWORD_DIRS = {
    "DeepSeek-R1-7B": REPO_ROOT / "data" / "DeepSeek-R1-Distill-Qwen-7B",
    "Gemma-2-9B":     REPO_ROOT / "data" / "Gemma-2-9B",
    "Llama-3.1-8B":   REPO_ROOT / "data" / "Llama-3.1-8B-Instruct",
    "Qwen2.5-7B":     REPO_ROOT / "data" / "Qwen2.5-7B-Instruct",
    "Qwen3.5-9B":     REPO_ROOT / "data" / "Qwen3.5-9B",
}

JUDGE_CSVS = {
    "DeepSeek-R1-7B": (REPO_ROOT / "analysis_claude" / "claude_judgements.csv",        "DeepSeek-R1-Distill-Qwen-7B"),
    "Gemma-2-9B":     (REPO_ROOT / "analysis_claude" / "gemma_judgements_haiku.csv",   "Gemma-2-9B"),
    "Llama-3.1-8B":   (REPO_ROOT / "analysis_claude" / "llama31_judgements_haiku.csv", "Llama-3.1-8B-Instruct"),
    "Qwen2.5-7B":     (REPO_ROOT / "analysis_claude" / "qwen25_judgements_haiku.csv",  "Qwen2.5-7B-Instruct"),
    "Qwen3.5-9B":     (REPO_ROOT / "analysis_claude" / "qwen35_judgements_haiku.csv",  "Qwen3.5-9B"),
}

CONDITIONS = ["base", "critical", "presupposition"]
Z = 1.959964


def normal_ci(k: int, n: int) -> tuple[float, float, float]:
    """(rate, lo, hi) via normal approximation, clipped to [0, 1]."""
    p = k / n
    half = Z * np.sqrt(p * (1 - p) / n)
    return p, max(0.0, p - half), min(1.0, p + half)


def wilson_ci(k: int, n: int) -> tuple[float, float]:
    p = k / n
    denom = 1 + Z**2 / n
    center = (p + Z**2 / (2 * n)) / denom
    half = (Z / denom) * np.sqrt(p * (1 - p) / n + Z**2 / (4 * n**2))
    return max(0.0, center - half), min(1.0, center + half)


def keyword_ever_flip(model: str, cond: str) -> tuple[int, int]:
    csv_path = KEYWORD_DIRS[model] / f"{cond}_multiturn.csv"
    df = pd.read_csv(csv_path)
    turn_cols = [c for c in (f"Response_Turn_{t}" for t in range(1, 6)) if c in df.columns]
    flips = sum(
        any(response_flipped(str(row.get(c, ""))) for c in turn_cols)
        for _, row in df.iterrows()
    )
    return int(flips), len(df)


def judge_ever_flip(model: str, cond: str) -> tuple[int, int]:
    csv_path, model_col = JUDGE_CSVS[model]
    df = pd.read_csv(csv_path)
    sub = df[(df["model"] == model_col) & (df["question_type"] == cond)].copy()
    sub["flip"] = sub["judgement"].astype(str).str.lower() == "true"
    per_q = sub.groupby("question")["flip"].any()
    return int(per_q.sum()), len(per_q)


def fmt_pm(p, lo, hi):
    return f"{p*100:.1f}$\\pm${(hi - lo)*50:.1f}"


def main():
    rows, tex_lines = [], []
    for model in KEYWORD_DIRS:
        cells = {}
        for method, fn in (("keyword", keyword_ever_flip), ("judge", judge_ever_flip)):
            for cond in CONDITIONS:
                k, n = fn(model, cond)
                p, lo, hi = normal_ci(k, n)
                wlo, whi = wilson_ci(k, n)
                rows.append({
                    "model": model, "method": method, "condition": cond,
                    "n_flip": k, "n_total": n, "rate": round(p, 4),
                    "normal_lo": round(lo, 4), "normal_hi": round(hi, 4),
                    "wilson_lo": round(wlo, 4), "wilson_hi": round(whi, 4),
                })
                cells[(method, cond)] = fmt_pm(p, lo, hi)
        tex_lines.append(
            f"  {model:<15} & & "
            + " & ".join(cells[("keyword", c)] for c in CONDITIONS)
            + " & "
            + " & ".join(cells[("judge", c)] for c in CONDITIONS)
            + " \\\\"
        )

    pd.DataFrame(rows).to_csv(OUT_CSV, index=False)
    OUT_TEX.write_text("\n".join(tex_lines) + "\n", encoding="utf-8")

    print(pd.DataFrame(rows).to_string(index=False))
    print(f"\nSaved -> {OUT_CSV}\nSaved -> {OUT_TEX}\n")
    print("LaTeX rows:\n" + "\n".join(tex_lines))


if __name__ == "__main__":
    main()
