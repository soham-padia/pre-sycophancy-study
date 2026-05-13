#!/usr/bin/env python3
"""Compare inter-judge agreement between Claude Haiku and Llama 3.1 70B judgements.

Computes:
  - Overall % agreement and Cohen's kappa
  - Breakdown by question_type and turn
  - Flip rate per judge
  - Confusion matrix
  - Disagreement examples

Usage:
    python analysis/compare_judges.py
    python analysis/compare_judges.py --output analysis_claude/judge_comparison.txt
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.metrics import cohen_kappa_score, confusion_matrix

REPO_ROOT = Path(__file__).resolve().parents[1]

LLAMA_CSV = REPO_ROOT / "analysis" / "judge_outputs" / "ndif_judgements_trajectory.csv"
HAIKU_CSV = REPO_ROOT / "analysis_claude" / "claude_judgements.csv"
DEFAULT_OUTPUT = REPO_ROOT / "analysis_claude" / "judge_comparison.txt"


def filter_first_flip(df: pd.DataFrame, judge_col: str = "judgement_haiku") -> pd.DataFrame:
    """Keep only turns up to and including the first flip per question (per judge_col).
    Rows after the first True are dropped. Questions that never flip are kept in full.
    """
    df = df.sort_values(["model", "question_type", "question", "turn"]).copy()

    # For each question find the first flip turn; mark rows after it for removal
    first_flip = (
        df[df[judge_col] == True]
        .groupby(["model", "question_type", "question"])["turn"]
        .min()
        .rename("first_flip_turn")
    )
    df = df.join(first_flip, on=["model", "question_type", "question"])
    df = df[df["first_flip_turn"].isna() | (df["turn"] <= df["first_flip_turn"])]
    return df.drop(columns=["first_flip_turn"]).reset_index(drop=True)


def load_and_merge() -> pd.DataFrame:
    llama = pd.read_csv(LLAMA_CSV)
    haiku = pd.read_csv(HAIKU_CSV)

    llama["judgement"] = llama["judgement"].astype(str).str.lower().map({"true": True, "false": False})
    haiku["judgement"] = haiku["judgement"].astype(str).str.lower().map({"true": True, "false": False})

    merged = pd.merge(
        llama, haiku,
        on=["model", "question_type", "question", "turn"],
        suffixes=("_llama", "_haiku"),
    )
    return merged


def fmt_pct(n, d):
    return f"{n}/{d} ({100 * n / d:.1f}%)" if d else "0/0"


def section(title: str) -> str:
    return f"\n{'=' * 70}\n{title}\n{'=' * 70}"


def build_report(df: pd.DataFrame) -> str:
    from scipy import stats as scipy_stats

    lines = []
    lines.append("INTER-JUDGE AGREEMENT: Claude Haiku 4.5 vs Llama 3.1 70B Instruct")
    lines.append(f"Overlapping rows: {len(df)}")

    y_llama = df["judgement_llama"].astype(int).tolist()
    y_haiku = df["judgement_haiku"].astype(int).tolist()

    agree = (df["judgement_llama"] == df["judgement_haiku"]).sum()
    kappa = cohen_kappa_score(y_llama, y_haiku)

    lines.append(section("OVERALL"))
    lines.append(f"Agreement : {fmt_pct(agree, len(df))}")
    lines.append(f"Cohen's κ : {kappa:.4f}  ({interpret_kappa(kappa)})")

    # Flip rates per judge
    lines.append(section("FLIP RATES"))
    llama_flips = df["judgement_llama"].sum()
    haiku_flips = df["judgement_haiku"].sum()
    lines.append(f"Llama 70B  flips: {fmt_pct(llama_flips, len(df))}")
    lines.append(f"Haiku 4.5  flips: {fmt_pct(haiku_flips, len(df))}")

    # Confusion matrix
    lines.append(section("CONFUSION MATRIX (rows=Llama, cols=Haiku)"))
    cm = confusion_matrix(y_llama, y_haiku)
    lines.append(f"              Haiku=False  Haiku=True")
    lines.append(f"Llama=False   {cm[0][0]:>11}  {cm[0][1]:>10}")
    lines.append(f"Llama=True    {cm[1][0]:>11}  {cm[1][1]:>10}")

    # By question type
    lines.append(section("AGREEMENT BY QUESTION TYPE"))
    for qtype, grp in df.groupby("question_type"):
        a = (grp["judgement_llama"] == grp["judgement_haiku"]).sum()
        k = cohen_kappa_score(
            grp["judgement_llama"].astype(int).tolist(),
            grp["judgement_haiku"].astype(int).tolist(),
        )
        lines.append(f"  {qtype:<20}  agreement={fmt_pct(a, len(grp))}  κ={k:.4f}")

    # By turn
    lines.append(section("AGREEMENT BY TURN (pressure level)"))
    for turn, grp in df.groupby("turn"):
        a = (grp["judgement_llama"] == grp["judgement_haiku"]).sum()
        k = cohen_kappa_score(
            grp["judgement_llama"].astype(int).tolist(),
            grp["judgement_haiku"].astype(int).tolist(),
        )
        lines.append(f"  Turn {turn}  agreement={fmt_pct(a, len(grp))}  κ={k:.4f}")

    # ------------------------------------------------------------------ #
    # Confidence analysis
    # ------------------------------------------------------------------ #
    lines.append(section("CONFIDENCE DISTRIBUTIONS"))
    for name, col in [("Llama 70B", "confidence_llama"), ("Haiku 4.5", "confidence_haiku")]:
        s = df[col]
        lines.append(
            f"  {name:<12}  mean={s.mean():.1f}  std={s.std():.1f}  "
            f"min={s.min()}  25%={s.quantile(.25):.0f}  "
            f"median={s.median():.0f}  75%={s.quantile(.75):.0f}  max={s.max()}"
        )
    pct_llama_100 = (df["confidence_llama"] == 100).mean() * 100
    lines.append(f"\n  Llama gives 100% confidence on {pct_llama_100:.1f}% of all rows")
    lines.append(f"  Haiku max confidence: {df['confidence_haiku'].max()} (never reaches 100)")

    lines.append(section("CONFIDENCE CORRELATION"))
    pearson_r, pearson_p = scipy_stats.pearsonr(df["confidence_llama"], df["confidence_haiku"])
    spearman_r, spearman_p = scipy_stats.spearmanr(df["confidence_llama"], df["confidence_haiku"])
    lines.append(f"  Pearson r  : {pearson_r:.4f}  (p={pearson_p:.4f})")
    lines.append(f"  Spearman ρ : {spearman_r:.4f}  (p={spearman_p:.4f})")

    lines.append(section("CONFIDENCE WHEN JUDGES AGREE vs DISAGREE"))
    agree_mask = df["judgement_llama"] == df["judgement_haiku"]
    for name, col in [("Llama 70B", "confidence_llama"), ("Haiku 4.5", "confidence_haiku")]:
        a_mean = df.loc[agree_mask, col].mean()
        d_mean = df.loc[~agree_mask, col].mean()
        lines.append(f"  {name:<12}  agree={a_mean:.1f}  disagree={d_mean:.1f}  Δ={a_mean - d_mean:+.1f}")

    lines.append(section("CONFIDENCE BY JUDGEMENT LABEL"))
    for name, jcol, ccol in [
        ("Llama 70B", "judgement_llama", "confidence_llama"),
        ("Haiku 4.5", "judgement_haiku", "confidence_haiku"),
    ]:
        t_mean = df.loc[df[jcol] == True, ccol].mean()
        f_mean = df.loc[df[jcol] == False, ccol].mean()
        lines.append(f"  {name:<12}  conf(flip=True)={t_mean:.1f}  conf(flip=False)={f_mean:.1f}  Δ={t_mean - f_mean:+.1f}")

    lines.append(section("CONFIDENCE BY TURN"))
    lines.append(f"  {'Turn':<6}  {'Llama mean':>12}  {'Haiku mean':>12}  {'Δ (L-H)':>10}")
    for turn, grp in df.groupby("turn"):
        lm = grp["confidence_llama"].mean()
        hm = grp["confidence_haiku"].mean()
        lines.append(f"  {turn:<6}  {lm:>12.1f}  {hm:>12.1f}  {lm - hm:>+10.1f}")

    lines.append(section("HIGH-CONFIDENCE DISAGREEMENTS (both ≥ 85, judges disagree, sample 10)"))
    hi_conf = df[
        (df["confidence_llama"] >= 85) &
        (df["confidence_haiku"] >= 85) &
        (df["judgement_llama"] != df["judgement_haiku"])
    ]
    lines.append(f"  Total high-confidence disagreements: {len(hi_conf)}")
    sample_hi = hi_conf.sample(min(10, len(hi_conf)), random_state=42)
    for _, row in sample_hi.iterrows():
        lines.append(
            f"\n  [{row['question_type']} | turn {row['turn']}]"
            f"\n  Q: {row['question'][:90]}"
            f"\n  Llama={row['judgement_llama']} (conf={row['confidence_llama']}) — {row['rationale_llama'][:110]}"
            f"\n  Haiku={row['judgement_haiku']} (conf={row['confidence_haiku']}) — {row['rationale_haiku'][:110]}"
        )

    # Disagreement examples (all confidence levels)
    lines.append(section("DISAGREEMENT EXAMPLES — all confidence (sample 10)"))
    disagree = df[df["judgement_llama"] != df["judgement_haiku"]]
    sample = disagree.sample(min(10, len(disagree)), random_state=42)
    for _, row in sample.iterrows():
        lines.append(
            f"\n  [{row['question_type']} | turn {row['turn']}]"
            f"\n  Q: {row['question'][:90]}"
            f"\n  Llama={row['judgement_llama']} (conf={row['confidence_llama']}) — {row['rationale_llama'][:100]}"
            f"\n  Haiku={row['judgement_haiku']} (conf={row['confidence_haiku']}) — {row['rationale_haiku'][:100]}"
        )

    return "\n".join(lines)


def interpret_kappa(k: float) -> str:
    if k < 0:
        return "poor"
    if k < 0.20:
        return "slight"
    if k < 0.40:
        return "fair"
    if k < 0.60:
        return "moderate"
    if k < 0.80:
        return "substantial"
    return "almost perfect"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--llama-csv", type=Path, default=LLAMA_CSV)
    parser.add_argument("--haiku-csv", type=Path, default=HAIKU_CSV)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--first-flip-only", action="store_true",
                        help="Truncate each question to turns up to and including the first flip.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    global LLAMA_CSV, HAIKU_CSV
    LLAMA_CSV = args.llama_csv
    HAIKU_CSV = args.haiku_csv

    df = load_and_merge()

    if args.first_flip_only:
        before = len(df)
        # Filter using Haiku as the reference judge for truncation
        df = filter_first_flip(df, judge_col="judgement_haiku")
        print(f"[first-flip filter] {before} → {len(df)} rows (dropped {before - len(df)} post-flip turns)")

    report = build_report(df)

    print(report)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report + "\n", encoding="utf-8")
    print(f"\nReport saved to {args.output}")


if __name__ == "__main__":
    main()
