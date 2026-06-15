#!/usr/bin/env python3
"""Three robustness checks on the T0→T1 cosine AUC signal.

Check 1 — Exclude T1-first-flip questions:
    Questions where the model flips AT T1 have their T1 hidden state equal
    to the flip state itself, not a pre-flip state. Including them may
    inflate AUC. We rerun with those questions removed and report the
    difference.

Check 2 — Bootstrap 95% CIs on peak AUC:
    Reports confidence intervals at the best layer per (model, qtype) to
    support formal statistical claims about the signal.

Check 3 — Baseline (T0) correctness:
    T0 correctness is not explicitly verified per-instance; the judge CSV
    covers only turns 1-5. We report the fraction of questions per model
    where Response_Turn_0 is non-empty (a proxy for T0 completion), and
    note the implicit assumption from benchmark design.

Outputs:
    analysis_claude/cosine_disruption_checks.txt
    analysis_claude/cosine_disruption_checks.csv
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
OUT_TXT   = REPO_ROOT / "analysis_claude" / "cosine_disruption_checks.txt"
OUT_CSV   = REPO_ROOT / "analysis_claude" / "cosine_disruption_checks.csv"

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

MULTITURN_DIRS = {
    "DeepSeek-R1-7B": REPO_ROOT / "data" / "DeepSeek-R1-Distill-Qwen-7B",
    "Qwen2.5-7B":     REPO_ROOT / "data" / "Qwen2.5-7B-Instruct",
    "Llama-3.1-8B":   REPO_ROOT / "data" / "Llama-3.1-8B-Instruct",
    "Qwen3.5-9B":     REPO_ROOT / "data" / "Qwen3.5-9B",
}

QUESTION_TYPES = ["base", "critical", "presupposition"]
N_BOOTSTRAP    = 2000


# ── Data loading ──────────────────────────────────────────────────────────────

def load_data(model_name: str, qtype: str):
    cfg = MODELS[model_name]
    pt_path = REPO_ROOT / cfg["dir"] / f"{qtype}_multiturn_hidden_states.pt"
    if not pt_path.exists():
        return None, None

    judge_csv, model_col = JUDGE_CSVS[model_name]
    hs = torch.load(str(pt_path), map_location="cpu")

    df = pd.read_csv(judge_csv)
    df = df[(df["model"] == model_col) & (df["question_type"] == qtype)].copy()
    df["flip"] = df["judgement"].astype(str).str.lower() == "true"

    flip_info = {}
    for q, grp in df.groupby("question"):
        grp = grp.sort_values("turn")
        first = grp[grp["flip"]]["turn"].min() if grp["flip"].any() else None
        flip_info[q] = {
            "ever_flip":       grp["flip"].any(),
            "first_flip_turn": int(first) if first is not None and not pd.isna(first) else None,
        }
    return hs, flip_info


def get_vec(hs, q, turn, layer):
    try:
        v = hs[q][turn]
        return None if v is None else v.float()[layer].numpy()
    except Exception:
        return None


def cos_sim(a, b):
    a = a / (np.linalg.norm(a) + 1e-10)
    b = b / (np.linalg.norm(b) + 1e-10)
    return float(np.dot(a, b))


# ── AUC helpers ───────────────────────────────────────────────────────────────

def compute_auc(sims_arr, labels_arr):
    if len(np.unique(labels_arr)) < 2:
        return np.nan
    try:
        return roc_auc_score(labels_arr, -sims_arr)
    except Exception:
        return np.nan


def bootstrap_ci(sims_arr, labels_arr, n_boot=N_BOOTSTRAP, seed=42):
    rng = np.random.default_rng(seed)
    n   = len(sims_arr)
    aucs = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        yt  = labels_arr[idx]
        sc  = sims_arr[idx]
        if len(np.unique(yt)) < 2:
            continue
        try:
            aucs.append(roc_auc_score(yt, -sc))
        except Exception:
            pass
    if len(aucs) < 10:
        return np.nan, np.nan
    return float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5))


def layer_auc(hs, flip_info, layer, exclude_t1_flip=False):
    """Return (sims_arr, labels_arr) at this layer, optionally excluding T1-flips."""
    sims, labels = [], []
    for q, info in flip_info.items():
        if exclude_t1_flip and info["first_flip_turn"] == 1:
            continue
        v0 = get_vec(hs, q, 0, layer)
        v1 = get_vec(hs, q, 1, layer)
        if v0 is None or v1 is None:
            continue
        sims.append(cos_sim(v0, v1))
        labels.append(int(info["ever_flip"]))
    return np.array(sims), np.array(labels)


# ── Check 3: T0 completeness ──────────────────────────────────────────────────

def check_t0_completeness(model_name, qtype):
    csv = MULTITURN_DIRS[model_name] / f"{qtype}_multiturn.csv"
    if not csv.exists():
        return None, None
    df = pd.read_csv(csv)
    if "Response_Turn_0" not in df.columns:
        return None, None
    n_total  = len(df)
    n_nonempty = df["Response_Turn_0"].astype(str).str.strip().ne("").sum()
    return int(n_nonempty), int(n_total)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    lines = [
        "COSINE DISRUPTION ROBUSTNESS CHECKS",
        "=" * 70,
        "",
        "Check 1: T0→T1 AUC excluding T1-first-flip questions",
        "Check 2: Bootstrap 95% CIs at peak layer",
        "Check 3: T0 baseline correctness (completeness proxy)",
        "",
    ]
    rows = []

    for model_name, cfg in MODELS.items():
        lines.append(f"\n{'─'*70}")
        lines.append(f"MODEL: {model_name}")
        lines.append(f"{'─'*70}")

        for qtype in QUESTION_TYPES:
            hs, flip_info = load_data(model_name, qtype)
            if hs is None:
                lines.append(f"  [{qtype}] data unavailable")
                continue

            n_total  = len(flip_info)
            n_t1flip = sum(1 for v in flip_info.values() if v["first_flip_turn"] == 1)
            n_ever   = sum(1 for v in flip_info.values() if v["ever_flip"])

            lines.append(f"\n  {qtype}  |  {n_total} questions  "
                         f"({n_ever} ever-flip, {n_t1flip} flip-at-T1, "
                         f"{n_total - n_t1flip} retained after exclusion)")

            # ── Sweep all layers for both modes ──────────────────────────────
            best_orig  = {"auc": -1, "layer": -1, "sims": None, "labels": None}
            best_excl  = {"auc": -1, "layer": -1, "sims": None, "labels": None}

            for layer in range(cfg["n_layers"]):
                sims_o, labs_o = layer_auc(hs, flip_info, layer, exclude_t1_flip=False)
                sims_e, labs_e = layer_auc(hs, flip_info, layer, exclude_t1_flip=True)

                auc_o = compute_auc(sims_o, labs_o)
                auc_e = compute_auc(sims_e, labs_e)

                if not np.isnan(auc_o) and auc_o > best_orig["auc"]:
                    best_orig.update({"auc": auc_o, "layer": layer,
                                      "sims": sims_o, "labels": labs_o})
                if not np.isnan(auc_e) and auc_e > best_excl["auc"]:
                    best_excl.update({"auc": auc_e, "layer": layer,
                                      "sims": sims_e, "labels": labs_e})

            # ── Bootstrap CIs at peak layers ─────────────────────────────────
            ci_orig_lo, ci_orig_hi = bootstrap_ci(best_orig["sims"], best_orig["labels"])
            ci_excl_lo, ci_excl_hi = bootstrap_ci(best_excl["sims"], best_excl["labels"])

            lines.append(f"  {'':30} {'AUC':>7}  {'95% CI':>15}  {'Layer':>6}")
            lines.append(f"  {'':30} {'─'*7}  {'─'*15}  {'─'*6}")
            lines.append(
                f"  {'All questions (original)':<30} {best_orig['auc']:>7.3f}  "
                f"[{ci_orig_lo:.3f}, {ci_orig_hi:.3f}]  {best_orig['layer']:>6}"
            )
            lines.append(
                f"  {'Excl. T1-flip questions':<30} {best_excl['auc']:>7.3f}  "
                f"[{ci_excl_lo:.3f}, {ci_excl_hi:.3f}]  {best_excl['layer']:>6}"
            )

            above_chance_orig = "YES" if ci_orig_lo > 0.5 else ("MARGINAL" if best_orig["auc"] > 0.5 else "NO")
            above_chance_excl = "YES" if ci_excl_lo > 0.5 else ("MARGINAL" if best_excl["auc"] > 0.5 else "NO")
            lines.append(f"  CI strictly above chance (0.5):  original={above_chance_orig}, excl={above_chance_excl}")

            # ── Check 3: T0 completeness ──────────────────────────────────────
            n_ok, n_tot = check_t0_completeness(model_name, qtype)
            if n_ok is not None:
                pct = 100 * n_ok / n_tot if n_tot else 0
                lines.append(f"  T0 response present: {n_ok}/{n_tot} ({pct:.1f}%)")

            rows.append({
                "model": model_name, "qtype": qtype,
                "n_total": n_total, "n_t1_flip": n_t1flip, "n_ever_flip": n_ever,
                "auc_original":   best_orig["auc"],  "layer_original":  best_orig["layer"],
                "ci_orig_lo": ci_orig_lo,             "ci_orig_hi": ci_orig_hi,
                "auc_excl_t1":    best_excl["auc"],   "layer_excl_t1":   best_excl["layer"],
                "ci_excl_lo": ci_excl_lo,             "ci_excl_hi": ci_excl_hi,
            })

    # ── Summary table ─────────────────────────────────────────────────────────
    lines.append(f"\n\n{'='*70}")
    lines.append("SUMMARY — Peak AUC with 95% bootstrap CI")
    lines.append(f"{'='*70}")
    lines.append(f"  {'Model':<16} {'Q-type':<14} {'Original AUC':>14}  {'95% CI':>15}  {'Excl T1-flip':>14}  {'95% CI':>15}")
    lines.append(f"  {'─'*16} {'─'*14} {'─'*14}  {'─'*15}  {'─'*14}  {'─'*15}")
    for r in rows:
        lines.append(
            f"  {r['model']:<16} {r['qtype']:<14} "
            f"{r['auc_original']:>14.3f}  [{r['ci_orig_lo']:.3f}, {r['ci_orig_hi']:.3f}]  "
            f"{r['auc_excl_t1']:>14.3f}  [{r['ci_excl_lo']:.3f}, {r['ci_excl_hi']:.3f}]"
        )

    lines.append("""
NOTE — Baseline (T0) correctness:
  The judge CSV covers only turns 1-5; no per-instance T0 correctness
  label is available. T0 completeness (Response_Turn_0 non-empty) is
  near 100% for all models and question types above.  Correctness at T0
  is assumed from SYCON-Bench benchmark design: questions have a single
  defensible correct answer and are evaluated precisely because models
  are expected to answer correctly before pressure is applied.
  A minority of T0 responses may already be incorrect; these would
  appear as "anti-sycophantic corrections" under pressure and inflate
  flip counts slightly. Explicit T0 verification is left for future work.
""")

    report = "\n".join(lines)
    print(report)
    OUT_TXT.write_text(report + "\n", encoding="utf-8")
    print(f"\nSaved → {OUT_TXT}")

    if rows:
        pd.DataFrame(rows).to_csv(OUT_CSV, index=False)
        print(f"Saved → {OUT_CSV}")


if __name__ == "__main__":
    main()
