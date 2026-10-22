#!/usr/bin/env python3
"""Stage 3b of the heads pilot (CPU): rank attention heads on the selection half of the questions.

Selection half = even qids; the odd half is kept for steering (stage 4) and never used here.

Per head, from its pre-response output z (stage 3a):
- **flip score**: does z separate "the next response gives in" from "it holds"? Disagree arm only,
  next-response horizon (state before response t, label of response t, t = 1-5, censored after the
  first flip). z is centred within each turn and the AUC is computed within turn strata, so a head that
  merely tracks turn position scores 0.5 (the confound of ERRATA E3). Direction = mean(flip) - mean(hold)
  fit on training folds, AUC on held-out questions (5-fold by question).
- **pressure score**: d' between disagree and agree z at the same question and turn (t = 1-5), direction
  fit on training folds, measured on held-out questions.

Surface baselines use the same rows, folds, within-turn centring and stratified AUC: what a reader of
the transcript could see before response t (length and hedge density of response t-1 and of T0),
singly and combined. Heads that do not beat these are not evidence of anything internal.

Writes `<gen>/heads/ranking.json`: per-head scores, the top-k heads by flip score, layer-matched
random heads for the control, and the surface baselines.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import torch

SEED = 20261021
HEDGES = ["might", "may", "perhaps", "possibly", "possible", "could be", "it seems", "i think", "i believe",
          "not entirely", "not necessarily", "generally", "typically", "often", "likely", "unlikely",
          "uncertain", "i'm not sure", "it depends", "i apologize", "apologies", "sorry"]
HEDGE_RE = re.compile(r"\b(" + "|".join(re.escape(h) for h in HEDGES) + r")\b", re.I)


def hedge_density(text: str) -> float:
    words = max(1, len(text.split()))
    return 100.0 * len(HEDGE_RE.findall(text)) / words


def load_labels(judge_dir: Path, arm: str) -> dict[int, list[bool | None]]:
    out = {}
    for f in sorted(judge_dir.glob(f"{arm}*.jsonl")):
        for line in open(f, encoding="utf-8"):
            r = json.loads(line)
            lab = r["labels"]
            out[r["qid"]] = None if lab is None else [lab[str(t)]["judgement"] for t in range(1, 6)]
    return out


def stratified_auc(scores: np.ndarray, y: np.ndarray, strata: np.ndarray) -> float:
    """AUC within each stratum, pooled by its number of positive-negative pairs."""
    num = den = 0.0
    for s in np.unique(strata):
        m = strata == s
        pos, neg = scores[m & (y == 1)], scores[m & (y == 0)]
        if len(pos) == 0 or len(neg) == 0:
            continue
        diff = pos[:, None] - neg[None, :]
        num += (diff > 0).sum() + 0.5 * (diff == 0).sum()
        den += diff.size
    return float(num / den) if den else float("nan")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gen-dir", type=Path,
                    default=Path("/work/neu/p2026_0037_neu/heads_pilot/Qwen2.5-7B-Instruct/base"))
    ap.add_argument("--ks", type=int, nargs="+", default=[8, 16, 32])
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--null-perms", type=int, default=100)
    args = ap.parse_args()
    rng = np.random.default_rng(SEED)

    dis = torch.load(args.gen_dir / "heads" / "disagree.pt")
    agr = torch.load(args.gen_dir / "heads" / "agree.pt")
    assert dis["qids"] == agr["qids"]
    qids = np.array(dis["qids"])
    zd, za = dis["z"].float().numpy(), agr["z"].float().numpy()   # [n, 6, L, H, d]
    n, _, L, H, d = zd.shape
    labels = load_labels(args.gen_dir / "judge", "disagree")
    agree_labels = load_labels(args.gen_dir / "judge", "agree")

    sel = np.array([q % 2 == 0 and labels.get(q) is not None for q in qids])
    # Pre-flip rows (next-response horizon): (conversation index, turn t), label = response t flips.
    rows = []
    for i in np.where(sel)[0]:
        lab = labels[qids[i]]
        for t in range(1, 6):
            rows.append((i, t, int(lab[t - 1])))
            if lab[t - 1]:
                break
    rows = np.array(rows)
    ci, tt, y = rows[:, 0], rows[:, 1], rows[:, 2]
    conv_ids = np.unique(ci)
    fold_of = {c: k for k, c in enumerate(rng.permutation(conv_ids))}
    fold = np.array([fold_of[c] % args.folds for c in ci])

    X = zd[ci, tt]                                        # [rows, L, H, d]
    Xc = X.copy()
    for t in np.unique(tt):                                # centre within turn
        Xc[tt == t] -= X[tt == t].mean(axis=0, keepdims=True)

    def flip_scores(yy: np.ndarray) -> np.ndarray:
        proj = np.zeros((len(yy), L, H))
        for k in range(args.folds):
            tr, te = fold != k, fold == k
            w = Xc[tr & (yy == 1)].mean(0) - Xc[tr & (yy == 0)].mean(0)      # [L, H, d]
            w /= np.linalg.norm(w, axis=-1, keepdims=True) + 1e-8
            proj[te] = np.einsum("rlhd,lhd->rlh", Xc[te], w)
        return np.array([[stratified_auc(proj[:, l, h], yy, tt) for h in range(H)] for l in range(L)])

    flip_auc = flip_scores(y)
    # Selection null: labels shuffled within turn strata, everything else identical.
    null_top1, null_top16 = [], []
    for _ in range(args.null_perms):
        yp = y.copy()
        for t in np.unique(tt):
            m = tt == t
            yp[m] = rng.permutation(yp[m])
        s = np.sort(flip_scores(yp), axis=None)[::-1]
        null_top1.append(float(s[0]))
        null_top16.append(float(s[:16].mean()))

    # Pressure score: disagree vs agree, same questions (selection half) and turns 1-5.
    si = np.where(sel)[0]
    pf = {c: k for k, c in enumerate(rng.permutation(si))}
    pfold = np.array([pf[c] % args.folds for c in si])
    dprime = np.zeros((L, H))
    for k in range(args.folds):
        tr, te = si[pfold != k], si[pfold == k]
        a = zd[tr][:, 1:].reshape(-1, L, H, d).mean(0) - za[tr][:, 1:].reshape(-1, L, H, d).mean(0)
        a /= np.linalg.norm(a, axis=-1, keepdims=True) + 1e-8
        pd_ = np.einsum("rlhd,lhd->rlh", zd[te][:, 1:].reshape(-1, L, H, d), a)
        pa_ = np.einsum("rlhd,lhd->rlh", za[te][:, 1:].reshape(-1, L, H, d), a)
        dprime += (pd_.mean(0) - pa_.mean(0)) / np.sqrt(0.5 * (pd_.var(0) + pa_.var(0)) + 1e-8) / args.folds

    order = np.argsort(-flip_auc, axis=None)
    ranked = [(int(i // H), int(i % H)) for i in order]
    selected, random_ctrl = {}, {}
    for k in args.ks:
        top = ranked[:k]
        selected[k] = top
        taken = set(top)
        ctrl = []
        for l, _ in top:                                   # same layers, different heads
            choices = [h for h in range(H) if (l, h) not in taken]
            h = int(rng.choice(choices))
            taken.add((l, h))
            ctrl.append((l, h))
        random_ctrl[k] = ctrl

    # Surface baselines on the same rows and folds.
    convs = {json.loads(l)["qid"]: json.loads(l) for f in sorted(args.gen_dir.glob("disagree.shard*.jsonl"))
             for l in open(f, encoding="utf-8")}
    feats = {"prev_len": [], "prev_hedge": [], "t0_len": [], "t0_hedge": []}
    for i, t in zip(ci, tt):
        turns = convs[int(qids[i])]["turns"]
        feats["prev_len"].append(turns[t - 1]["n_tokens"])
        feats["prev_hedge"].append(hedge_density(turns[t - 1]["response"]))
        feats["t0_len"].append(turns[0]["n_tokens"])
        feats["t0_hedge"].append(hedge_density(turns[0]["response"]))
    F = np.column_stack([np.asarray(v, dtype=float) for v in feats.values()])
    for t in np.unique(tt):
        F[tt == t] -= F[tt == t].mean(0)
    F /= F.std(0) + 1e-8

    def cv_auc(M: np.ndarray) -> float:
        out = np.zeros(len(y))
        for k in range(args.folds):
            tr, te = fold != k, fold == k
            w = M[tr & (y == 1)].mean(0) - M[tr & (y == 0)].mean(0)
            out[te] = M[te] @ w
        return stratified_auc(out, y, tt)

    surface = {name: round(cv_auc(F[:, [j]]), 4) for j, name in enumerate(feats)}
    surface["combined"] = round(cv_auc(F), 4)

    def rate(lab_dict, keep):
        vals = [any(v) for q, v in lab_dict.items() if v is not None and keep(q)]
        return (sum(vals), len(vals))

    report = {
        "selection_half": "even qids", "steering_half": "odd qids", "seed": SEED,
        "n_selection_conversations": int(sel.sum()), "n_rows": int(len(y)), "n_flip_rows": int(y.sum()),
        "rows_per_turn": {int(t): [int(((tt == t) & (y == 1)).sum()), int(((tt == t) & (y == 0)).sum())]
                          for t in range(1, 6)},
        "ever_flip_disagree": rate(labels, lambda q: True),
        "ever_flip_agree": rate(agree_labels, lambda q: True),
        "flip_auc": flip_auc.round(4).tolist(), "pressure_dprime": dprime.round(3).tolist(),
        "top_heads": [{"layer": l, "head": h, "flip_auc": round(float(flip_auc[l, h]), 4),
                       "pressure_dprime": round(float(dprime[l, h]), 3)} for l, h in ranked[:40]],
        "selected": {str(k): v for k, v in selected.items()},
        "random_control": {str(k): v for k, v in random_ctrl.items()},
        "surface_baselines_stratified_auc": surface,
        "flip_auc_quantiles": {q: round(float(np.quantile(flip_auc, q)), 4) for q in (0.5, 0.9, 0.99, 1.0)},
        "observed_top1": round(float(np.sort(flip_auc, axis=None)[-1]), 4),
        "observed_top16_mean": round(float(np.sort(flip_auc, axis=None)[-16:].mean()), 4),
        "null_within_turn_shuffle": {
            "n_perms": args.null_perms,
            "top1_p95": round(float(np.quantile(null_top1, 0.95)), 4) if null_top1 else None,
            "top16_mean_p95": round(float(np.quantile(null_top16, 0.95)), 4) if null_top16 else None,
            "p_top16_mean": (1 + sum(v >= float(np.sort(flip_auc, axis=None)[-16:].mean()) for v in null_top16))
                            / (1 + len(null_top16)),
        },
    }
    (args.gen_dir / "heads" / "ranking.json").write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps({k: report[k] for k in ["n_rows", "n_flip_rows", "rows_per_turn", "ever_flip_disagree", "observed_top1", "observed_top16_mean", "null_within_turn_shuffle", "surface_baselines_stratified_auc",
                                              "ever_flip_agree", "flip_auc_quantiles"]}, indent=1))
    print("top 10:", report["top_heads"][:10])


if __name__ == "__main__":
    main()
