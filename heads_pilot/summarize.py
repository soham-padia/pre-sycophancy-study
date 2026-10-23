#!/usr/bin/env python3
"""Final stage of the heads pilot (CPU): one report of what steering did, with its controls.

Reads the stage-1 generations and labels, the head ranking, and every steering condition with its
labels, and writes `<gen>/RESULTS.json` and `<gen>/RESULTS.md`. Exploratory: the pilot uses questions
whose results the project has already seen, so nothing here is a confirmatory finding.

Per steering condition and arm (steering half, odd qids): ever-flip rate with a Wilson 95% CI, the
paired change against the baseline condition on the same questions (discordant counts and a bootstrap
95% CI over questions), mean response length, the share of responses at the token budget, and the
share with a word repeated three times in a row (degeneracy). Also: whether the steering baseline
reproduced stage 1 exactly, and T0 correctness when steering starts at T0.
"""
from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

import numpy as np

TRIPLE = re.compile(r"\b(\w+) \1 \1\b")
SEED = 20261021


def wilson(k: int, n: int) -> list[float]:
    if n == 0:
        return [float("nan")] * 2
    z, p = 1.96, k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return [round(c - h, 4), round(c + h, 4)]


def read_jsonl(folder: Path, pattern: str) -> list[dict]:
    return [json.loads(l) for f in sorted(folder.glob(pattern)) for l in open(f, encoding="utf-8")]


def ever_flip(judge_dir: Path, arm: str) -> dict[int, bool]:
    out = {}
    rows = [json.loads(l) for f in sorted(judge_dir.glob(f"{arm}.*jsonl")) if ".t0." not in f.name
            for l in open(f, encoding="utf-8")]
    for r in rows:
        if r.get("labels"):
            out[r["qid"]] = any(r["labels"][str(t)]["judgement"] for t in range(1, 6))
    return out


def t0_correct(judge_dir: Path, arm: str) -> dict[int, str]:
    return {r["qid"]: r["labels"]["t0"] for r in read_jsonl(judge_dir, f"{arm}.t0*.jsonl") if r.get("labels")}


def text_stats(convs: list[dict], first_turn: int) -> dict:
    resp = [t for c in convs for t in c["turns"] if t["t"] >= first_turn]
    if not resp:
        return {}
    return {"n_responses": len(resp), "mean_tokens": round(float(np.mean([t["n_tokens"] for t in resp])), 1),
            "capped_share": round(sum(not t["finished"] for t in resp) / len(resp), 4),
            "triple_repeat_share": round(sum(bool(TRIPLE.search(t["response"])) for t in resp) / len(resp), 4)}


def paired(a: dict[int, bool], b: dict[int, bool], rng) -> dict:
    q = sorted(set(a) & set(b))
    if not q:
        return {}
    x = np.array([a[i] for i in q], float); y = np.array([b[i] for i in q], float)
    boots = [float(np.mean(y[s] - x[s])) for s in (rng.integers(0, len(q), len(q)) for _ in range(5000))]
    return {"n": len(q), "baseline_rate": round(float(x.mean()), 4), "rate": round(float(y.mean()), 4),
            "delta": round(float((y - x).mean()), 4), "delta_ci95": [round(float(np.quantile(boots, 0.025)), 4),
                                                                      round(float(np.quantile(boots, 0.975)), 4)],
            "flip_to_hold": int(((x == 1) & (y == 0)).sum()), "hold_to_flip": int(((x == 0) & (y == 1)).sum())}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gen-dir", type=Path,
                    default=Path("/work/neu/p2026_0037_neu/heads_pilot/Qwen2.5-7B-Instruct/base"))
    ap.add_argument("--k", type=int, default=16)
    args = ap.parse_args()
    rng = np.random.default_rng(SEED)
    g = args.gen_dir
    res = {"note": "exploratory pilot on already-seen questions; not a confirmatory result"}

    stage1 = {arm: ever_flip(g / "judge", arm) for arm in ("disagree", "agree")}
    res["stage1"] = {arm: {"n": len(v), "ever_flip": sum(v.values()), "rate_ci95": wilson(sum(v.values()), len(v))}
                     for arm, v in stage1.items()}
    t0 = t0_correct(g / "judge", "disagree")
    res["stage1"]["t0"] = {k: sum(v == k for v in t0.values()) for k in ("correct", "incorrect", "unclear")}
    ranking = json.loads((g / "heads" / "ranking.json").read_text())
    res["ranking"] = {k: ranking[k] for k in ("top_heads", "observed_top16_mean", "null_within_turn_shuffle",
                                              "surface_baselines_stratified_auc", "rows_per_turn")}
    res["ranking"]["top_heads"] = ranking["top_heads"][: args.k]
    res["ranking"]["random_control"] = ranking["random_control"][str(args.k)]

    sdir = g / "steer" / f"k{args.k}"
    stage1_conv = {(c["arm"], c["qid"]): c for c in read_jsonl(g, "*.shard*.jsonl")}
    res["steering"] = {}
    for cond_dir in sorted(p for p in sdir.iterdir() if p.is_dir()) if sdir.exists() else []:
        name = cond_dir.name
        entry = {}
        for arm in ("disagree", "agree"):
            convs = read_jsonl(cond_dir, f"{arm}.shard*.jsonl")
            flips = ever_flip(cond_dir / "judge", arm)
            entry[arm] = {"n": len(flips), "ever_flip": sum(flips.values()),
                          "rate_ci95": wilson(sum(flips.values()), len(flips)), "text": text_stats(convs, 1)}
            if name != "baseline" and (sdir / "baseline" / "judge").exists():
                entry[arm]["vs_baseline"] = paired(ever_flip(sdir / "baseline" / "judge", arm), flips, rng)
            if name == "baseline":
                same = sum(all(t["response"] == stage1_conv[(arm, c["qid"])]["turns"][t["t"]]["response"]
                               for t in c["turns"]) for c in convs if (arm, c["qid"]) in stage1_conv)
                entry[arm]["reproduces_stage1"] = f"{same}/{len(convs)} conversations identical"
        res["steering"][name] = entry

    t0dir = g / "steer" / f"k{args.k}_t0"
    res["t0_steering"] = {}
    for cond_dir in sorted(p for p in t0dir.iterdir() if p.is_dir()) if t0dir.exists() else []:
        lab = t0_correct(cond_dir / "judge", "disagree")
        res["t0_steering"][cond_dir.name] = {k: sum(v == k for v in lab.values()) for k in ("correct", "incorrect", "unclear")}
        res["t0_steering"][cond_dir.name]["text"] = text_stats(read_jsonl(cond_dir, "disagree.shard*.jsonl"), 0)

    (g / "RESULTS.json").write_text(json.dumps(res, indent=1) + "\n")
    lines = ["# Heads pilot results (exploratory)", "", f"Model Qwen2.5-7B-Instruct, base prompt, k = {args.k} heads.", ""]
    s1 = res["stage1"]
    lines += [f"Stage 1: gave in at least once in {s1['disagree']['ever_flip']}/{s1['disagree']['n']} pushback "
              f"conversations and {s1['agree']['ever_flip']}/{s1['agree']['n']} agreement conversations; "
              f"T0 {s1['t0']}.", ""]
    nul = res["ranking"]["null_within_turn_shuffle"]
    lines += [f"Ranking: top-16 mean flip AUC {res['ranking']['observed_top16_mean']} (shuffled-label 95th "
              f"percentile {nul['top16_mean_p95']}, p = {round(nul['p_top16_mean'], 3)}); surface baselines "
              f"{res['ranking']['surface_baselines_stratified_auc']}.", "", "| condition | arm | gave in | 95% CI | "
              "change vs baseline [95% CI] | mean tokens | triple-repeat |", "|---|---|---|---|---|---|---|"]
    for name, e in res["steering"].items():
        for arm in ("disagree", "agree"):
            a = e[arm]; v = a.get("vs_baseline", {})
            ch = f"{v['delta']:+.3f} [{v['delta_ci95'][0]:+.3f}, {v['delta_ci95'][1]:+.3f}]" if v else "—"
            lines.append(f"| {name} | {arm} | {a['ever_flip']}/{a['n']} | {a['rate_ci95']} | {ch} | "
                         f"{a['text'].get('mean_tokens', '—')} | {a['text'].get('triple_repeat_share', '—')} |")
    if "baseline" in res["steering"]:
        lines += ["", "Baseline reproduces stage 1: " + ", ".join(
            f"{arm} {res['steering']['baseline'][arm].get('reproduces_stage1')}" for arm in ("disagree", "agree"))]
    if res["t0_steering"]:
        lines += ["", "T0 correctness when steering starts at T0: " + "; ".join(
            f"{k}: {v['correct']} correct / {v['incorrect']} incorrect / {v['unclear']} unclear"
            for k, v in res["t0_steering"].items())]
    (g / "RESULTS.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
