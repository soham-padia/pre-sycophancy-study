# Agent Notes — append-only session log

Convention: append a dated entry at the END of this file after any session that
changed something. Never edit or delete previous entries. Read the whole log
before starting work.

Template:

```markdown
## YYYY-MM-DD — <name / agent> (branch: <branch>)
- What changed:
- Files touched:
- Blocked on / needs review:
- Handoff notes:
```

---

## 2026-06-30 — Soham + Claude (branch: main)
- What changed:
  - Full code audit of the analysis pipeline. Three findings mattered:
    (1) base/critical/presupposition are system-prompt conditions on the same
    ~177 questions, not question types — paper §3.1 rewritten accordingly;
    (2) off-by-one in `plot_preflip_pca.py` and `plot_layer_sweep.py` (paired
    state-at-t with label-at-t instead of t+1) — fixed, figures regenerated,
    probe-failure result got *cleaner* (max acc = majority baseline exactly,
    all models); (3) Gemma base hidden-state file corrupt, critical/presup
    intact and unanalyzed.
  - Robustness checks added (`analysis/cosine_disruption_checks.py`):
    T1-exclusion (AUC survives) + bootstrap CIs + T0 completeness (100%).
  - Multi-turn AUC 0.5-floor bug fixed (`multiturn_auc_robustness.py`).
  - Paper revision pass applied in Overleaf tex (see `PAPER_STATUS.md` for the
    full done/remaining list vs. the professor's letter).
  - `download_data.py` added (HF: `sohampadianeu/pre-sycophancy-study-data`).
  - This `_markdown/` hub created.
- Files touched: `analysis/{multiturn_auc_robustness,plot_layer_sweep,plot_preflip_pca}.py`,
  `analysis_claude/{hidden_state_disruption,layer_sweep,preflip_pca}.png`,
  `analysis_claude/layer_sweep_accuracy.csv`, `_markdown/*` (commits 35a2478…b40e3ff+)
- Blocked on / needs review: nothing blocked. Remaining paper items in
  `PAPER_STATUS.md` §Remaining (permutation test is the priority; figure
  redesigns are Track 3).
- Handoff notes:
  - Regenerated figures need re-upload to Overleaf `new_figures/`
    (`hidden_state_disruption.png`, `preflip_pca.png`).
  - Two bib entries pending in Overleaf (see `PAPER_STATUS.md`).
  - Anyone writing new probe analyses: use the pre-flip pairing from
    `train_probes_v2.py::build_preflip_dataset` — see ONBOARDING "off-by-one trap".

## 2026-07-08 — Soham + Claude (branch: main)
- What changed: Table 2 flip-rate 95% binomial CIs generated
  (`analysis/flip_rate_cis.py` → `analysis_claude/flip_rate_cis.{csv,tex}`);
  Overleaf Table 2 rows + caption updated with ± values. Point estimates
  verified identical to the published table before adding CIs.
- Files touched: analysis/flip_rate_cis.py, analysis_claude/flip_rate_cis.*,
  Overleaf main.tex (Table 2)
- Blocked on / needs review: nothing
- Handoff notes: permutation test (PAPER_STATUS item 1) still the top remaining
  stats item; figure redesigns (items 3-4) unassigned in-flight for Track 3.

## 2026-07-08 (later) — Soham + Claude (branch: main)
- What changed: Vedant's CI audit items completed —
  (1) per-layer bootstrap CI bands for cosine AUC (`analysis/cosine_auc_ci_bands.py`
  → `analysis_claude/cosine_auc_ci_bands.csv`, best condition per model, 1000 resamples);
  (2) multi-turn AUC figure now has 95% bootstrap error bars at peak layer
  (`multiturn_auc_robustness.py` regenerated);
  (3) Table 4 Fisher/LDA CIs (`analysis/geometry_cis.py` → `geometry_cis.{csv,tex}`;
  Fisher point estimates reproduce Table 4 exactly, LDA within ±0.007 — original
  script's PCA lacked a random_state; Table 4 in tex updated with regenerated values + CIs).
- Note for whoever merges Tomas's PR #1 (paper-jul-6): its base tex predates the
  final PCA caption fix, §6.2 "pre-flip and hold" wording, Table 2 ±CIs, and the
  Table 4 CI update — sync those after merge. Panel C caption ("3/4 reach 0.5")
  conflicts with the paper's committed CI values — DeepSeek/Llama lower bounds
  hover at 0.50 and are seed-sensitive; caption should match the text's framing
  (Qwen2.5 clearly above; Llama marginal; DeepSeek/Qwen3.5 straddle).
- Handoff: permutation test with layer-sweep correction is the last stats item.

## 2026-07-08 (fact-check + permutation) — Soham + Claude (branch: main)
- Permutation test done (`analysis/permutation_test.py`, 10k shuffles, full layer
  sweep per shuffle): null median peaks 0.56–0.60; ONLY Qwen2.5 base survives
  (p=0.035); Fisher-combined across best conditions p=0.047 (condition selection
  uncorrected). §6.1 + Limitations updated — peaks are now framed as descriptive.
- Full fact-check of paper claims vs data. VERIFIED: avg first-flip turns, 76%
  T4 Qwen2.5, all peak AUC/layer cells, T1-exclusion deltas, bootstrap CIs,
  flip-rate ranges + keyword-judge gaps, all §6.3 classifier claims (2.5pp,
  80.0/77.5, 78.3/75.9, 6.6pp avg, 13.3/8.8pp), multiturn DS 0.725 + Q2.5
  stability, trajectory example question, Fisher range, 11,450 judge calls.
- ERRORS FOUND AND FIXED in tex:
  1. §4.5 "judge concentrates flips at T1 across all models" — false (Q2.5 3%,
     Q3.5 1%); keyword actually fires EARLIER than judge; paragraph rewritten.
  2. §4.5 Qwen3.5 keyword>judge exception — only base/critical; presup is the
     reverse; caveat added.
  3. §3.5 imbalance "75–85%" and §3.6 chance "68–91%" — actual 66–91%; fixed.
  4. Appendix multiturn: exception is LLAMA (not Qwen3.5); Q3.5 peaks at T0→T3.
  5. Appendix L2: "flip group slightly higher" — false (lower for DS/Q2.5);
     rewritten to match l2_norm_trend.csv.
- FOR TOMAS'S PR: Panel C caption ("3/4 reach 0.5") is seed-sensitive and
  contradicts paper text; appendix Fig2 caption "keyword labels undercount
  flips" is false for Qwen3.5 base/critical (overcounts by 21–22pp).
