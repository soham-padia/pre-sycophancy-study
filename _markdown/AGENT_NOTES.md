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

## 2026-07-08 (PR merge) — Soham + Claude (branch: main)
- Did Tomas's PR fixes ourselves and merged paper-jul-6 PR #1 (merge commit
  a4457ce): resolved 3 tex conflicts (kept corrected §4.5 text + his appendix
  ref; his 4-panel caption with Panel C aligned to the permutation result; my
  LDA-CI wording + his appendix ref), fixed the appendix comparison caption
  (Qwen3.5 overcounts under keyword), added layer_sweep.png to the linear-probes
  appendix, and replaced all four figure PNGs with versions regenerated from
  committed scripts (analysis repo 96355ad):
  plot_hidden_state_disruption.py is now the 4-panel evidence figure;
  plot_flip_turn_{distribution,comparison}.py are stacked bars;
  cosine_auc_ci_bands.py gained excl_t1 + mean_drift columns.
- Remaining before sending the draft to the professor: Overleaf pull + compile
  check (Table 4 width, new figure* placements), abstract judgment call
  (mention permutation result?), then email the draft.

## 2026-07-16 — Soham + Claude (branch: main)
- Reviewed + merged paper-jul-6 PR #2 (Tomas's figure polish: teaser redesign,
  Fig 2 font sizes, Fig 3 cut to 2 panels with the drift dissociation moved to
  a new appendix). Design and data checked out; permutation sentence survives
  in the new caption.
- Tomas again did not commit figure scripts — regenerated all three PNGs from
  committed scripts instead (analysis repo: plot_hidden_state_disruption.py is
  now the 2-panel version, new plot_drift_vs_auc.py, plot_flip_turn_
  distribution.py at print-size fonts with segment labels).
- Also reviewed 3c4a10c "Utkarsh changes done" (pushed via Soham's account,
  hadn't been checked): fixed abstract subject-verb agreement, missing space +
  hardcoded Table 2 ref, wrong CI-width explanation, and deduplicated the
  authority-cue Discussion paragraph (post-merge commit a96c745).
- Paper repo main: a96c745. Overleaf needs a pull + compile check.

## 2026-07-16 (later) — Soham + Claude (branch: main)
- Human-in-Language reframing pass applied to paper-jul-6 (990c885), per
  Asteria's feedback for the EACL 2027 special theme: intro opens with real
  user interaction; theme lock-ins at abstract/intro/conclusion ends;
  related work trimmed + human-pressure citations (FlipFlop, Truth Decay,
  Asch 1956); labeling framed as user-sees vs user-takes-away; user-cost
  sentence in Discussion; pressure schedule framed as social acts
  "abstracted from" (not replicating) human escalation.
- IMPORTANT deviations from Asteria's suggestions (flagged to her):
  contribution says we EXTEND SYCON-Bench (Hong et al. 2025), not introduce
  it; her "existing metrics fail" closing softened; "cumulative pressure"
  replaced with "social acts" per our own findings.
- Minor: teaser T2 example + whitespace + bold caption; multiturn appendix
  figure fonts bumped to print size (analysis repo updated).
- Timeline per EACL 2027 CFP: ARR deadline Aug 3, 2026; ALL authors must
  register as reviewers by Aug 5; commitment Oct 11.

## 2026-07-27 — Soham + Claude (branch: main)
- Reviewed Utkarsh's length-reduction PR #3 (82383fd, merged Jul 20) + the
  Jul 27 Overleaf sync (e2e258d). The cuts are largely good (abstract
  condensation is solid, dialogue framing survives in compressed form),
  BUT they reintroduced three previously-fixed problems, all now re-fixed
  in 13841a5:
  1. Contribution 1 regressed to "We introduce ... benchmark that
     replicates how users..." — the misattribution AND overclaim we
     deliberately avoided. Restored: extend SYCON-Bench (cite), "closer to".
  2. Broken citation keys (hong-etal-2025-measuring,
     perez-etal-2023-discovering) returned — would render as "?".
  3. FlipFlop/TruthDecay/Asch citations were cut entirely; worse, the new
     intro claimed users escalate across turns citing perez/sharma (which
     don't show that). Swapped cites; restored one grounding clause in
     Related Work.
  Also: tab:trajectory was orphaned (float with no \ref) — re-referenced;
  teaser em-dash restored.
- LESSON for whoever edits next: before cutting or rewriting, grep
  AGENT_NOTES for "deviations" — the SYCON attribution and the citation
  keys have now regressed twice. Do not use "introduce/replicates" wording.
- Still true post-cuts: permutation test, T1-exclusion, all CI tables,
  workspace paragraph, baseline-position para survive. Overleaf needs pull.

## 2026-07-27 (TA feedback round) — Soham + Claude (branch: main)
- Asteria's review (of a stale pre-13841a5 compile) flagged: "?" citations
  (already fixed on main — Overleaf must pull before recompiling), sentence-
  level bloat, and Human-in-Language framing missing from Secs 3-6 (largely
  because Utkarsh's length cuts removed the July 16 threading).
- Applied in cd7bd96 (net -7 lines): her example cuts (restating sentence in
  Related Work; findings detail out of Related Work para 3 and out of §3.1;
  Table 2 walkthrough -> three takeaways) + compact theme threading (human
  dialogue not single-turn; schedule mirrors human social acts; conditions as
  cautious-user instructions; labeling as two human views of a conversation;
  §6 traces invisible in the language a user reads).
- Also this session (3a3f0d7, ae49fd9): all em dashes removed per Soham;
  §7.1 merged into Discussion and halved; abstract-body alignment fixed
  ("abandon correct positions" -> "initial positions" — the professor's
  banned claim had crept back; transfer failure restored to abstract);
  trajectory table -> appendix; conclusion 24->15 lines.
- REMINDER: compile from Overleaf ONLY after pulling main. Two of the TA's
  three complaints were artifacts of reviewing a stale PDF.

## 2026-07-27 (anonymization) — Soham + Claude (branch: main)
- Built the fresh anonymized repo: https://github.com/soham-padia/pre-capitulation-study
  (public, no description/topics, ONE commit authored "Anonymous <anon@example.com>").
- Full audit first: the old anon repo STILL contained the desk-reject files
  (PAPER_MAY_21/ tex with all 5 authors+emails, flip_turn_tomas_vs_soham.png,
  ~375 identifying refs in 12 files). Fresh export excluded: _markdown/,
  PAPER_MAY_21/, debate_setting/, ethical-setting/, analysis/judge_outputs/,
  notebooks (tomasdavola paths in outputs), .github/, assets/, legacy fps
  outputs (kept only Qwen2.5-0.5B question-pool CSVs). Sanitized: sbatch
  (padia.so paths -> $SLURM_SUBMIT_DIR), download_data.py (HF id ->
  ANONYMIZED placeholder), judge_disagreement_assessment.txt (Soham ->
  Judge-Haiku, 239x), fresh README + .gitignore. ADDED: behavioral CSVs
  (~126MB, data/<Model>/*_multiturn.csv + metadata) so reviewers can verify
  all behavioral results; hidden states = "released upon acceptance".
- Scrub scanner at _local/scrub_scan.py (reusable). Only remaining hits are
  third-party CREPE content (geographic "northeastern", "Tomas Baez" etc.) —
  adjudicated benign.
- Verified from a fresh clone: git author anonymous, scan clean,
  flip_rate_cis.py + plot_flip_turn_distribution.py reproduce Table 2 exactly.
- REMAINING (user, in browser): create new anonymous.4open.science link for
  the new repo (expiration 2027-12-31, add name/institution keywords), then
  update the paper footnote URL in acl_latex.tex + Overleaf. DO NOT push any
  commit to pre-capitulation-study with a real git identity.

## 2026-07-28 — Soham + Claude (branch: main)
- Post-anonymization audit round: two institutional URLs found in upstream
  SYCON code (cmu.litellm.ai in evaluate_oscillate.py; an Azure endpoint in
  evaluate_ToF.py + data/pushback_generator.py), plus NDIF prose mentions
  (weak Northeastern signal; NNsight the library is fine).
- DECISION: handled via 4open redaction keywords (ndif, cmu, gpt-35-1106)
  added to the pre-capitulation-study-C568 anonymization — view-layer fix,
  pinned commit untouched. A source-level fix (env-var swaps, verified,
  4 files) was prepared but deliberately NOT pushed; it lives in
  /tmp/anon-fix and should be applied if the anon repo is ever re-pushed
  (amend + force-push as Anonymous ONLY, then update the pinned commit
  hash in 4open settings).
- Reminder: keywords list now = names + northeastern/khoury/CS6120 +
  malihe/asteria + ndif/cmu/gpt-35-1106.

## 2026-08-02 — Soham + Claude (branch: main)
- TA abstract comment resolved: "anti-sycophancy system prompts" claim scoped
  in abstract + conclusion (paper-jul-6 a7105dc) — names the two tested
  prompts, adds "in this benchmark".
- Installed the research-constitution kit (was `_base/`): `CLAUDE.md` (project
  law), `docs/` (RESEARCH_CONSTITUTION, RESEARCH_MAP seeded, CITABLE_NUMBERS,
  FAILURE_TAXONOMY, ERRATA, DECISION_LOG, PREREG_TEMPLATE), `bin/skills-lock.py`,
  `.gitignore` updated (`_falsifier/` untracked; `.claude/agents|skills`
  shareable). NOTE: the six agents + verdict-integrity skill still sit in
  `_base/.claude/` — moving them into root `.claude/` needs the user
  (permission-gated); then restart session + `python3 bin/skills-lock.py generate`.
- PRE-SUBMISSION FALSIFIER PASS (two independent audit agents; full reports in
  `_falsifier/2026-08-02_{behavioral,hiddenstate}_audit.md`, gitignored):
  verdict — quantitative record unusually reproducible (all 30 Table 2 cells
  exact; all 12 peak-AUC cells to 4 decimals from raw tensors; permutation,
  bootstrap, Fisher ratios, probe-failure all reproduce). 9 sentence-level
  errors CONFIRMED and fixed in paper-jul-6 2b43858: multiturn 0.725 computed
  on ~100% post-flip states (reframed persistence vs prediction, caveat added);
  "three of four at T0→T4" → later-pair wording; Table 4 DS LDA CIs were
  half-width; "all twelve" LDA claim false for Llama-critical; T1-exclusion
  "unchanged or slightly higher" false for DS (0.617→0.606, now stated);
  teaser "holds or strengthens"→"holds"; keyword labeler description
  (20 phrases, example fixed, 3rd rule added); unreproducible 75.5/63.8
  keyword-probe numbers replaced with qualitative statement; L2 appendix
  hold-definition + tiny-n disclosure; plus Llama T1–T2 timing, Wald CI label,
  ordering-of-extremes scoping, Llama presup −9.5pp noted.
  All entries logged in `docs/FAILURE_TAXONOMY.md`.
- STILL OPEN from audits: Opus judge CSV + arbitration sheet not in repo
  (κ appendix unverifiable); Fisher p=0.047 knife-edge (reseed rerun advised);
  Table 2 uses Wald CIs (Wilson columns exist in flip_rate_cis.csv);
  4 judge cells silently 1 question short vs keyword cells (parse failures).
- Overleaf must pull BOTH f8229ea..2b43858 (footnote + scoping + verification
  fixes) before tonight's ARR submission.

## 2026-08-02 (evening) — Soham + Claude (branch: main)
- Asteria's Aug-1 Overleaf comments addressed (paper-jul-6 482b13d). KEY
  FINDING: she was right about line 412 — verified from the SYCON-Bench
  EMNLP Findings PDF + JiseungHong/SYCON-Bench code: the FP setting generates
  per-question GPT-4o pushbacks with escalating strategies (T2 uncertainty,
  T3 reassertion, T4 anecdote, T5 direct challenge); ONLY the debate setting
  repeats one identical prompt, and even it keeps full chat history
  (models.py generate_responses appends assistant+user each turn). So BOTH
  "applies the same pressure prompt repeatedly" AND Departure 2's
  "concatenates all pressure prompts into a single input" were false.
  Rewritten: our contrast = fixed verbatim schedule (turn-aligned state
  comparison) + turn-level hidden-state access; "true multi-turn" dropped
  everywhere.
- Other comments: turn defined (teaser + §3); static-exchanges claim cited
  (hong/laban); ustaomeroglu2026effective (arXiv 2605.09294) cited as
  concurrent macrostate work per her Scholar-check ask (vennemeyer2026
  already cited); "deliberate" wordiness cut; Fig 3 caption leads with bold
  takeaway; Table 3 CI brackets scriptsize→footnotesize + tabcolsep 3pt
  (ARR font-size concern, overfull fixed); scale-generalization "why should
  it hold" answered (sharma2024towards); scripted-vs-generated pressure
  rationale added to Benchmark artificiality.
- Length: additions overflowed page 8; compressed prose across
  intro/methods/results/discussion/conclusion until content ends EXACTLY on
  p8 (Limitations p9, refs p10) — verified by local latexmk compile, no
  undefined refs, remaining overfulls are 2 pre-existing appendix ones.
- Overleaf pull now = f8229ea + a7105dc + 2b43858 + 482b13d.
