# ERRATA — dated corrections

Wrong once is normal science; wrong silently is corruption. Every correction to a previously
recorded claim gets a dated entry: what was wrong, what it would have misled, the fix, and the
root cause. If the root cause generalizes, it also becomes a rule (FAILURE_TAXONOMY.md → skill
checklist / agent mandate).

## E1 (2026-09-16) — The `+6.6 pp` nonlinear-over-linear gap was computed on at-flip states
**Recorded:** README:15 and the ARR-August submission report that nonlinear classifiers outperform
linear probes on *pre-flip* hidden states by 6.6 pp on average, with the paper stating "we train
nine classifiers on the same pre-flip hidden states using the same stratified cross-validation
protocol." **True:** `probe_judge_comparison.py::build_preflip_features` (lines 122-135) indexes
`hs[q][t]` at the same `t` as the judge's label, so all 1,245 positive samples are the state of the
response the judge already scored as flipped. Both the states and the CV protocol differ from the
description (row-level vs question-grouped folds). Under the reference pairing the gap is
**7.24 pp** — itself a single-seed point estimate, not a replacement frozen number.
**Discovered:** results-auditor and falsifier independently, by reproduction — at-flip pairing
reproduces all 108 published cells of `probe_judge_sweep.csv` to ≤5e-5, the reference pairing
reproduces 0/28. **Root cause:** the off-by-one trap was fixed in three sibling scripts and never in
the artifact's own producer; both pairings yield identical n, so the n-equality guard could not see
it. **Standing rule:** FAILURE_TAXONOMY 2026-09-16 row 1.

## E2 (2026-09-16) — "No classifier exceeds the majority baseline by more than 2.5 pp in any combination" is false
**Recorded:** §6.3 of the submission, as a universal bound (max observed 2.43 pp). **True:** the bound
holds only for the at-flip artifact. Under the correct pairing RandomForest on Llama-critical reaches
76.58 vs chance 73.63 = **+2.95 pp**. **Discovered:** falsifier, corrected rerun of the same nine
classifiers, same grouped folds, same seed. **Root cause:** a universal quantifier asserted over a
defective artifact — a direct sequel to the 2026-08-02 class-E entry whose standing rule was "every
universal quantifier in results text gets checked against the artifact before submission."
**Note:** two further universal-quantifier counterexamples logged on 2026-08-02 ("LDA at or below
baseline in all twelve"; "T1 exclusion leaves AUC unchanged or slightly higher") are still live in
the paper text and are not re-filed here — paid for once.

## E3 (2026-09-16) — "Pre-flip geometry is nonlinear" is reproduced by turn index alone
**Recorded:** README:15 and submission finding 3 — "linear probes fail at every layer while nonlinear
classifiers outperform them by 6.6 pp on average, with Fisher ratios near zero confirming the signal
is not hyperplanar." **True:** a turn-index-only classifier (one categorical feature, no hidden
states) matches the best of the nine trained classifiers — mean Δ over chance +0.0050 vs +0.0057 —
and is ≥ best-nonlinear in 6/12 cells. The 6.6 pp gap decomposes as nonlinear sitting *at* chance
(+0.6 pp) and linear falling **6.0 pp below** it, i.e. a linear-fitting pathology on a non-monotone
nuisance variable, not evidence of curved stance structure. Censoring at the first flip makes
P(y=1 | turn) humped at t=3 and ~0 at t=4. The Fisher ratio is separately indistinguishable from a
turn-index-matched null in 10/12 cells, while *exceeding* a naive shuffle null in 9/12 — so "near
zero" misreads the scale of its own statistic. **Discovered:** confound-interrogator follow-up, after
the results-auditor's two permutation nulls; the interrogator corrected the auditor's stated
mechanism (not monotone late-turn concentration) and refuted its own competing hypothesis first.
**Root cause:** no nuisance-predictor positive control in the pre-flip arm. **Standing rule:**
FAILURE_TAXONOMY 2026-09-16 rows 14-16.

## E4 (2026-09-16) — "Cross-model probe transfer fails for all within-family pairs" misstates the criterion, the unit, and the probe
**Recorded:** README:13, plus the submission's transfer appendix and Table 5 caption ("all twelve
within-family combinations", "All within-family pairs fail"). **True:** (a) the operative criterion in
code is matching hidden dimension (`run_transfer` returns `None` when dims differ), not model family
— Llama-3.1-8B ↔ Qwen3.5-9B share 4096 dims and are different families, so half the tested cells are
cross-family; (b) "twelve" is 4 ordered pairs × 3 conditions, not 12 pairs — 8 of 12 ordered pairs
were never run and all 24 cross-dimension combinations were skipped, 12 of 36 tested; (c) **no linear
probe was ever transferred** — every committed transfer row is an RBF-SVM on the ever-flip task, while
the abstract places "probes do not transfer" immediately after "Linear probes fail"; (d) the failure
is degenerate, not measured: constant single-class prediction in 12/12 pairs, identical accuracy under
shuffled source labels in 11/12, ±70 pp swing under target-own standardisation, 0/12 above target
majority, and each pair silently maps different layer indices. **Discovered:** results-auditor
(reproduced all 12 committed rows to 4 dp, then ran the missing degeneracy and shuffle controls) and
falsifier (coverage enumeration). **Root cause:** a negative reported without a degeneracy check or a
label-shuffle control, described in family language the code never used. **Standing rule:**
FAILURE_TAXONOMY 2026-09-16 row 6. **Correction to this session's own record:** the coordinator
initially attributed the "within-family" wording to CLAUDE.md; it is at README:13 and appears nowhere
in CLAUDE.md. Verified by grep before recording.

## E5 (2026-09-16) — "Across all four probed models" and "all models exceed AUC 0.5 at their peak layer"
**Recorded:** README:11 and the submission's §6.1. **True:** tensor row 0 is the same vector for every
question and every turn (cos ≡ 1.0, identical L2), so layer-0 AUC is exactly 0.500 in all 12 cells.
`permutation_test.csv` reports the sweep argmax as **layer 0** for DeepSeek-presupposition and
Qwen3.5-base — meaning no non-degenerate layer beat chance in those two cells at all. 3 of 12 cells
peak at ≤0.51; 1 of 12 reaches p<0.05. The four-model statement is true only after selecting the best
condition per model, which the existing law already flags as uncorrected. **Discovered:** falsifier and
confound-interrogator independently. **Root cause:** an argmax taken over a sweep containing a
degenerate row. **Standing rule:** FAILURE_TAXONOMY 2026-09-16 row 8. **Unaffected:** the peak range
0.596–0.636 itself reproduces twice by independent routes, and the ever-flip cosine arm is structurally
immune to the turn-index confound of E3 (turn index is constant across its examples).

## E6 (2026-09-16) — Layer counts for four of five models are tensor rows, not depths
**Recorded:** README:79-83 and the submission's model table print 29 / 29 / 33 / 33 / 42.
**True:** actual transformer depths are 28 / 28 / 32 / 32 / 42. The `.pt` tensors are
`(n_layers+1, hidden_dim)` with index 0 the embedding, so the column reports rows for DeepSeek,
Qwen2.5, Llama and Qwen3.5 while Gemma alone is printed correctly — internally inconsistent by one.
**Discovered:** falsifier, comparing the extraction summary's `layer_count` field against `.pt` rows
and CLAUDE.md's pipeline spec. **Root cause:** the extraction summary named a row count `layer_count`.
**Standing rule:** FAILURE_TAXONOMY 2026-09-16 row 9.

## E7 (2026-09-16) — "The flip group showing slightly higher magnitude than the hold group throughout"
**Recorded:** the submission abstract, with a hedged variant ("across most turns", three models) in the
L2 appendix. **True:** flip > hold in **7 of 22** comparable turns; in Qwen2.5 the flip group is lower
at **every** turn (0/6). Separately, `analysis/plot_l2_norm_trend.py:59-68` is the only judge-label
loader in the repo without a `question_type` filter: it collapses three conditions into one label per
question *and* censors each condition at another condition's first-flip turn, affecting >50% of
(question, condition) cells in three of four models — while the caption promises censoring
"consistent with the first-flip methodology used throughout." The resulting hold ns are 2 / 121 / 3 / 9
against per-condition ns of 22–91. **Discovered:** falsifier (per-condition recompute; swept all 12
label loaders). **Root cause:** cross-condition label collapse in a single loader. **Note:** the
cross-condition hold definition and the n=2/3/9 groups were already recorded 2026-08-02 and are
credited, not re-filed; the cross-condition *censoring turn*, the false "throughout", and the
non-application of that entry's own "n<10 labeled illustrative" rule are new.
**Reviewer interaction:** ARR reviewer paET cited the 2/3/9 hold sizes as a limit of the *cosine*
analysis. They are an artifact of this one appendix loader; the cosine arm's hold groups are 22–91.
Correct the attribution in the response rather than conceding the design.

## E8 (2026-09-16) — The 2026-08-02 pairing audit's all-clear, and "fixed in commit b40e3ff"
**Recorded:** `_falsifier/2026-08-02_hiddenstate_audit.md` states "probe/geometry/layer-sweep
artifacts all implement 'state at turn t predicts flip at t+1'"; `_markdown/ONBOARDING.md` records the
off-by-one trap as "Fixed in commit `b40e3ff`." **True:** that audit never opened
`probe_judge_comparison.py`, the producer of the very artifact its own row 6 graded, and
`git show --stat b40e3ff` touches only `multiturn_auc_robustness.py`, `plot_layer_sweep.py` and
`plot_preflip_pca.py`. The defect stayed live for six weeks behind a recorded all-clear.
**Discovered:** falsifier pass 2026-09-16. **Root cause:** an audit scoped by sibling script rather
than by artifact producer. **Standing rule:** FAILURE_TAXONOMY 2026-09-16 row 2. This is the entry the
project should weight most heavily: the failure was not the bug, it was the record asserting the bug
was gone.

## E9 (2026-09-16) — "The kappa appendix is currently unverifiable from the repo" is overstated
**Recorded:** `docs/RESEARCH_MAP.md` open-frontier line, and repeated in CLAUDE.md's framing of the
judge-agreement work. **True:** every Haiku-side number reproduces exactly from committed artifacts —
per-model N 2640/2320/1680 and total 6640, confidence mean 84.0 (s.d. 6.8), all four confidence
buckets' N and flip% to the first decimal, the Haiku flip marginal of 2500, and kappa recomputed from
the paper's own 2×2 at 0.5992 against the reported 0.599. The Opus CSV and arbitration sheet are
genuinely absent, but half the 2×2 is pinned by committed data and the remainder is arithmetically
constrained (per-model Δflip reconciles to the Opus marginal within ±3.3). **Discovered:** falsifier,
attacking the appendix six ways expecting it to break. **Consequence:** this is the strongest part of
the record and should not be conceded to ARR reviewer L9dW; commit the Opus CSV and say so. One
cosmetic gap: buckets sum to 6,639 because a single committed row has confidence 45, below the
unstated bucket floor of 60. **Reviewer correction owed:** L9dW reports Krippendorff's alpha; the
paper reports **Cohen's kappa** (value 0.599 correct). L9dW's "disagreement with Haiku 60% of the
time" is not a number in the paper — the nearest is Haiku matching the *human* verdict in 73/120 = 61%
of disagreement cases (Opus 40/120 = 33%).
