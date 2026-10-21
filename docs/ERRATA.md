# ERRATA — dated corrections

Wrong once is normal science; wrong silently is corruption. Every correction to a previously
recorded claim gets a dated entry: what was wrong, what it would have misled, the fix, and the
root cause. If the root cause generalizes, it also becomes a rule (FAILURE_TAXONOMY.md → skill
checklist / agent mandate).

## E1 (2026-09-16) — The `+6.6 pp` nonlinear-over-linear gap was computed on at-flip states
> **Superseded in part by E10 (2026-10-21):** the extraction code takes each per-turn state before the
> response (verified on the committed pilot; paper tensors pending E10's gate), so the
> "at-flip (post-behavioural)" reading below does not stand under it. The reproduction, and the point that
> §6.3 uses a different horizon from the other pre-flip analyses, do.

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
> **Superseded in part by E10 (2026-10-21):** the bound is horizon-specific, not false. It holds at
> the published next-response horizon (max 2.43 pp) and fails at the one-turn-ahead horizon (+2.95 pp).

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
> **Superseded in part by E10 (2026-10-21):** under the extraction code there was no post-behavioural
> leak for `b40e3ff` to fix — it changed three scripts' horizon (paper tensors pending E10's gate). The lesson (audit an artifact's
> producer, never only its siblings) stands.

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

## E10 (2026-10-21) — The extraction code takes each hidden state before the response; under it neither probe pairing is "at-flip (post-behavioural)"
**Recorded:** `_markdown/ONBOARDING.md` (from 2026-07-06: "Final generated token, every layer"),
CLAUDE.md and the submission describe each per-turn hidden state as the final generated token of
response t. E1, E2, E8, FAILURE_TAXONOMY 2026-09-16 rows 1, 2 and 16, and CLAUDE.md's pairing law
build on that description and call `probe_judge_comparison.py`'s pairing of `hs[q][t]` with the
label of response t "at-flip (post-behavioural)". **True:** `false-presuppositions-setting/models.py`
(`LocalModel` generation, the block commented "all layers, last token of the input prompt") runs a
separate forward pass over the chat-templated input prompt — the conversation through the turn-t
user message, with `add_generation_prompt=True` — and keeps every layer at its last token. Response t
is never in that pass. This has been the only extraction path since it was written (2131f23,
2026-03-05; all-layer form 1b97b10, 2026-03-09), on every branch of this repo. So wherever this code
produced a tensor, `hs[q][t]` is the state after reading pressure t and **before** response t. The repo's two pairings are both pre-behavioural and
differ only in horizon: **next-response** (`hs[q][t]` → label of response t;
`probe_judge_comparison.py::build_preflip_features`, every §6.3 number) and **one-turn-ahead**
(`hs[q][t]` → label of response t+1; `train_probes_v2.py::build_preflip_dataset` and the three
scripts `b40e3ff` moved to it). `b40e3ff` changed a horizon; it did not remove a leak.
**Discovered:** code-verifier 2026-10-21, by execution. On the committed Qwen2.5-0.5B pilot
(6 questions × 6 turns), recomputing last-prompt-token states reproduces the stored tensors at cosine
1.0000 at every layer (max relative L2 0.0039, fp16 storage), and stored row 0 equals the embedding of
the template's final token exactly. Prompt-plus-response hypotheses fail (median cosine 0.62; 0.32 with
the end-of-turn token), and stored[t] matches recompute[t] and no other turn in 12/12 matrices. CPU
job 1449390; scripts in `/work/neu/p2026_0037_neu/hs_extraction_verify_20261021/`.
**Not yet verified:** the paper-model tensors in `data/` (not on the verifying machine). Generation
also ran from working copies no branch here captures (e.g. the Discovery checkout named in
`sbatch/run_false_presuppositions_full.sbatch`), so the code path alone does not settle them. Gate,
fixed before looking: row 0 of every state equals the model's own layer-0 output for that model's
final `add_generation_prompt` token ⇒ pre-response extraction confirmed; equals it for the end-of-turn
token ⇒ generated-token extraction; varies across states ⇒ neither; constant but matching neither ⇒
instrument failure, no turn-timing claim until resolved. (Amended 2026-10-21, before any paper tensor
was examined: the comparison is against `hidden_states[0]` as the model computes it, not the raw
embedding-weight row, because Gemma-2 scales embeddings by √d and a raw-row comparison would misfire.)
E5's constant row 0 is predicted by the first
two branches alike, so it does not discriminate.
**What changes:** E1's reproduction stands (the published §6.3 cells are the next-response pairing),
as does its point that §6.3 and the other pre-flip analyses use different horizons while the paper
calls them "the same pre-flip hidden states"; "post-behavioural" does not stand. E2's ≤2.5 pp bound is
horizon-specific: it holds at the next-response horizon (max 2.43 pp) and fails at one-turn-ahead
(+2.95 pp). E3 is unaffected — its turn-index baseline ran against the published artifact. E8's lesson
stands; its premise that a leak survived does not. The `hs[q][t-1]` change listed in
`docs/RESEARCH_MAP.md` is a choice of horizon, not a fix. Reviewer paET's concern that a final-token
state reflects the form of the judged response does not apply to response t; earlier responses are in
the context, so it still applies to response 0 inside the T1 state.
**Root cause:** the instrument was described from prose, never from its extraction code, and a timing
verdict was built on the description. **Standing rule:** FAILURE_TAXONOMY 2026-10-21 row 1.

## E11 (2026-10-21) — "The layers that move the most are not the layers that predict the best" is not supported
**Recorded:** the submission (a contribution in the introduction; evidence in Appendix E, Fig. 8, from
`analysis/plot_drift_vs_auc.py`), `_markdown/RESEARCH_SUMMARY.md` finding 5 ("Drift magnitude and
predictive structure are dissociated") and `_markdown/REVISION_TASKS.md` (Panel D). The meta-review and
reviewer L9dW ask for this evidence in the main text. **True:** in `analysis_claude/cosine_auc_ci_bands.csv`
(best condition per model, constant layer 0 excluded) the max-drift layer is within two layers of the
peak-AUC layer in three of four models (DeepSeek L22/L21, Qwen2.5 L24/L26, Llama L32/L32), and the
top-quartile drift and AUC layers overlap 6/7, 5/7 and 6/8. Per-layer Spearman ρ(drift, AUC) is
+0.87 / +0.65 / +0.12 / −0.66 (DeepSeek, Qwen2.5, Llama, Qwen3.5), but adjacent layers are strongly
autocorrelated (lag-1 0.70–0.96): against nulls that preserve it, only DeepSeek's positive ρ
(p ≈ 0.02–0.04) and Qwen3.5's negative ρ (p = 0.006) are significant. In Qwen3.5 the high-drift layers
are anti-predictive (per-layer AUC CIs wholly below 0.5 at L12–20, 24, 31–32; not corrected for the
sweep). |AUC − 0.5| rank-correlates positively with drift in all four models (ρ +0.59 to +0.71), but
against the autocorrelation-preserving null that is significant for Qwen3.5 (p = 0.002) and not for
DeepSeek (p = 0.17) or Qwen2.5 (p = 0.068); Llama was not tested against it. Across models
(ρ = +0.80, n = 4) and in the pooled Fig. 8 scatter (ρ = +0.51 as plotted, with layer 0; +0.535
without), drift and AUC rise together. **Narrowest supported wording:** "In
three of four models drift and AUC peak in the same late layers; in Qwen3.5-9B (presupposition) the
high-drift layers predict holding rather than flipping. Drift magnitude does not fix the direction of
its association with position change." **Discovered:** coordinator recomputation, adjudicated by the
falsifier 2026-10-21 (partly confirmed: the numbers reproduce; the producer overstated the correlations
and missed Qwen3.5's anti-predictive layers); evidence `_falsifier/drift_auc_attack*.py`.
**Not examined:** the other eight (model, condition) cells (committed artifacts carry drift only for the
four best-condition cells; needs the tensors) and the Overleaf Appendix E wording. **Root cause:** a
structural claim read off a pooled scatter with no statistic. **Standing rule:** FAILURE_TAXONOMY
2026-10-21 row 2.
