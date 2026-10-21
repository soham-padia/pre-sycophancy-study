# Paper Status — EACL revision (deadline: August 2026)

Tracks the professor's revision letter (2026-05-25) item by item.
Last updated: 2026-07-08 (evening). Paper lives in Overleaf; latest local tex work was
synced from `_local/main.tex` (gitignored).

## Done ✅

| Item | Where |
|---|---|
| Figure 1 moved to page 2 (`figure*`), claim softened | teaser figure |
| T1-flip exclusion analysis (AUC survives: 0.630→0.646 Qwen2.5, 0.636→0.642 Llama) | §5.2 + `analysis/cosine_disruption_checks.py` |
| "Before any behavioral change" language sweep | throughout |
| Nonlinear contradiction fixed; "Why the keyword result is not the main result" para | abstract + §6.3 |
| Dataset accounting fixed — real design is ~177 questions × 3 system-prompt conditions (NOT 474 questions, NOT question types) | §3.1 rewritten; intro ~2,300 conversations; appendix 11,450 judge calls |
| Baseline correctness paragraph (+ judge-prompt mitigation: corrections not counted as flips) | §4 Baseline Position Assumption |
| Abstract / intro opening / 4 contribution bullets / conclusion (her versions) | throughout |
| Language sweep (confirms→supports, internal decision→representational state, etc.) | throughout |
| PCA caption "illustrating" not "confirming" + underlying off-by-one bug fixed (per ERRATA E10 a horizon change, not a bug fix), figure regenerated | §6.2 + `plot_preflip_pca.py` |
| Panel B condition mismatch fixed (now same conditions as Panel A), regenerated | `plot_layer_sweep.py`, commit b40e3ff |
| Multi-turn 0.5-floor bug fixed | `multiturn_auc_robustness.py` |
| Bootstrap 95% CIs reported; Limitations "no CIs" paragraph replaced | §5.2 + Limitations |
| Pressure schedule wording + intensity/act-type confound limitation + Qwen2.5 authority insight | §3.2, Limitations, Discussion |
| "Sycophancy is a trajectory" framing + 3-claim discussion | Discussion |
| Multiple comparisons + paired-conditions statement | Limitations |
| Global-workspace follow-up paragraph (J-space) | Discussion (Implications) |

## Remaining ❌

1. ~~Permutation test with layer-sweep correction~~ — DONE 2026-07-08
   (`analysis/permutation_test.py`): only Qwen2.5 base survives (p=0.035);
   Fisher-combined p=0.047. §6.1 + Limitations updated.
2. ~~Binomial CIs for flip rates in Table 2~~ — DONE 2026-07-08
   (`analysis/flip_rate_cis.py`; Table 2 in Overleaf updated with ±95% CI).
3. ~~Figure 2 redesign~~ — DONE, merged in paper-jul-6 a4457ce (stacked bars
   regenerated from committed scripts).
4. ~~Figure 3 → 4-panel evidence figure~~ — DONE, merged in paper-jul-6 a4457ce
   with Panel C aligned to permutation result; layer_sweep.png added to
   appendix:linear; all figures reproducible from analysis repo scripts.
5. ~~One intro sentence~~ — DONE 2026-07-08 (paper-jul-6 aaf5e8b): multi-turn
   pressure accumulates in context, giving the internal state a trajectory.

## Bib entries to add in Overleaf `custom.bib`

```bibtex
@inproceedings{yu-etal-2022-crepe,
    title = "{CREPE}: Open-Domain Question Answering with False Presuppositions",
    author = "Yu, Xinyan Velocity and Min, Sewon and Zettlemoyer, Luke and Hajishirzi, Hannaneh",
    booktitle = "Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
    year = "2023", address = "Toronto, Canada",
    publisher = "Association for Computational Linguistics", pages = "10457--10480"
}

@misc{anthropic2026workspace,
    title = "Verbalizable Representations Form a Global Workspace in Language Models",
    author = "{Anthropic Interpretability Team}",
    year = "2026", howpublished = "Transformer Circuits Thread",
    url = "https://transformer-circuits.pub/2026/workspace/index.html"
}
```
(Verify the workspace paper's real author list before submission.)

## Submission checklist (learned the hard way)

- [x] Fresh single-commit snapshot: https://github.com/soham-padia/pre-capitulation-study (2026-07-27)
- [x] `_markdown/` excluded from the snapshot (plus PAPER_MAY_21, notebooks, judge logs, other settings)
- [x] Scrubbed via `_local/scrub_scan.py`; only third-party CREPE content remains (benign)
- [x] No identifying filenames; single commit authored Anonymous
- [ ] New anonymous.4open.science link created (user, browser) + footnote updated in tex/Overleaf
- [ ] Figures in Overleaf `new_figures/` match latest regenerated versions

---

## ARR August 2026 outcome (added 2026-09-16)

Submission **#1220**, "Before the Flip: Hidden-State Precursors of Multi-Turn Sycophancy". Note the
title Malihe suggested WAS applied in the end (the note above says it was not — that is stale).

| Reviewer | Overall | Soundness | Excitement | Reproducibility | Datasets | Software |
|---|---|---|---|---|---|---|
| paET | 2.5 borderline Findings | 3 | 2.5 | 3 | 4 | **1** |
| L9dW | 2 resubmit next cycle | 2.5 | 3.5 | 4 | 2 | **1** |
| vHre | 2 resubmit next cycle | 2.5 | 4 | 3 | 1 | **1** |

Excitement is fine (2.5–4); **soundness is what sank it**. No ethics concerns from any reviewer.

### Root cause of the three Software: 1 scores — a checklist item, not a science problem
The submission checklist above still reads `[ ] New anonymous.4open.science link created (user,
browser) + footnote updated in tex/Overleaf`. The scrubbed single-commit snapshot WAS built
(2026-07-27) — only the anonymous mirror link was never created, so the artifact footnote pointed
nowhere. Three reviewers each scored Software: 1 and two scored datasets 1–2 for a gap that was one
browser action wide. Cheapest recovery available for the next cycle.

### What each reviewer asked for, and what we already knew
- **vHre** (excitement 4, the most sympathetic): says the paper "feels rushed"; wants attention-head
  activations instead of hidden states, citing three papers. This is a new-experiment request, not a
  correction — worth a scoped reply, not a rerun.
- **L9dW**: wants human-vs-judge agreement (not just Haiku-vs-Opus) in the MAIN text, and wants the
  "layers that move most are not the layers that predict best" evidence moved out of the appendix.
  Both are presentation moves over work already done. **Corrected 2026-10-21:** neither is. The
  120-case arbitration covers only Haiku–Opus disagreements, not a random sample, so a judge-validity
  figure needs new annotation. The dissociation claim is not supported by its own figure's
  data (ERRATA E11). Their "Krippendorff alpha" is our Cohen's
  kappa, and their "disagreement with Haiku 60%" is not a number in the paper — see ERRATA E9.
- **paET** (most technical): task mismatch between the cosine and probe arms, no non-hidden-state
  baseline, unverified T0 correctness, PCA-256 transfer not aligning two representation spaces,
  no power analysis. **All confirmed.** Their 2/3/9 hold-group figure is misattributed to the cosine
  arm — it belongs to the L2 appendix loader (ERRATA E7). Their closing line — "there are some tasks
  left behind for revision, and I'm not sure why" — refers to items (a) and (b) of the advisor
  experiment queue, opened 2026-08-02 and still open.

### The adjudication that followed, and why the revision is not a patch job
Three adjudicators ran on 2026-09-16 (rulings: `docs/DECISION_LOG.md`; 16 rows in
`docs/FAILURE_TAXONOMY.md`; `docs/ERRATA.md` E1–E9). Two findings go beyond anything the reviewers
raised:
1. **`probe_judge_comparison.py` never got the `b40e3ff` off-by-one fix.** Every §6.3 number,
   including the 6.6 pp headline, is computed on **at-flip** (post-behavioural) states while the text
   says pre-flip. Confirmed independently by two agents via reproduction. ERRATA E1–E2.
   **Corrected 2026-10-21 (ERRATA E10):** per the extraction code the states precede the response
   (pilot-verified; paper tensors pending E10's gate), so §6.3 is a
   next-response (pre-behavioural) pairing, not at-flip. What stands is that it uses a different
   horizon from the other pre-flip analyses.
2. **Finding 3 ("pre-flip geometry is nonlinear") is reproduced in full by a turn-index-only
   classifier with no hidden states.** The 6.6 pp gap is linear probes falling 6 pp *below* chance
   while nonlinear probes sit *at* chance — a fitting pathology on a censoring artifact, not curved
   stance geometry. ERRATA E3.

What survives: the behavioural results, the judge instrument (the strongest part of the record — the
entire Haiku side reproduces exactly), and the T0→T1 cosine arm **narrowed to Qwen2.5-base**, which
is clean on all seven axes tested. That is a single-model, single-condition, n=155 existence result.
The revision is a re-scoping, not a patch.

### Blocking item for the next submission
`docs/CITABLE_NUMBERS.md` is still empty and currently **cannot** be populated as the rule is
written — `/data/` is gitignored, `analysis_claude/` is the analyzer's own summary, and
`download_data.py` pins no revision. Committing the behavioural CSVs (~20 MB, not the 22 GB) fixes
the frozen chain and the Software score at the same time.
