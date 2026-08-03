# ARR August 2026 submission form — field-by-field answers

For the OpenReview form at ACL ARR 2026 August. Deadline: **Aug 04, 11:59 AM UTC**.
Paper: "Before the Flip: Hidden-State Precursors of Multi-Turn Sycophancy" (paper-jul-6 @ `d758797`).

## Before finalizing — hard requirements

- [ ] **Re-upload the PDF.** Any PDF compiled before `d758797` contains a duplicated
  Ethics block with contradictory DeepSeek licenses and illegible references
  (lowercase "llm/ai/llama"). Pull `d758797` into Overleaf, recompile, re-upload.
- [ ] **Confirm author order before submitting.** The PDF and the author list lock at
  the deadline (other metadata may stay editable during a short grace period, but the
  form's boilerplate references past cycles — treat the deadline as final for everything).
  Current form entry: Utkarsh, Soham, Tomas, Vedant, Malihe, Asteria (the May submission
  had Asteria before Malihe — team call).
- [ ] Verify in the compiled PDF: title says "Before the Flip", page 9 starts with
  "Limitations", no author names anywhere, 4open footnote link resolves.
- [ ] Immediately after submitting: **all six authors complete the author registration
  form** in the author console (mandatory for committing to EACL; reviewer duty policy).

## Main form fields

| Field | Answer |
|---|---|
| Title | Before the Flip: Hidden-State Precursors of Multi-Turn Sycophancy |
| Keywords (general) | sycophancy, multi-turn dialogue, hidden states, probing, interpretability, alignment, LLM-as-judge evaluation, social pressure |
| TL;DR | Extending SYCON-Bench's false-presupposition setting into multi-turn dialogues, we find a weak, layer-localized hidden-state precursor of sycophantic capitulation at the first pressure turn, though reliable detection remains beyond current probing methods. |
| Abstract | Use the exact abstract from the tex at `d758797` (matches the PDF) |
| Paper Type | Long |
| Research Area | **Special Theme (conference specific)** = EACL 2027's "The Human in Language" (single-select; no secondary). Reasons: the theme's scope explicitly includes interpretability, so the pool is not stats-naive; theme-pool reviews assess theme fit, which is what supports the theme-track commitment on Oct 11; the paper ranks higher against theme competition than against mainline interpretability papers; and the theme track is what the team told the advisor and TA. Research-area keywords (probing, robustness, calibration/uncertainty) still steer matching within the pool. |
| Research Area Keywords | probing, robustness, calibration/uncertainty |
| Contribution Types | Model analysis & interpretability; NLP engineering experiment; Publicly available software and/or pre-trained models; Data analysis |
| Languages Studied | English |
| Previous URL | **Leave empty** — the May submission (#4581) was desk-rejected BEFORE review; the CFP resubmission policy covers prior versions *reviewed* at ARR. (If in doubt, email support@aclrollingreview.org.) |
| Explanation of Revisions PDF | Leave empty (resubmission-only) |
| Justification for Author Changes | Leave empty |
| Reassignment Request AC | This is not a resubmission |
| Reassignment Request Reviewers | This is not a resubmission |
| Software archive | Leave empty (anonymized repo link in the paper covers it; every upload is an anonymization surface) |
| Data archive | Leave empty (same reason — this is how May went wrong) |
| Preprint (ARR anonymous preprint) | no |
| Preprint Status | "We are **considering** releasing a non-anonymous preprint in the next two months" — do NOT pick the binding "no preprint" option; "considering" keeps the arXiv option open at no cost |
| Existing Preprints | Empty |
| Preferred Venue | EACL |
| Visa Needs | yes |
| Country of Origin | IN |
| Consent to Share Data | yes |
| Consent to Share Submission Details | On behalf of all authors, we agree |
| Author Submission Checklist | yes |
| License Agreement | On behalf of all authors, I agree |

## Responsible NLP checklist

Answers that CHANGED since the May submission are marked **(changed)**.

| Q | Answer | Elaboration |
|---|---|---|
| A1 potential risks | Yes | Ethics Statement |
| B artifacts | Yes | — |
| B1 cite creators | Yes | Sections 3.1, 4.2; Appendices A and I (model papers are cited in the Table 10 caption, not in Section 3.3 itself) |
| B2 licenses | Yes | Ethics Statement |
| B3 PII | N/A | Artifacts derive from public benchmarks (SYCON-Bench, CREPE); Section 3.1 notes they contain no personally identifiable information |
| B4 offensive content | N/A | Same; open-domain factual questions, Section 3.1 |
| B5 documentation | Yes | Sections 3.1, 3.2, 4; Appendix A |
| B6 data statistics | Yes | Sections 3.1, 5.1; Table 2; Appendix I |
| C computational experiments | Yes | — |
| C1 size/budget | Yes | Section 3.3; Appendix I |
| C2 setup/hyperparameters | Yes **(changed from N/A)** | Sections 3.5, 3.6, 6.2, 6.3; Appendix I (defaults stated; no hyperparameter search performed, stated explicitly) |
| C3 descriptive statistics | Yes **(changed from No)** | Bootstrap 95% CIs (Sections 5.2, 6.1, Limitations); sweep-corrected permutation test (Section 6.1); binomial CIs (Table 2); Wilson CIs (Table 3) |
| C4 packages | Yes | Section 4.2; Appendix I |
| D human annotators | Yes | — |
| D1 instructions | N/A | All annotations (120 arbitration cases) were provided by the paper's authors; the question exempts author-provided annotation |
| D2 recruitment/payment | N/A | Same |
| D3 data consent | N/A | Same |
| D4 ethics board | N/A | No human subjects; author-only annotation of model outputs |
| D5 annotator population | N/A | All annotations by the authors |
| E AI assistants | Yes | — |
| E1 AI assistant info | Yes | Ethics Statement (Claude Code disclosure); Claude judge models are research objects, documented in Section 4.2 and Appendix A |

## After submission

- All authors register as ARR reviewers by **Aug 5** (per the EACL CFP; confirm the
  exact date in the author console after submitting).
- EACL 2027 commitment deadline: **Oct 11** (select the Human in Language theme track there).
- Post-submission experiment queue (for author response): T0-correctness fraction +
  restricted rerun (~1 day), pressure-blind judge (~1 day), neutral-pressure control
  arm pilot on Qwen2.5 (~2–3 days cluster). See `docs/RESEARCH_MAP.md`.
