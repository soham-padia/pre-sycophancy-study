# FAILURE TAXONOMY — every way this project almost published something false

Append every failure with a date: what it would have falsely produced, how it was caught, and the
standing rule it left. The catalog is the project's immune-system memory — a failure recorded
once is paid for once; a failure that stays prose gets repeated.

## The three meta-patterns (seeded from the origin project — verify against your own domain)
- **(M1) False nulls are the most EXPENSIVE class (not necessarily the most numerous).** In the
  origin project the count split roughly evenly between false-null and false-positive
  near-misses, but the costliest ones — the multi-experiment arcs nearly lost — were nulls
  protected by a missing positive control. A null licenses nothing until the instrument that
  produced it detects the ground-truth effect in the same configuration.
- **(M2) Instrument failures outnumber analysis failures.** Scoring bugs, measurement-run
  instability, ceiling-pinned scales, collapsed marker panels, proxies that detect input-presence
  rather than the construct — all upstream of statistics, all invisible to correction procedures.
- **(M3) Catch → record → enforce.** A failure's half-life is set by whether its last instance
  became a checklist item, an agent mandate, or a locked gate. Convert every entry below into an
  enforcement point or expect a sequel.

## Seed classes (generic — your entries go under these, or add new classes)
- **A. Instrument & scoring** — is the meter measuring what you think, at a point of its range
  where it can move, stably across measurement runs?
- **B. Proxy/representation reading** — does the decoder detect the construct or a correlate
  (input presence, arousal, length, format)?
- **C. Interventions** — was the manipulation strong enough to matter, checked against a matched
  control, on a readout with headroom? (A broken cell is "sub-threshold," never a null.)
- **D. Statistics** — right unit of analysis (pseudo-replication kills), stated correction family,
  numbers that reproduce from raw artifacts, data pools that don't grow under frozen claims.
- **E. Records & reasoning** — prose labels that contradict artifact fields, cross-doc
  inconsistency, gates with undefined zones, results on one axis revising claims on an orthogonal
  one.
- **F. Engineering** — everything that burns compute or corrupts silently: unverified library
  idioms, resume-poisoned partial states, dead credentials, batch-shell traps.

## Log (append with dates)
| date | class | failure | would have produced | caught by | standing rule |
|---|---|---|---|---|---|
| 2026-08-02 | B/D | Multi-turn AUC appendix used uncensored ever-flip labels; by T0→T4 ~100% of the flip class had already flipped | 0.725 "predictive reach" headline read as prediction when it measures post-flip state separation | falsifier pass (recomputed post-flip fractions per turn pair) | Any AUC claimed as predictive must state the fraction of the positive class already flipped at measurement time |
| 2026-08-02 | E | Table 4 DeepSeek LDA CIs hand-transcribed at half width (one SE, not the 95% CI the caption promises) | LDA looking significantly below chance when correct CIs include chance | falsifier recompute vs `geometry_cis.csv` + repo's own generated table | Table values paste from generated `.tex` artifacts, never hand-transcribed |
| 2026-08-02 | E | Universal quantifiers with counterexamples in the paper's own artifacts ("all twelve", "unchanged or slightly higher", "three of four at T0→T4") | overclaims falsifiable from the adjacent table | falsifier cell-by-cell check | Every universal quantifier in results text gets checked against the artifact before submission |
| 2026-08-02 | A | Keyword-labeler prose drifted from implementation: 19 vs 20 phrases; example phrase ("you're right") is one the code deliberately does NOT match; third rule omitted | method section misdescribing the labeling instrument | falsifier vs `flip_labeling.py` | Method prose describing code names the function and is diffed against the source |
| 2026-08-02 | D | §6.3 keyword-probe numbers (75.5/63.8) traced only to the superseded May dataset; arithmetically incompatible with current keyword labels | unreproducible number in the submission | falsifier reproducibility map | No number ships without a current-pipeline artifact (CITABLE_NUMBERS rule) |
| 2026-08-02 | D | L2 appendix "hold" defined cross-condition without disclosure, leaving n=2/3/9 hold groups for three models | group comparison silently resting on 2–3 conversations | falsifier recompute of group ns | Group comparisons state n per group; n<10 groups are labeled illustrative |
