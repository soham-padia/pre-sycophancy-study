# P5 prereg — what do the layer profiles show across all 12 cells? — DRAFT

Authorized by: meta-review ask 5 ("critical layer-wise evidence from the appendix should move to the
main text"); reviewer L9dW. Locked before the drift profiles of the eight non-best cells are computed.
Design reviews: pending.

## Motivation (frozen)
The drift–AUC dissociation is not supported in the four best cells (ERRATA E11). The main-text layer
evidence must be what the full 12 cells support, with statistics that respect layer autocorrelation.

## Design
- Per cell, rows 1..L: the T0→T1 cosine AUC with a 95% bootstrap CI (2,000, seed 20261021); mean drift
  1 − cos(h_T0, h_T1); the cell's sweep-corrected 95% permutation threshold (existing test).
- Per cell: Spearman ρ(drift, AUC) and ρ(drift, |AUC − 0.5|), p against 1,000 phase-randomised
  surrogates of the drift profile (seed 20261021); the gap in layers between max-drift and peak-AUC.
- Figure for the main text: the 12 AUC profiles with CIs and thresholds; drift on a twin axis.

## Gates (LOCKED; counts over the 12 cells, no pooling)
- **Co-location:** "drift and AUC peak within two layers of each other in k of 12 cells" — always
  reported with k.
- **Direction:** the number of cells with ρ(drift, AUC) significantly negative (p < 0.05, Holm over 12)
  is reported; "high-drift layers predict holding" may be said only of those cells.
- **Dissociation:** the word "dissociation", or "the layers that move most are not the layers that
  predict best", is earned only iff ρ(drift, |AUC − 0.5|) is ≤ 0 with p < 0.05 (Holm) in a majority
  (≥ 7) of cells. Otherwise FORBIDDEN, and the earned sentence is: "Layers with larger T0→T1 drift tend
  to be more predictive in either direction in k of 12 cells; drift magnitude does not fix the sign of
  the association."
- Any per-layer CI is labelled uncorrected for the sweep wherever it appears.
