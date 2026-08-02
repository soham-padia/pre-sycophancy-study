# CITABLE_NUMBERS — the frozen chain

**The rule:** nothing may be cited (in a paper, poster, README claim, or conversation with a
collaborator) unless it has a row here. Every row must reproduce from committed raw artifacts by
an independent recomputation — not by re-running the analyzer that produced it. If a number will
be cited, it will one day be recomputed by a hostile reader; save them the trip.

**Freezing protocol:**
- Freeze the number together with a **data-snapshot identifier** (commit SHA of the data files,
  not just the script — data pools grow and silently change frozen numbers' meaning).
- State the unit of analysis, the n, the test, and the correction family alongside the number.
- New findings = new dated rows (ADDENDUM). Never edit a frozen row; strike-and-supersede with a
  date if a correction is unavoidable, and log it in ERRATA.md.
- De-cited numbers stay visible, struck, with a dated reason.

| Claim (with unit, n, test, correction family) | Number(s) | Caveats (attached, always travel with the number) | Source (data-snapshot SHA + artifact paths) |
|---|---|---|---|
| | | | |
