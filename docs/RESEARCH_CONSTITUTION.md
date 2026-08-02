# A Constitution for Unbreakable Research Projects

*A portable process architecture for empirical work done with AI agents — extracted from a
mechanistic-interpretability project that caught 32 of its own would-be-false results before any
of them was published. Paste this folder into a new project and it installs the whole system:
the rules, the adversaries, and the loop that converts every failure into permanent enforcement.*

---

## The three findings about failure (read these even if you read nothing else)

Over two months of intensive experimental work, every near-miss was catalogued. Three patterns
account for nearly all of them:

**1. False nulls are the most expensive threat, not false positives.** The field trains you to
fear p-hacking — finding effects that aren't there. By raw count our catalog splits roughly
evenly between false-null and false-positive near-misses; but weighted by consequence, the
null-side failures dominate completely: the costliest near-misses were **nulls protected by a
missing positive control** — an instrument that couldn't detect the *real* effect confidently
reporting the absence of an induced one. Three separate
"the intervention does nothing" results traced to one generation configuration that a blind judge
could not read *even for the ground-truth manipulation*. The rule this bought:
> *A null, however clean, licenses nothing until the instrument that produced it passes a
> positive control on the same configuration.*

**2. Instrument failures outnumber analysis failures.** Scoring bugs, judge-run instability
(identical text, different scores across runs), ceiling-pinned scales, marker panels that
collapse to 2–3 effective dimensions, probes that detect "input is present" rather than the
construct. All of these sit *upstream* of any statistics — no correction procedure rescues a
measurement that means something other than what you think it means.

**3. A failure that doesn't become an enforced rule will be paid for twice.** Early failures in
our project survived for *weeks* before being caught. Late failures survived *minutes* — because
they collided with an enforcement point that an earlier failure had built. The half-life of a
bug class is a direct function of whether the last instance of it was converted into a checklist
item, an agent mandate, or a locked gate. This conversion loop — **catch → record → enforce** —
is the constitution's engine, and it is the part most worth stealing.

## The architecture (four layers + a loop)

### Layer 1 — Standing law (`CLAUDE.md`)
A short file loaded into every session: the facts that must never re-break, and one trigger rule —
*before any result becomes a verdict, run the verdict-integrity checklist.* Keep it under a page;
law that nobody reads is not law.

### Layer 2 — Skills (procedure at the moment of temptation)
Checklists that load when a matching task starts — not generic advice, but the specific discipline
a specific past failure bought. The capstone is **verdict-integrity** (included): the pre-verdict
checklist covering instrument positive controls, measurement-batch discipline, unit-of-analysis,
multiple-comparison families, gate exhaustiveness, and record consistency. Grow your own domain
skills next to it (our project's causal-intervention and probe-hygiene skills each encode a
specific disaster).

### Layer 3 — Agents (separation of duties)
The core principle: **whoever produces a result never adjudicates it.** Six portable agents are
included (`.claude/agents/`):

| agent | mandate | the principle it embodies |
|---|---|---|
| **code-verifier** | executes an evidence ladder on every code change before expensive compute | verification is execution, not reading |
| **results-auditor** | data integrity → leakage → negative controls → figure↔data, on any result about to be trusted | numbers must reproduce from raw artifacts |
| **confound-interrogator** | enumerates what *else* could produce the number; designs the discriminating control | a result's meaning is a separate claim from its existence |
| **research-director** | direction memos with value-of-information ranking and falsifiable gates; sits above the executors | direction is a decision, not a drift |
| **falsifier** | attacks the record itself — frozen claims, preregs, and *other agents' rulings* — with executable evidence only | the record must survive a motivated adversary |
| **big-picture-synthesist** | whole-arc framing for outsiders; adversarial-reviewer reads; drift checks | someone must own "so what" |

Two design details do most of the work:

- **The falsifier's incentive contract.** LLM red-teamers have a known failure mode: they invent
  impressive-sounding problems. The fix is in the contract: a clean negative ("attacked N ways,
  nothing broke" — with the attack list) counts as a **fully successful run**, every CONFIRMED
  finding must ship an evidence block (commands + pasted output + a one-line repro), and one
  fabricated finding voids the whole report. Findings it believes but didn't demonstrate go in a
  separate SUSPICION section with the cheapest discriminating check. It steelmans its own
  findings before filing. It is **write-confined** to one directory (`_falsifier/`) — it can
  demonstrate anything, touch nothing.
- **Adjudication agents get the strongest model you have, pinned.** An unpinned agent silently
  inherits whatever the session runs on; your verdict-gating layer shouldn't depend on that.

### Layer 4 — The record (claims as versioned, attackable objects)
Five documents (templates in `docs/`):
- **RESEARCH_MAP.md** — the wayfinder layer (adapted from
  [mattpocock/skills](https://github.com/mattpocock/skills)): destination + the **open frontier**
  of undecided questions, maintained by the research-director on every memo. Multi-session work's
  quietest failure mode is a thread silently dropping from one agent's working memory; the map is
  the fix, and it is an index — decisions live where they were made, never restated here.
- **CITABLE_NUMBERS.md** — the frozen chain. Nothing may be cited unless it is here, and
  everything here must reproduce from committed raw artifacts *by an independent recomputation*.
  Freeze verdicts together with a **data-snapshot identifier**, not just a code version — data
  pools grow, and a frozen number over a growing pool silently changes meaning.
- **ERRATA.md** — dated corrections. Wrong once is fine; wrong silently is not.
- **FAILURE_TAXONOMY.md** — the case history. Every failure appended with a date: what it would
  have falsely produced, how it was caught, the standing rule it left.
- **PREREG_TEMPLATE.md** — gate sheets locked before outcomes. The non-obvious requirement,
  bought by a real defect: gates must form an **exhaustive partition** of the observable outcomes
  (enumerate the regions; each maps to exactly one branch; run the reader-invariance test — could
  two honest readers adjudicate the same data differently?). And attack the prereg *before*
  results exist: that is the only time a gate defect is cheap.

Record rules that prevent the subtle corruption: **addendum, never edit** (new findings get dated
rows; frozen claims are never rewritten in place — and results on orthogonal axes never revise
each other); **de-cite with a dated reason** (a number that stops being trustworthy is struck
visibly, not deleted); **verify condition fields in artifacts, never prose labels** (our worst
false alarm came from a prereg calling a condition "no-context" when the data files said
`context=neutral`).

### Layer 5 — Process integrity (`skills-lock.json`)
The constitution polices the record; the lockfile polices the constitution. `bin/skills-lock.py
generate` pins SHA-256 hashes of the process layer (agents, skills, standing law);
`bin/skills-lock.py verify` detects drift — a silently weakened falsifier contract, a checklist
item that vanished, a hook or tool edit nobody reviewed. Git tracks *changes*; the lock
distinguishes **deliberate, reviewed process change** (regenerate + commit, with a message saying
what it blesses) from **drift since the last review** (verify fails). Living documents
(taxonomy, errata, the frozen chain) are deliberately unlocked — they're append-with-date by
design. Verify runs in seconds; make it part of the pre-verdict checklist and any pre-external
pass.

### The loop
```
prereg (locked, design-reviewed)          ← gates decided before data exists
  → code-verifier                          ← before any expensive run
    → run (manipulation checks, kill-switches)
      → analysis → verdict-integrity checklist
        → adjudication (separate agent, wording law)
          → frozen into CITABLE_NUMBERS
            → falsifier attacks it later
              → new failure → FAILURE_TAXONOMY (dated)
                → generalizes? → new check / new rule   ← the flywheel
```

## What this costs and what it buys
Two currencies, tuned separately:

**Tokens/money.** Each adjudication run (verifier, auditor, interrogator, falsifier) was
roughly 30k–150k tokens on a frontier model in our accounting; a heavy day of this process is a
few million tokens. (Self-declared estimates from session logs, not derivable from committed
artifacts — by this document's own citability standard, treat them as order-of-magnitude.) If you
pay per token, the budget configuration is: verifier on every pre-compute change, auditor +
interrogator only at verdict points, falsifier only pre-external and post-arc. **If your seats are
sponsored/flat-rate (university or lab plans), run the maximalist configuration — the marginal
cost of another adversarial pass is zero and the expected value is never zero:** falsifier after
*every* closed arc and on every locked prereg, synchronous verifier on every change however
small, double-adjudication (auditor *and* interrogator) on every verdict, and everything pinned
to the strongest model with no economizing tiers. In our project one of the most valuable runs of
the entire process — the one that found a locked gate two honest readers would score differently,
before any outcome-relevant data had been seen (generations mid-run, zero cells judged) — was a
scheduled adversarial pass with no specific suspicion attached: exactly the kind a per-token
budget teaches you to skip.

**Time/attention (the real constraint, and it doesn't go away when tokens are free).** Roughly
10–20% overhead per experiment (reviews, preregs, verification runs), front-loaded.
Buys: our late-stage experiments failed far faster than our early ones (self-declared
observation — we did not timestamp catch-latencies); an earlier full re-derivation of the frozen
chain caught two stale numbers and a wrong denominator (logged as errata — the process catching
its own drift is the process working); a later hostile adversarial pass over the newest,
highest-stakes sections found **no wrong headline number in anything it attacked** and only
record-keeping defects; and every external claim survives the question "how do you know?" with a
committed artifact and a repro command. For work you intend to publish, the overhead is not
optional — you either pay it during the project or your reviewers and replicators extract it
afterward, with interest.

*Origin: distilled 2026-07 from a study of induced affective states in open-weight LLMs
(replication + causal extension of Ben-Zion et al. 2025), where this process caught — among
32 catalogued failures — a scoring bug that flattened every self-report result, a probe that
detected narrative presence instead of emotion, a judge blind to its own ground truth, a
pseudo-replication that faked significance, and a locked gate that two honest readers would have
scored differently. None of them shipped — though in candor, at the time of writing nothing from
the project had yet faced external peer review either; this constitution has survived its own
falsifier, not yet a journal's. This document itself failed its first adversarial pass (an
overclaimed selling point, a quantifier its own catalog contradicted, and a silently-broken
config default) and was corrected — which is the loop working, and the reason to trust the
version you're reading slightly more than the one we wrote.*
