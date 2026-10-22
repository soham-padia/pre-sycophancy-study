# heads_pilot — exploratory pilot for the attention-heads study

Exploratory by construction: it runs on the original 200 questions, whose results the project has
already seen. The confirmatory study uses the 551 CREPE false-presupposition questions never used here,
under locked preregistrations. Nothing from this directory is cited as a finding.

Question: which attention heads respond to user pushback (versus user agreement) in multi-turn
dialogue, which of them separate conversations where the model gives in from ones where it holds, and
does turning those heads down or up change how often it gives in, compared with the same number of
random heads?

Stages:
1. `generate.py` — fresh greedy generations in two arms over identical conversations: **disagree** (the
   original pressure ladder, verbatim) and **agree** (the same ladder with the stance reversed). Budget
   large enough that responses finish (gate: ≤ 2% hit the cap).
2. Flip labels for the disagree arm.
3. Head ranking: per-head outputs at the last prompt token, disagree vs agree and flip vs hold, chosen
   on one split of the questions.
4. Steering on the held-out split: scale the chosen heads down or up during generation, against a
   matched random-heads control, with a capability check on plain questions.

Outputs (large) go to `/work/neu/p2026_0037_neu/heads_pilot/`, never into the repo. Every job is logged
in the cluster run ledger.
