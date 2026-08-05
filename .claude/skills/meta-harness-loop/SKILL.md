---
name: meta-harness-loop
description: Drive the meta-harness optimization loop — propose next experiment, execute, capture results, iterate. Use when the user says "/meta-harness-loop", "run the next iteration", "what's the next experiment", "propose a hypothesis", or references Bayesian MMM optimization / harness iteration logs. Implements the Lee et al. 2026 meta-harness loop (arXiv:2603.28052).
user-invocable: true
allowed-tools:
  - Bash(*)
  - Read
  - Write
  - Edit
  - Grep
  - Glob
  - Agent
  - WebFetch
  - WebSearch
---

# /meta-harness-loop — drive the optimization iteration loop

**Academic root:** *Meta-Harness: End-to-End Optimization of Model Harnesses* (Lee, Nair, Zhang, Lee, Khattab & Finn, 2026, arXiv:2603.28052)
**Visualization:** See `docs/meta-harness-loop-dashboard-design.md` for the dashboard spec; prototype HTMLs at `docs/meta-harness-loop-prototype{-v2,}.html`.
**SFT training data:** Our Gemma-4-31B-MMM-SFT adapters were trained on 100+ Opus-authored examples extracted from the meta-harness iteration log — `data/sft_train.jsonl` + `data/sft_train_v2.jsonl`.

Arguments passed: `$ARGUMENTS` (e.g. "next iteration", "propose hypothesis", "what did we learn from iteration 14", "show frontier")

---

## Step 1: Know the loop's pieces

The meta-harness loop is a Bayesian optimization over harness configurations. Each iteration:

1. **Propose** a hypothesis (a harness variant to try) based on the frontier + lineage.
2. **Execute** the harness variant on the target task (currently: Bayesian MMM optimization).
3. **Capture** the score, mechanism, novelty verdict, tokens, cost.
4. **Self-critique** (expert agent): is the result novel, or a duplicate of a prior approach?
5. **Promote or reject** the checkpoint. Promoted checkpoints extend the frontier.
6. **Log** to the iteration log → feeds the next proposal.

**State files (HPC):**
- `/shared/project/<account>/llm/training/meta-harness/iterations.jsonl` — append-only iteration log (one JSON per iteration)
- `/shared/project/<account>/llm/training/meta-harness/frontier.json` — current Pareto frontier (score vs cost)
- `/shared/project/<account>/llm/training/meta-harness/lineage.json` — checkpoint parent-child tree
- `/shared/project/<account>/llm/training/meta-harness/approaches.json` — dedup table (approach signature → occurrences)

**Local (Windows) files:**
- `docs/meta-harness-loop-dashboard-design.md` — full dashboard spec (379 lines, read this for the visualization layer)
- `docs/meta-harness-loop-prototype-v2.html` — working HTML prototype (417 lines)
- `docs/meta-harness-loop-prototype.html` — earlier prototype (384 lines)
- `docs/CHRONICLE.md` — historical iteration entries

---

## Step 2: Understand the current state before proposing

Before proposing the next experiment, ALWAYS read:

1. `iterations.jsonl` (tail -20) — what's been tried recently
2. `frontier.json` — what the current best is
3. `lineage.json` — parent chain of the frontier
4. `approaches.json` — what's been tried and rejected (avoid duplicates)

If `$ARGUMENTS` is empty or "status": print a 1-page summary — current frontier, last 5 iterations, approaches tried count, budget consumed.

---

## Step 3: Propose a hypothesis

A hypothesis has:
- **mechanism**: code | prompt | tool | sampling | eval | architecture
- **novelty**: new | variant | mixed | duplicate
- **hypothesis**: one-sentence claim ("adding retry-on-5xx to the eval harness will improve score by 0.02")
- **parent checkpoint**: which frontier checkpoint this builds on
- **expected delta**: signed, e.g. +0.02 score, +5% cost
- **test plan**: what the harness variant actually does

**Novelty check is mandatory.** Before promoting a hypothesis, query `approaches.json` for the approach signature. If the signature already exists with the same mechanism+outcome, the hypothesis is a duplicate — reject and propose a different one.

---

## Step 4: Execute the harness variant

The execution target is currently Bayesian MMM optimization. The harness variant modifies one of:
- the prompt template the model responds to
- the tools available (e.g. add retry-on-5xx)
- the sampling parameters (temperature, top_p)
- the eval harness (how score is computed)
- the architecture (which model, which adapter)

Run via `scripts/extract_training_data.py --mmm-root ../../marketing-mix` (or the current harness driver). Capture: score, tokens used, cost, mechanism trace.

---

## Step 5: Self-critique

Spawn an expert agent (via Agent tool) to self-critique:
- Is the score improvement real, or within noise? (use the learnings dumbbell — mean ± std of trial scores)
- Is the approach novel? (query approaches.json)
- Does it extend the frontier? (Pareto check: score vs cost)
- Should it be promoted, rejected, or escalated?

Verdicts: `promote` | `reject` | `escalate` (escalate = mixed-mechanism, needs human review)

---

## Step 6: Log and iterate

Append to `iterations.jsonl`:
```json
{
  "checkpoint": 15,
  "parent": 14,
  "mechanism": "tool",
  "novelty": "new",
  "hypothesis": "add retry-on-5xx to eval harness",
  "score": 0.63,
  "delta_vs_frontier": +0.02,
  "tokens": 5610,
  "cost_usd": 0.084,
  "outcome": "promoted",
  "approach_signature": "tool+retry+5xx",
  "timestamp": "2026-08-05T..."
}
```

Update `frontier.json` if promoted. Update `approaches.json` either way (increment occurrences). Update `lineage.json` with the parent edge.

**Iteration budget**: check the budget tracker — if tokens or USD spent exceeds the configured cap, stop and surface to the user. Do not silently continue past budget.

---

## Step 7: Dashboard (optional)

The dashboard prototypes at `docs/meta-harness-loop-prototype-v2.html` render the iteration log as an academic-paper-style dashboard. Open in a browser to visualize:
- Checkpoint timeline (score over iterations, colored by outcome)
- Pareto frontier (score vs cost, pareto-optimal highlighted)
- Novelty audit (mechanism-type × self-critique verdict)
- Stop-hook log (allow/block/escalate events)
- Token spend (cumulative, per-actor)
- Goal heatmap (sub-objective progress)
- Approach dedup (signature → occurrences)
- Expert influence (parent → child decision lineage)
- Learnings dumbbell (trial score distribution per checkpoint)
- Lineage tree (parent-child, depth, promoted?)
- Sankey trace (research-finding → expert-decision → action → outcome)
- Cost tracker (cumulative USD, per-actor breakdown)
- Promotion debug (as-decided-score vs recomputed-score, would-flip?)

**Design register:** Academic journal, not SaaS. Serif headings, muted slate-blue accent, dark mode default. See `docs/meta-harness-loop-dashboard-design.md` §2 for full visual spec. The dashboard is a research artifact, not a product — every visual choice serves "is the loop converging, and can I trust the result?"

---

## Anti-patterns

- DO NOT propose a hypothesis without reading `approaches.json` first — duplicates waste budget.
- DO NOT execute past the budget cap without surfacing.
- DO NOT log an iteration without the self-critique verdict — unverified scores pollute the frontier.
- DO NOT skip the approach-signature computation — it's how the dedup table works.
- DO NOT treat the dashboard as a status display — it's a research instrument. Every figure must answer "is the loop converging?"

---

## Related

- `docs/CHRONICLE.md` — historical iteration entries (search "meta-harness")
- `docs/meta-harness-loop-dashboard-design.md` — full dashboard spec
- `scripts/extract_training_data.py` — pulls SFT pairs from the iteration log
- Memory: `project_v4flash0731_speed_optimization.md` — example of a multi-iteration optimization loop (V4-Flash speed work, 11 attempts)
- Academic paper: arXiv:2603.28052 (Lee et al. 2026)
