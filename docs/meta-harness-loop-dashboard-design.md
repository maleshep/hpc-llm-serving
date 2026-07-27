# /meta-harness-loop — Dashboard Design Specification

**Status:** Draft for review · **Owner:** Aman Khan · **Date:** 2026-07-14
**Academic root:** *Meta-Harness: End-to-End Optimization of Model Harnesses* (Lee, Nair, Zhang, Lee, Khattab & Finn, 2026, arXiv:2603.28052)
**Visualization lineage:** Niklaus, *harness-optimization* (HF Space, 2026); Harness Forge `pareto.py` (001TMF, 2026)

---

## 1. Design Philosophy

The dashboard is a **research artifact**, not a product. Every visual choice serves the question *"is the loop converging, and can I trust the result?"* — never decoration. The register is borrowed from academic journals: serif headings, figure captions with citations, a restrained palette, and minimal chartjunk (Tufte). The reader should feel they are inspecting a lab notebook, not operating a SaaS tool.

This is load-bearing for the public repo: a dashboard that *looks* like a startup reads as one; a dashboard that reads as a research contribution signals the skill's intent (generalized optimization in the token-bound regime) before a single word of the README.

### Principles

1. **Subtraction over addition.** No drop shadows, no gradients, no card chrome, no neon. A 1px rule border + generous padding define a figure. If a visual element doesn't carry meaning, remove it.
2. **One accent.** A single muted slate-blue carries all "primary" semantics. Ochre and oxblood are reserved for warning/failure states. Green appears only for "promoted/success" — and only where success is a meaningful category (not on every positive number).
3. **Type carries hierarchy.** Serif for headings/abstract/captions; sans-serif for data labels and axis ticks; monospace for numeric values and code. The reader's eye follows the same path it would in a paper.
4. **Motion is meaning.** Animations reveal *when* the loop made progress, not *that* it is alive. A promoted checkpoint animates differently from a rejected one. Static states are the default; motion is earned.
5. **Every figure is self-documenting.** A figure number, title, one-sentence description, and a source citation. A reader who sees only one chart should understand what it shows and where the idea came from.

---

## 2. Visual Register

### 2.1 Color — Dark mode (default) + Light mode

Dark mode is the default (the dashboard is watched during long overnight runs; a dark surface reduces fatigue and lets the accent and data marks read sharply). The register is "a photograph of a printed page in low light," not a developer tool — warm graphite, not blue-black; desaturated accent, not saturated slate.

```
DARK MODE (default)
--bg          #16140f   warm graphite (NOT blue-black — warmth reads as paper-at-night, not emissive screen)
--surface     #1d1a14   chart frame background (one step up from bg)
--surface-2   #252118   nested surface (tooltip, selected state)
--ink         #e8e2d4   warm off-white text (NOT pure white — pure white glares)
--ink-soft    #a59e8e   secondary text, axis labels
--ink-faint   #6a6454   tertiary text, gridlines-text
--rule        #2e2a20   dividers, chart borders
--rule-soft   #221f18   gridlines
--accent      #8a93b0   muted fountain-pen blue (desaturated — reads as ink on paper, not screen blue)
--accent-soft #5a6480   secondary accent fills (bars, areas)
--accent-faint #363d4e   faint accent (gridline-tinted)
--warn        #b8956a   muted ochre — blocked, mixed-mechanism
--good        #8aa37a   muted sage — promoted, success
--bad         #b88a8a   muted oxblood — rejected, escalated, failure

LIGHT MODE (toggle)
--bg          #faf8f3   paper warm-white
--surface     #fffefb
--surface-2   #f2efe7
--ink         #1c1a16
--ink-soft    #4a4640
--ink-faint   #8a857c
--rule        #d6d0c4
--rule-soft   #ebe7dc
--accent      #3a4566   deeper ink-blue (needs more weight on light)
--accent-soft #6a7388
--accent-faint #c4cad6
--warn        #8a6a3a
--good        #4a6450
--bad         #8a4a4a
```

**Rules:** No more than 3 hues on screen at once in the data layer (accent + one state color + neutral). Semantic colors (good/warn/bad) are *muted* — never saturated. Saturation reads as alarm; this dashboard never alarms, it informs. **The accent is a pair, not one color inverted** — the dark-mode accent (`#8a93b0`) is brighter/warmer than the light-mode accent (`#3a4566`) because dark backgrounds absorb visual weight; the same hue needs more luminance on dark to read at the same perceived intensity.

### Theme toggle — "printed" register, not "dashboard"

The toggle is a **bracketed monospace text-link**, not a button. No rounded corners, no chrome, no hover-fill. It reads like a footnote option in an academic PDF (the hyperref `href` register), not a SaaS control.

- Markup: `[ dark ] / [ light ]` top-right, `ui-monospace` 11px, `--ink-faint` default; the active option is `--ink` (bold-weight by contrast, not by font-weight).
- No border, no background, no padding box — just the text and the slash separator.
- Hover: the inactive option shifts to `--accent` (the only interactive cue).
- Persists to `localStorage['mh-theme']`, default `dark`.

This is the smallest possible visual footprint for the function, matching the "subtraction over addition" principle (§1). A pill button would announce itself as interactive; the bracketed link is present but not ornamented.

### 2.2 Typography

```
--serif   'Iowan Old Style', 'Palatino Linotype', Palatino, 'URW Palladio L', P052, Georgia, serif
--sans    ui-sans-serif, system-ui, -apple-system, 'Segoe UI', Roboto, sans-serif
--mono    ui-monospace, 'SF Mono', 'Cascadia Mono', Menlo, Consolas, monospace
```

| Role | Family | Size | Weight | Notes |
|------|--------|------|--------|-------|
| Page title (H1) | serif | 32px | 600 | letter-spacing -0.01em |
| Figure title | serif | 16px | 600 | |
| Abstract / description | serif | 14px | 400 italic | ink-soft |
| Body / axis labels | sans | 11px | 400 | ink-soft |
| Numeric values (status strip) | mono | 18px | 500 | ink |
| Tooltip value | mono | 13px | 500 | |
| Tooltip label | sans | 11px | 400 | ink-soft |
| Eyebrow / caption label | sans | 10px | 600 | letter-spacing 0.14em, uppercase, accent |

### 2.3 Spacing & Layout

- Max content width: **1180px**, centered. Wide enough for a 2-up figure row, narrow enough to read like a page.
- Page padding: 56px top, 64px sides, 96px bottom (breathing room below the fold).
- Figure frame: 1px rule border, 28px inner padding (top 28, sides 24, bottom 20).
- Figure caption: 14px margin above, padding 0 4px (aligns with frame inner edge).
- Grid: 12-column implicit, but figures flow in a 1-up (hero) or 2-up (supporting) grid.
- Status strip: flex, 48px gap, wrapped, bordered top+bottom.

---

## 3. Tooltip System (detailed — the user's refinement #1)

Every interactive chart has a **hover tooltip** that follows the pointer and respects the academic register. Tooltips are the primary way the reader drills into a checkpoint without leaving the dashboard.

### 3.1 Visual spec

- Container: `--surface-2` background, 1px `--rule` border, **no shadow, no blur** (academic = flat). 12px padding, 10px radius (the *only* rounded element — signals "interactive").
- Pointer: a thin 8px triangular notch pointing down at the hovered mark, same background + border.
- Typography inside: label (sans 11px, `--ink-soft`) above value (mono 13px, `--ink`). Key-value pairs stacked, 4px gap.
- A 2px left border in the figure's accent color ties the tooltip to the chart it came from.

### 3.2 Content model

Every tooltip is a **structured key-value record**, not free text. Each chart defines its tooltip fields; all share this shape:

```
┌─────────────────────────────────┐
│▎ CHECKPOINT 006                 │   ← header, sans 10px uppercase, accent left-border
│                                 │
│ Score          0.61              │   ← label sans 11px ink-soft · value mono 13px ink
│ Mechanism      code              │
│ Novelty        mixed             │
│ Hypothesis     add retry-on-5xx  │   ← wraps to 2 lines max, ellipsis
│ Tokens         5,610             │
│ Cost           $0.084            │
│ Outcome        rejected          │   ← colored: good/warn/bad
│ Δ vs frontier  −0.02             │
└─────────────────────────────────┘
```

### 3.3 Tooltip field map (per chart)

| Chart | Tooltip fields |
|-------|---------------|
| Checkpoint timeline | checkpoint, score, mechanism, novelty, hypothesis, outcome, Δ-frontier, tokens |
| Pareto frontier | checkpoint, score, cost, pareto-optimal?, blended-score, mechanism |
| Novelty audit | checkpoint, mechanism-type, self-critique verdict, is-novel, hypothesis |
| Stop-hook log | checkpoint, event-type (allow/block/escalate), reason, block-count, max-blocks |
| Token spend | checkpoint, research/expert/action tokens, cumulative, % of budget |
| Goal heatmap | sub-objective, checkpoint, progress, weight, what-changed |
| Approach dedup | approach-signature, occurrences, last-outcome, novelty, cost |
| Expert influence | checkpoint, parent-checkpoint, decision-summary, lineage-status |
| Learnings dumbbell | checkpoint, trial-scores, mean, std, cleared-min-delta |
| Lineage tree | checkpoint, parent, depth, promoted?, frontier-best? |
| Sankey trace | research-finding, expert-decision, action, outcome, tokens-moved |
| Cost tracker | checkpoint, cumulative-usd, cumulative-tokens, per-actor breakdown |
| Promotion debug | checkpoint, as-decided-score, recomputed-score, would-flip?, lambda |

### 3.4 Behavior

- **Hover → show** with 120ms delay (avoid flicker on rapid pointer movement).
- **Follow pointer** but constrained to the chart frame (never overflows the figure).
- **Pin on click** — a click pins the tooltip so the reader can scroll/compare; a second click unpins. Pinned tooltips stack with a z-index.
- **Keyboard accessible** — tab cycles marks; Enter pins.
- **Mobile** — tap shows tooltip centered above the mark; tap elsewhere dismisses.

---

## 4. Animation Spec — "Generous" Framer Motion

Motion is built with **Framer Motion** (React) / D3 transitions (standalone). The principle: **every state change animates, with spring physics, and the motion encodes meaning.**

### 4.1 Spring defaults

```
spring: { type: "spring", stiffness: 180, damping: 26, mass: 1 }
```

- Promoted checkpoint appears: spring-in (scale 0.6→1, opacity 0→1), 400ms.
- Rejected checkpoint appears: fade-in (opacity 0→1), 200ms — subtler, because rejection is not a "win."
- Frontier step-line extends: path animates left→right via `pathLength` 0→1, 600ms ease-out.
- Bar grow (novelty/spend): height 0→value, staggered 60ms per bar.
- Tooltip: opacity+translateY(4px→0), 120ms.
- Theme toggle: cross-fade bg+surfaces 300ms, accent hue shifts 450ms.

### 4.2 The play-button timeline (the centerpiece — the user's specific ask)

Figure 1 (Checkpoint Trajectory) is **animated and scrubbable**, mirroring Niklaus's `lab-frontier-climb.html`. This is the hero interaction.

**Controls:** A play/pause button (⏯) + a scrubber slider spanning checkpoint 1→N below the chart. A speed toggle (1× / 2× / 4×).

**Behavior on play:**
1. The animation steps through checkpoints one at a time (default 1.2s per checkpoint at 1×).
2. At each step: the checkpoint's dot springs in (promoted=solid accent+scale-bounce, rejected=hollow+fade). The frontier step-line extends to include it. A vertical "playhead" line marks the current checkpoint.
3. **The tooltip evolves** — at each step, the tooltip auto-pins to the current checkpoint showing its record (mechanism, hypothesis, outcome). As the playhead advances, the tooltip updates to the next checkpoint's values. This is the "see the tooltip evolve" the user asked for: the reader watches the loop's reasoning unfold checkpoint-by-checkpoint.
4. If a checkpoint was promoted, a brief accent flash radiates from the dot (150ms, opacity 0.4→0) — the "win" signal.
5. Pausing freezes the playhead; the scrubber lets the reader jump to any checkpoint; the tooltip follows.

**Why this matters:** The play button turns a static trajectory into a *narrated* one. The reader doesn't just see "it went up" — they see *which decision* moved the frontier, *what was tried* at the stuck points, and *how long* the loop spent before the breakthrough. That's the loop's story, told in its own terms.

### 4.3 Other animated charts

- **Pareto frontier**: new Pareto-optimal points spring in with a star; dominated points fade to faint. Hover shows the promotion margin.
- **Stop-hook log**: blocks appear as the playhead passes (when scrubbing the timeline), so the reader sees *when enforcement fired* relative to the trajectory.
- **Token spend**: bars grow + the cumulative line draws left→right, synced to the timeline playhead when both are visible.

---

## 5. The Thirteen Figures (full spec)

Each figure below: **what it shows, why it matters, data source, tooltip fields, animation.** Layout: Figure 1 is hero (full width); Figures 2–5 are a 2×2 supporting grid; Figures 6–13 fill below in 2-up rows.

### Figure 1 — Checkpoint Trajectory & Compounding Frontier (HERO, animated, scrubbable)
- **Shows:** X=checkpoint #, Y=goal-progress score. Filled dots=promoted, hollow=rejected. Step-line=frontier best-so-far. Play button + scrubber animate through checkpoints; tooltip evolves with the playhead.
- **Why:** The headline. Is the loop converging or thrashing? Where did it get stuck?
- **Data:** `log.jsonl` (eval events) + `frontier.json`.
- **Tooltip:** checkpoint, score, mechanism, novelty, hypothesis, outcome, Δ-frontier, tokens.
- **Animation:** play-button scrubber (§4.2); promoted dots spring+bounce, rejected fade; frontier path animates `pathLength`.

### Figure 2 — Quality–Cost Pareto Frontier
- **Shows:** X=token cost, Y=quality score. ★=Pareto-optimal, •=dominated. Floor line = min acceptable quality. Frontier line connects optimal points.
- **Why:** Not just "did it work" but "did it work efficiently" — the token-bound regime's core question.
- **Data:** `frontier.json` (pareto array).
- **Tooltip:** checkpoint, score, cost, pareto-optimal?, blended-score, mechanism.
- **Animation:** optimal stars spring in; dominated points fade.

### Figure 3 — Mechanism Novelty Audit
- **Shows:** Per checkpoint, a stacked bar: novel (accent) / mixed (ochre) / parameter-only (oxblood). Running tally of novel vs param-only.
- **Why:** Is the loop exploring the mechanism space or thrashing on parameter tuning? The paper's central anti-pattern made visible.
- **Data:** `log.jsonl` (is_novel_mechanism) + `self-critique.md`.
- **Tooltip:** checkpoint, mechanism-type, self-critique verdict, is-novel, hypothesis.
- **Animation:** bars grow staggered 60ms; running tally counter ticks up.

### Figure 4 — Stop-Hook Enforcement Log
- **Shows:** Timeline of Stop events: allow (sage), block (ochre), escalate (oxblood). Each tick = one event; rows by type.
- **Why:** Proves the defining property — the loop doesn't quit when the worker runs out of ideas. Shows how often enforcement fired and whether any checkpoint hit the escalation valve.
- **Data:** `log.jsonl` (block/allow events).
- **Tooltip:** checkpoint, event-type, reason, block-count, max-blocks.
- **Animation:** ticks appear as timeline playhead passes (synced to Fig 1).

### Figure 5 — Token Expenditure by Role
- **Shows:** Stacked bar per checkpoint (research/expert/action). Cumulative line vs budget ceiling.
- **Why:** In the token-bound regime this is the analogue of compute-hours/iteration. Where does the money go?
- **Data:** `log.jsonl` (cost_tokens per event_type).
- **Tooltip:** checkpoint, research/expert/action tokens, cumulative, % of budget.
- **Animation:** bars grow; cumulative line draws left→right.

### Figure 6 — Goal Convergence Heatmap
- **Shows:** Rows=sub-objectives (parsed from goal spec), cols=checkpoints. Cell=progress (rose→amber→teal). Hover row=rubric+weight; hover cell=what-changed.
- **Why:** Which parts of the goal are stuck vs progressing? Catches the "overall score up but one sub-objective stalled" failure.
- **Data:** `outcome.md` per checkpoint + `state.json` goal decomposition.
- **Tooltip:** sub-objective, checkpoint, progress, weight, what-changed.

### Figure 7 — Approach Attempts Dedup (bubble scatter)
- **Shows:** X=cost, Y=score, bubble=novelty confidence, ring=frontier member, color=mechanism type. Overlapping bubbles=same approach (dedup working).
- **Why:** Visualizes the no-retry rule. Clusters of rejected-same-approach prove dedup is catching repeats.
- **Data:** `log.jsonl` + `tried_approaches.json` + `frontier.json`.
- **Tooltip:** approach-signature, occurrences, last-outcome, novelty, cost.

### Figure 8 — Expert-Decision Influence Graph (directed lineage)
- **Shows:** Nodes=checkpoints, directed edges="built on". Orphans (rejected but later reused) dashed. Green=promoted lineage, red=dead-end, amber=orphan-stacked-later.
- **Why:** The intellectual lineage of ideas — which expert decisions led somewhere vs dead ends.
- **Data:** `expert-decision.md` + `outcome.md` per checkpoint.
- **Tooltip:** checkpoint, parent-checkpoint, decision-summary, lineage-status.

### Figure 9 — Per-Checkpoint Learnings (dumbbell / range plot)
- **Shows:** Per checkpoint, dots=score across retries or sub-objectives; line=range; collapses-to-zero highlighted.
- **Why:** Noise characterization. Justifies the min_delta promotion threshold.
- **Data:** `outcome.md` + `frontier.json`.
- **Tooltip:** checkpoint, trial-scores, mean, std, cleared-min-delta.

### Figure 10 — Promotion Rule Debug (side-by-side)
- **Shows:** Two-panel: as-decided (stored score) vs recomputed-under-current-weights. Highlights checkpoints where the decision would flip.
- **Why:** Catches the "stale derived number" bug (Niklaus's lambda bug). Audit-grade rigor.
- **Data:** `frontier.json` + `log.jsonl` eval rows.
- **Tooltip:** checkpoint, as-decided-score, recomputed-score, would-flip?, lambda.

### Figure 11 — Cumulative Cost Tracker
- **Shows:** Line chart, X=checkpoint, Y=cumulative USD + tokens. Budget cap line. Per-checkpoint delta bars.
- **Why:** Is the loop within budget? Where does the money go over time?
- **Data:** `log.jsonl` (cost_usd + cost_tokens accumulated).
- **Tooltip:** checkpoint, cumulative-usd, cumulative-tokens, per-actor breakdown.

### Figure 12 — Approach Lineage Tree
- **Shows:** Root=incumbent (checkpoint 0), children=checkpoints that built on parent. Promoted=solid, rejected=dashed, frontier-best=gold ring.
- **Why:** How ideas compound — the evolutionary tree of the solution.
- **Data:** `frontier.json` + `log.jsonl` (base_checkpoint lineage).
- **Tooltip:** checkpoint, parent, depth, promoted?, frontier-best?.

### Figure 13 — Research-to-Outcome Trace (Sankey)
- **Shows:** Left=research findings (by source), middle=expert decisions, right=actions+outcomes. Flow width=token count. Color=outcome.
- **Why:** Which research actually moved the needle vs was ignored by the expert.
- **Data:** `research.md` + `expert-decision.md` + `outcome.md`.
- **Tooltip:** research-finding, expert-decision, action, outcome, tokens-moved.

---

## 6. Page Architecture

```
┌─────────────────────────────────────────────────────────┐
│ EYEBROW: Optimization Run · Dogfood Trial 01            │
│ H1: /meta-harness-loop                                   │
│ Subtitle (serif italic): the paradigm, one line          │
│ Authors/target line                                      │
│ ─────────────────────────────────────────────────────── │
│ ABSTRACT block (accent left-border, rule-soft bg)        │
│ ─────────────────────────────────────────────────────── │
│ STATUS STRIP (6 cells: status, checkpoints, frontier,    │
│   tokens, expert-calls, blocks-enforced)                 │
│ ─────────────────────────────────────────────────────── │
│ FIGURE 1 (HERO, full width, animated + play button)      │
│   [⏯ play] [scrubber ────●────] [1×/2×/4×]              │
│   caption                                                 │
│ ─────────────────────────────────────────────────────── │
│ FIG 2 (Pareto)        │ FIG 3 (Novelty)                  │
│ ──────────────────────│────────────────────────────────│
│ FIG 4 (Stop-hook)     │ FIG 5 (Token spend)              │
│ ──────────────────────│────────────────────────────────│
│ FIG 6 (Heatmap)       │ FIG 7 (Dedup)                    │
│ ──────────────────────│────────────────────────────────│
│ FIG 8 (Influence)     │ FIG 9 (Learnings)                │
│ ──────────────────────│────────────────────────────────│
│ FIG 10 (Promo debug)  │ FIG 11 (Cost tracker)            │
│ ──────────────────────│────────────────────────────────│
│ FIG 12 (Lineage tree) │ FIG 13 (Sankey)                  │
│ ─────────────────────────────────────────────────────── │
│ FOOTER: citation (mono), arXiv id                        │
└─────────────────────────────────────────────────────────┘
```

- **Theme toggle** (dark/light) top-right, persists to localStorage (`mh-theme`, default `dark`).
- **Sticky mini-status bar** appears on scroll: a thin strip showing just the current checkpoint # + score + a play-button that jumps back to Figure 1. Lets the reader scrub from anywhere.

---

## 7. Label Collision Rules (the user's refinement #2 — conflict fix)

The prototype had axis labels overlapping data values. Fixed by rule:

1. **Reserve plot padding:** left 52px (y-axis labels + axis title), bottom 44px (x-axis ticks + title), top 16px, right 24px. Data marks never enter this margin.
2. **No inline data values.** A point's value is shown only via tooltip on hover, never as a text label on the mark. The only inline text on a chart is axis ticks and the axis title.
3. **Axis title offset:** rotated y-title sits at x=−34px from the plot, centered on the plot height — clear of tick labels.
4. **Legend placement:** top-right inside the plot, in a 0-opacity-safe zone (top 10px, right 24px). If it would overlap data, move to below the caption.
5. **Gridlines behind data:** `--rule-soft`, 1px, no labels. Data marks always render above gridlines.
6. **Dense charts (heatmap, Sankey):** row labels truncated with ellipsis at 28 chars; full label in tooltip.

---

## 8. Data Flow

- **Source of truth:** `.optimization/state.json` (current status), `.optimization/log.jsonl` (append-only events), `.optimization/frontier.json` (Pareto frontier).
- **Materialized view:** `.optimization/dashboard/data/dashboard-data.json` — a denormalized snapshot rebuilt by `dashboard/server.py` every 5s (poll, not file-watch — Windows-safe).
- **Browser:** fetches `dashboard-data.json` on load + 5s auto-refresh (or SSE push if the reader keeps the tab focused).
- **Dual-write discipline (mirrors Stanford):** worker appends to `log.jsonl` + writes checkpoint subdirs; Stop hook writes `state.json.stop_hook`; frontier rewritten atomically (write-temp→rename) by the worker after each eval.

---

## 9. Implementation Notes

- **Standalone build:** D3 v7 + vanilla JS (mirrors Niklaus's embeds/ pattern — one HTML per chart, loadable independently). No build step. This is what ships in the public repo for portability.
- **React build (optional, for the React-based skill dashboard):** Framer Motion for animation, Recharts or visx for chart primitives. Same spec, same register. The design.md is framework-agnostic; the standalone build is the reference implementation.
- **Theming:** CSS variables (§2.1) swapped via a `data-theme` attribute on `<html>`. One stylesheet, both modes.
- **Accessibility:** every chart has a `<title>` and a text fallback table below the SVG (visually hidden, screen-reader visible). Color is never the sole encoder (shape + label too).

---

## 10. Open Questions (for review)

1. **Standalone D3 vs React+Framer?** The standalone build (D3, no build step) ships more portably for the public repo. The React build gives richer Framer Motion. Proposal: **standalone D3 as the reference implementation** (ships in repo), with the animation spec (§4) implementable in D3 transitions — D3 can do the play-button scrubber and spring-like easing without Framer. Confirm.
2. **SSE vs poll?** Poll (5s) is simpler and Windows-safe. SSE is real-time but adds a persistent connection. Proposal: poll, with SSE as a documented upgrade. Confirm.
3. **Pinned-tooltip stacking limit?** Proposal: max 3 pinned tooltips per chart; oldest unpinns. Confirm.
4. **Play-button default speed?** Proposal: 1.2s/checkpoint at 1×. Confirm.

---

## References

- Lee, Y., Nair, R., Zhang, Q., Lee, K., Khattab, O., & Finn, C. (2026). *Meta-Harness: End-to-End Optimization of Model Harnesses.* arXiv:2603.28052.
- Niklaus, J. (2026). *Hardening the LLM as an optimizer.* HuggingFace Space `joelniklaus/harness-optimization`.
- 001TMF (2026). *harness-forge.* GitHub `001TMF/harness-forge`.
- Tufte, E. R. *The Visual Display of Quantitative Information.* (chartjunk principle.)
