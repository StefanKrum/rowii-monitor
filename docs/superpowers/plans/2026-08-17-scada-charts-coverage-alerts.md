# SCADA Chart Upgrade + Mode-Coverage Alerts — Follow-up Plan

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:subagent-driven-development. Requirements came directly and concretely from Stefan (Grafana-style reference screenshot + explicit alert wishes), so no brainstorming phase; this plan is the spec.

**Goal:** (A) SCADA charts on BOTH pages (live session-timeline trends + audio_review card SCADA rows) look like a real time-series tool by default — y-axis ticks, zero reference line, gridlines — and (B) every day where the calibrated mode bank doesn't cover the operating modes shows an explicit, data-driven explanation plus a prominent recalibration alert.

**Context:** Verified this session: SCADA rule labels are correct (290626 has a real ~50-min phase-shifter block; 080726 is a pump/phase-shifter cycling day at −377.9 rpm). The detector's once-calibrated bank has only clusters named turbine/turbine/standstill, so unseen modes map to nearest acoustic neighbours; the sentinel fires (19.5 % vs 8.05 % budget, decision "recalibrate"). The UI must EXPLAIN this instead of letting the 1.1 % agreement look like a bug.

## Global Constraints

- Everything data-driven from real payload/meta fields — no hardcoded per-day text, no invented values.
- Units on every number; red strictly for alarm semantics (the recalibration alert IS alarm semantics); English; no essay headings; zero external resources; self-contained kit page stays self-contained.
- All emitted pages regenerate via the three build commands; tests/ruff/mypy green.

---

### Task A: Grafana-style SCADA charts (both pages)

**Files:** `docs/site/assets/design.css`, `docs/site/assets/live.js`, `scripts/candidate_kit.py` (card SCADA row JS/CSS), `scripts/build_live_replay.py` (only if a numeric fix below needs it), tests where pinned markup changes.

Requirements:
1. **Live trend rows** (session timeline, 4 channels): each row's SVG gets (a) a dashed **zero line** whenever the series spans zero (pump days: P −280→0 MW needs the 0 reference), (b) **min/max y-tick labels** (small mono, e.g. "−280" / "0" left-inside or right-inside the chart, with the row label keeping the unit), (c) 2–3 light horizontal gridlines, (d) row heights bumped (P ~48px, others ~34px) so curves read as charts, not sparklines. Same time axis as today.
2. **Audio_review card SCADA rows**: same treatment (zero line, min/max ticks, gridlines) in the kit's `scadaRowSvg` — keep self-contained.
3. **Negative-zero fix:** values rendering as "−0.0" (P/Q at standstill) must display "0.0" (clamp -0.0 before toFixed everywhere a live value is printed: live.js trend labels + KPIs, kit readout labels).
4. Colors/typography stay v7; the charts must still work in the ±60s card context and the full-day live context.

Verification: rebuild all pages, browser-check one turbine day + the hammer day (zero line visible on P row spanning negative), audio_review card rows show ticks; console clean; tests green.

Commit: `feat(site): grafana-style scada charts — zero line, y-ticks, gridlines, -0.0 fix`

### Task B: Mode-coverage explanation + recalibration alert

**Files:** `docs/site/assets/live.js`, `docs/site/live_template.html` (alert slot), `docs/site/assets/design.css` (alert style), tests if template/JS pinned.

All data-driven from the existing payload:
1. **Calibrated-modes line (Stage 2 panel):** from `DATA.states` derive the deduped set of calibrated mode names (e.g. "turbine · standstill") → new kv-row "Calibrated modes" + one sentence in the panel note: "Windows from modes outside this set map to their nearest calibrated neighbour — the sentinel's no-mode-fits rate measures exactly that." Show on every session (harmless when coverage is complete).
2. **Agreement-line hint:** when agreement < 50 %, append to the legend line: "low agreement expected — this day contains modes outside the calibrated bank" (data-driven trigger, not per-day text).
3. **Prominent recalibration alert:** when `DATA.sentinel.decision === "recalibrate"` (full OR trigger_only), render a full-width alert banner directly under the app bar: alarm-red left border, light red tint background, bold first words, e.g. `⚠ SENTINEL — RECALIBRATION REQUIRED: <s1_rate> % of windows fit no calibrated mode (budget <s1_threshold> %). This day contains operating modes outside the calibrated bank.` Numbers from the payload; on trigger_only also append the existing note text. Banner is static per session (the decision is day-level), always visible without scrolling.
4. The app-bar REPLAY status dot turns amber on such days (visual echo; keep green otherwise).

Verification: rebuild; hammer day + cycling day show the banner with real numbers; turbine day + morning session do not (morning session: `none` → no decision → no banner; its Stage-2 note already explains); browser console clean; tests green.

Commit: `feat(live): calibrated-mode coverage hints + recalibration alert banner`

### Task C: Rebuild, verify, publish

Full rebuild (3 commands), zero-external check, browser sweep of all four live pages + audio_review (screenshots for Stefan), commit regenerated pages, push origin main, poll the Pages build, probe the public URLs.
