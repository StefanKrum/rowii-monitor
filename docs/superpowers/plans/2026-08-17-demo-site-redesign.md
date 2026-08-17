# Demo-Site Redesign (v7 Light Monitoring) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rebuild `docs/site/` as a light monitoring-software UI: one merged "Audio & review" page, a 4-session live replay with named stage-1 features and full timeline legend, units on every number, English throughout.

**Architecture:** The site stays fully generated (no hand-edited HTML): `design.css` becomes the single v7 design system, `site_common.py` emits the shared app-bar/group-label chrome, `build_live_replay.py` builds one payload+page per session (two-pass for the cross-session switcher), and a renamed `publish_audio_review.py` composes `audio_review.html` from candidate-kit card data plus the strike/per-mode clips. The standalone offline kit under `results/candidate-kit/` stays self-contained and untouched in behavior.

**Tech Stack:** Python 3.13 (repo venv `.venv/bin/python`), pytest, vanilla JS (IIFE, no frameworks), hand-written CSS, numpy/pandas for payloads.

**Spec:** `docs/superpowers/specs/2026-08-17-demo-site-redesign-design.md` (D1–D10). Approved visual references get committed in Task 1.

## Global Constraints

- All site content in **English** (D8).
- **Every number carries its unit or label** — `MW`, `rpm`, `% of windows`, `σ`, `UTC`, `h`, `m³/s`, `pos`, `episodes`, `mel bands`, `%` (D7). New UI code must not print a bare number.
- **Red only for alarm semantics**; amber = transition/caution/EVENTS; green = ok/live/agreement; state colors keep the closed vocabulary (D6). No decorative accent.
- **No essay headings, no section numbers** — group labels only (D5).
- All numerals render in the mono stack with `font-variant-numeric: tabular-nums`.
- Pages must stay self-contained: no external HTTP(S) resources (`build_site.find_external_resource_urls` must return `[]` for every emitted page).
- Every number shown comes from real artifacts; mockup values were placeholders and must never be hardcoded.
- Run all commands from the repo root `repos/rowii-monitor/`; python is `.venv/bin/python`, tests via `.venv/bin/python -m pytest`.
- Commits: concise messages, **no Co-Authored-By lines**.

---

### Task 1: Commit the approved mockups as design references

The visual truth for every later task. The brainstorm session dir is gitignored and transient, so copy the two approved finals into the spec tree.

**Files:**
- Create: `docs/superpowers/specs/mockups/live-v7.html` (copy)
- Create: `docs/superpowers/specs/mockups/audio-review-v7.html` (copy)

**Interfaces:**
- Produces: committed reference HTML that Tasks 2, 8, 10, 11 open in a browser to match spacing/colors/labels.

- [ ] **Step 1: Copy the two approved mockups**

```bash
mkdir -p docs/superpowers/specs/mockups
cp ".superpowers/brainstorm/62897-1786922466/content/live-mockup-v7-light.html" docs/superpowers/specs/mockups/live-v7.html
cp ".superpowers/brainstorm/62897-1786922466/content/audio-review-mockup.html" docs/superpowers/specs/mockups/audio-review-v7.html
```

If the brainstorm dir no longer exists, stop and report — do not substitute other files.

- [ ] **Step 2: Verify and commit**

```bash
ls -la docs/superpowers/specs/mockups/
git add docs/superpowers/specs/mockups/
git commit -m "docs: commit approved v7 mockups as design references"
```

---

### Task 2: design.css — the v7 design system

Full rewrite of `docs/site/assets/design.css`. All page-level styles for index/sensors/live move here (the old per-page `<style>` blocks shrink to nothing in later tasks). Class names are semantic and shared.

**Files:**
- Modify: `docs/site/assets/design.css` (full replacement)
- Test: `tests/test_build_site.py` (append)

**Interfaces:**
- Produces: CSS classes consumed by Tasks 3, 4, 8, 9, 11: `.app-bar`, `.app-brand`, `.app-tabs`, `.app-status`, `.group-label`, `.panel`, `.session-cards`/`.session-card`, `.kpi-band`/`.kpi`, `.ribbon`/`.ribbon-seg`/`.ribbon-tick`/`.ribbon-playhead`, `.legend`, `.trend-row`/`.trend-label`/`.trend-chart`, `.spec-panel`, `.stage-grid`/`.stage`, `.kv-row`, `.verdict`, `.gauge`, `.fp-row`/`.fp-strip`, `.register-table`, `.badge` (+ `.sus`/`.tra`/`.state`/`.agree`/`.warn`/`.ev`), `.transport`, `.seg-tabs`, `.pill`, `.clip-card`, `.notice`, `.site-footer`.
- CSS custom properties every page relies on (exact values from spec §3): see `:root` block below.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_build_site.py`:

```python
def test_design_css_v7_tokens_and_no_legacy_look() -> None:
    css = (Path(__file__).resolve().parents[1] / "docs" / "site" / "assets" / "design.css").read_text()
    # v7 tokens (spec §3)
    for token in [
        "--paper: #e9ecf0", "--panel: #ffffff", "--ink: #18202a", "--hair: #d3d9e0",
        "--live: #0e8f6f", "--alarm: #c73a1d", "--warn-text: #a16207", "--warn-fill: #c07f10",
        "--s-turbine: #2563a8", "--s-pump: #7c4dbc", "--s-phase: #1d8a70",
        "--s-standstill: #6b7684", "--s-transition: #c07f10", "--s-unknown: #aab2bc",
    ]:
        assert token in css, token
    # v7 components exist
    for cls in [".app-bar", ".group-label", ".session-card", ".kpi-band", ".trend-row",
                ".stage-grid", ".register-table", ".transport", ".seg-tabs", ".clip-card"]:
        assert cls in css, cls
    # legacy look must be gone
    assert "Avenir Next" not in css
    assert "--paper: #eceef0" not in css
    assert ".topbar" not in css
```

Add `from pathlib import Path` to the test module imports if not present.

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_build_site.py::test_design_css_v7_tokens_and_no_legacy_look -v`
Expected: FAIL (`--paper: #e9ecf0` missing).

- [ ] **Step 3: Replace design.css with the v7 system**

Write `docs/site/assets/design.css` with exactly this content (port pixel details from `docs/superpowers/specs/mockups/live-v7.html` where a judgement call arises — the mockup wins):

```css
/* ROWII Monitor -- v7 design system: light monitoring software.
   Single stylesheet for every docs/site/*.html page. No external requests.
   Color roles are strictly functional (spec D6): red = alarm only,
   amber = transition/caution/EVENTS, green = ok/live/agreement.
   Every numeral is mono + tabular (spec D7 pairs each number with a unit). */

:root {
  color-scheme: light;
  --paper: #e9ecf0;          /* page background (cool technical gray) */
  --panel: #ffffff;
  --panel-2: #f7f9fb;        /* chart backgrounds inside panels */
  --panel-3: #fbfcfd;
  --ink: #18202a;
  --dim: #5c6b7d;
  --faint: #7a8798;
  --fainter: #97a2b0;
  --hair: #d3d9e0;
  --hair-2: #dfe4ea;
  --hair-3: #eef1f5;

  --live: #0e8f6f;
  --alarm: #c73a1d;
  --warn-text: #a16207;
  --warn-fill: #c07f10;
  --warn-border: #c9a227;

  --s-turbine: #2563a8;
  --s-pump: #7c4dbc;
  --s-phase: #1d8a70;
  --s-standstill: #6b7684;
  --s-transition: #c07f10;
  --s-unknown: #aab2bc;

  --font-ui: "Helvetica Neue", Helvetica, Arial, sans-serif;
  --font-mono: ui-monospace, "SF Mono", Menlo, Consolas, monospace;
  --radius: 4px;
  --shadow: 0 1px 2px rgba(24, 32, 42, .04);
}

* { box-sizing: border-box; }
html, body { height: 100%; }
body { margin: 0; background: var(--paper); color: var(--ink);
  font: 14px/1.5 var(--font-ui); -webkit-font-smoothing: antialiased; }
h1, h2, h3, h4, p { margin: 0 0 .5em; }
a { color: var(--ink); text-decoration-color: var(--hair); text-underline-offset: 2px; }
code, .mono, .num { font-family: var(--font-mono); font-variant-numeric: tabular-nums; font-size: .95em; }
::selection { background: #cfe0de; }

main.page { max-width: 1200px; margin: 0 auto; padding: 0 0 40px; }

/* ------------------------------------------------------------- app bar */
.app-bar { display: flex; align-items: center; gap: 18px; padding: 10px 18px;
  background: var(--panel); border-bottom: 1px solid var(--hair);
  position: sticky; top: 0; z-index: 40; }
.app-brand { font-size: 15px; letter-spacing: .04em; white-space: nowrap; }
.app-brand b { font-weight: 800; }
.app-brand span { font-weight: 300; color: var(--dim); }
.app-tabs { display: flex; gap: 2px; margin-left: 6px; flex-wrap: wrap; }
.app-tabs a { font-size: 12px; font-weight: 600; color: var(--dim); padding: 6px 11px;
  border-radius: var(--radius); text-decoration: none; white-space: nowrap; }
.app-tabs a:hover { color: var(--ink); background: var(--hair-3); }
.app-tabs a.active { color: var(--ink); background: var(--hair-3);
  box-shadow: inset 0 0 0 1px var(--hair); font-weight: 700; }
.app-status { margin-left: auto; display: flex; align-items: center; gap: 12px;
  font-size: 11px; color: var(--dim); white-space: nowrap; }
.app-status .rep { color: var(--live); font-weight: 700; letter-spacing: .08em; }
.app-status .clk { font-size: 13px; color: var(--ink); font-weight: 600; }

/* --------------------------------------------------------- group label */
.group-label { margin: 16px 18px 8px; display: flex; align-items: center; gap: 10px; }
.group-label .t { font-size: 10px; letter-spacing: .16em; color: var(--faint);
  font-weight: 800; white-space: nowrap; text-transform: uppercase; }
.group-label .ln { flex: 1; height: 1px; background: var(--hair); }
.group-label .cap { font-size: 10.5px; color: var(--fainter); white-space: nowrap; }
@media (max-width: 760px) { .group-label .cap { display: none; } }

/* --------------------------------------------------------------- panel */
.panel { background: var(--panel); border: 1px solid var(--hair);
  border-radius: var(--radius); box-shadow: var(--shadow); }
.panel-pad { padding: 12px 14px; }
.panel-head { display: flex; justify-content: space-between; align-items: baseline;
  margin-bottom: 9px; gap: 10px; flex-wrap: wrap; }
.panel-head .t { font-size: 10px; letter-spacing: .14em; font-weight: 700;
  color: var(--faint); text-transform: uppercase; }
.panel-head .r { font-size: 11px; color: var(--dim); }
.panel-head .r .ok { color: var(--live); font-weight: 700; }

.notice { margin: 0 18px; background: var(--panel); border: 1px solid var(--hair);
  border-left: 3px solid var(--warn-border); border-radius: var(--radius);
  padding: 10px 14px; font-size: 12.5px; color: var(--ink); box-shadow: var(--shadow); }

/* -------------------------------------------------------- session cards */
.session-cards { display: grid; grid-template-columns: repeat(4, 1fr); gap: 8px; margin: 0 18px; }
@media (max-width: 980px) { .session-cards { grid-template-columns: repeat(2, 1fr); } }
.session-card { display: block; background: var(--panel); border: 1px solid var(--hair);
  border-radius: var(--radius); padding: 9px 12px 10px; box-shadow: var(--shadow);
  text-decoration: none; color: var(--ink); }
.session-card:hover { border-color: var(--faint); }
.session-card.active { border-color: var(--ink); box-shadow: inset 3px 0 0 var(--ink), var(--shadow); }
.session-card .dt { font-size: 8.5px; letter-spacing: .06em; color: var(--faint);
  font-weight: 700; white-space: nowrap; }
.session-card .nm { font-size: 13px; font-weight: 800; margin: 2px 0 1px; letter-spacing: -.01em; }
.session-card .in { font-size: 10px; color: var(--dim); margin-bottom: 7px; }
.mini-ribbon { display: flex; height: 6px; border-radius: 2px; overflow: hidden; position: relative; }
.mini-ribbon i { height: 100%; }
.mini-ribbon .mt { position: absolute; top: -1px; width: 1.5px; height: 8px; background: var(--alarm); }

/* ------------------------------------------------------------ KPI band */
.kpi-band { display: grid; grid-template-columns: 1.3fr 1fr 1fr 1.1fr 1.1fr; gap: 8px; margin: 8px 18px 0; }
@media (max-width: 980px) { .kpi-band { grid-template-columns: repeat(2, 1fr); } }
.kpi { background: var(--panel); border: 1px solid var(--hair); border-radius: var(--radius);
  padding: 8px 11px 9px; box-shadow: var(--shadow); }
.kpi .k { font-size: 8.5px; letter-spacing: .1em; color: var(--faint); font-weight: 700;
  text-transform: uppercase; }
.kpi .v { font-size: 23px; font-weight: 300; margin-top: 5px; line-height: 1;
  white-space: nowrap; letter-spacing: -.01em; }
.kpi .v b { font-weight: 700; font-size: 16.5px; }
.kpi .v .u { font-size: 10.5px; color: var(--dim); font-weight: 500; }
.kpi .s { font-size: 8.5px; color: var(--faint); margin-top: 4px; }
.state-dot { display: inline-block; width: 9px; height: 9px; border-radius: 50%; margin-right: 6px; vertical-align: 2px; }
.text-alarm { color: var(--alarm); }

/* -------------------------------------------------- timeline instrument */
.ribbon-row { display: grid; grid-template-columns: 82px 1fr; gap: 8px; align-items: center; margin-bottom: 4px; }
.ribbon-label { font-size: 8px; letter-spacing: .05em; color: var(--faint); font-weight: 700;
  text-align: right; line-height: 1.3; text-transform: uppercase; }
.ribbon-label b { display: block; color: var(--ink); font-size: 9px; }
.ribbon { position: relative; display: flex; height: 24px; border: 1px solid #c6cdd6;
  border-radius: 2px; overflow: hidden; cursor: pointer; background: var(--panel-2); }
.ribbon.slim { height: 19px; }
.ribbon-seg { position: absolute; top: 0; bottom: 0; overflow: hidden; }
.ribbon-seg-label { display: block; font-size: 8.5px; font-weight: 700; color: rgba(255,255,255,.92);
  white-space: nowrap; overflow: hidden; text-overflow: ellipsis; padding-left: 5px;
  line-height: 24px; pointer-events: none; }
.ribbon.slim .ribbon-seg-label { line-height: 19px; }
.ribbon-tick { position: absolute; top: 0; width: 2px; height: 8px; background: var(--alarm); z-index: 3; }
.ribbon-playhead { position: absolute; top: -2px; bottom: -2px; width: 2px; background: var(--ink);
  box-shadow: 0 0 0 1px rgba(255,255,255,.75); pointer-events: none; z-index: 4; }
.legend { display: flex; flex-wrap: wrap; gap: 4px 13px; margin: 8px 0 0 90px;
  font-size: 9.5px; color: var(--dim); }
.legend .sw { display: inline-block; width: 8px; height: 8px; border-radius: 2px;
  margin-right: 4px; vertical-align: -1px; }
.legend .td { display: inline-block; width: 2px; height: 8px; background: var(--alarm);
  margin-right: 4px; vertical-align: -1px; }
.legend .agree { color: var(--live); font-weight: 700; }
.trend-wrap { position: relative; margin-top: 8px; }
.trend-row { display: grid; grid-template-columns: 82px 1fr; gap: 8px; align-items: stretch; margin-bottom: 2px; }
.trend-label { display: flex; flex-direction: column; justify-content: center;
  text-align: right; line-height: 1.2; }
.trend-label .n { font-size: 7.5px; letter-spacing: .05em; color: var(--faint);
  font-weight: 700; text-transform: uppercase; }
.trend-label .lv { font-size: 11px; color: var(--ink); font-weight: 700; }
.trend-label .lv .u { font-size: 8px; color: var(--faint); font-weight: 500; }
.trend-chart { border: 1px solid var(--hair-2); background: var(--panel-2);
  border-radius: 2px; position: relative; }
.trend-chart svg { display: block; width: 100%; height: 100%; }
.trend-playhead { position: absolute; top: 0; bottom: 0; width: 2px; background: var(--ink);
  box-shadow: 0 0 0 1px rgba(255,255,255,.75); pointer-events: none; z-index: 4; }
.trend-cap { margin: 5px 0 0 90px; font-size: 9px; color: var(--fainter); }
.trend-cap b { color: var(--ink); }

/* --------------------------------------------------------- spectrogram */
.spec-panel { position: relative; height: 110px; border: 1px solid #c6cdd6; border-radius: 2px;
  overflow: hidden; background: #1a1120; }
.spec-panel img { position: absolute; top: 0; left: 0; height: 100%; image-rendering: pixelated; }
.spec-playhead { position: absolute; top: 0; bottom: 0; left: 50%; width: 2px; background: #fff;
  box-shadow: 0 0 6px rgba(255,255,255,.7); z-index: 3; }
.spec-axis { position: absolute; font-size: 7.5px; color: rgba(255,255,255,.6); z-index: 2; }

/* ------------------------------------------------------ pipeline stages */
.stage-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 8px; margin: 0 18px; }
@media (max-width: 980px) { .stage-grid { grid-template-columns: 1fr; } }
.stage { padding: 10px 12px; }
.kv-row { display: flex; justify-content: space-between; gap: 8px; font-size: 10.5px;
  padding: 4px 0; border-bottom: 1px solid var(--hair-3); }
.kv-row:last-child { border-bottom: 0; }
.kv-row .k { color: var(--dim); }
.kv-row .v { color: var(--ink); font-weight: 700; text-align: right; }
.verdict { font-size: 9.5px; color: var(--dim); line-height: 1.45; margin-top: 7px;
  padding-top: 7px; border-top: 1px solid var(--hair-3); }
.verdict b { color: var(--live); }
.verdict.fired b { color: var(--alarm); }
.gauge { position: relative; height: 8px; background: var(--hair-3); border: 1px solid var(--hair-2);
  border-radius: 5px; margin: 7px 0 2px; }
.gauge .fill { position: absolute; left: 0; top: 0; bottom: 0; background: var(--live);
  border-radius: 5px 0 0 5px; }
.gauge.fired .fill { background: var(--alarm); }
.gauge .mark { position: absolute; top: -3px; bottom: -3px; width: 2px; background: var(--ink); }
.fp-row { display: grid; grid-template-columns: 130px 1fr 46px; gap: 7px; align-items: center; padding: 2.5px 0; }
.fp-row .nm { font-size: 8.5px; color: var(--dim); text-align: right; line-height: 1.2; }
.fp-bar { position: relative; height: 8px; background: var(--hair-3);
  border: 1px solid var(--hair-2); border-radius: 2px; }
.fp-bar::before { content: ""; position: absolute; left: 50%; top: -2px; bottom: -2px;
  width: 1px; background: #b9c2cd; }
.fp-bar i { position: absolute; top: 0; bottom: 0; border-radius: 1px; }
.fp-bar i.pos { background: var(--warn-fill); left: 50%; }
.fp-bar i.neg { background: var(--s-turbine); right: 50%; }
.fp-val { font-size: 9px; color: var(--ink); font-weight: 700; }
.fp-strip-wrap { border: 1px solid var(--hair-2); background: var(--panel-2);
  border-radius: 2px; margin-top: 7px; }
.fp-strip-wrap canvas { display: block; width: 100%; height: 18px; }
.fp-note { font-size: 8px; color: var(--fainter); margin-top: 4px; line-height: 1.4; }

/* ------------------------------------------------------- alarm register */
.register-table { width: 100%; border-collapse: collapse; font-size: 11px; }
.register-table th { text-align: left; font-size: 8.5px; letter-spacing: .08em; color: var(--faint);
  font-weight: 700; border-bottom: 1px solid var(--hair); padding: 5px 8px 4px;
  text-transform: uppercase; }
.register-table td { padding: 6px 8px; border-bottom: 1px solid var(--hair-3);
  vertical-align: top; color: #33404d; }
.register-table tr:last-child td { border-bottom: 0; }
.register-table tr.acked { opacity: .45; }
.register-table tr.future { opacity: .35; }
.register-table .listen { font-size: 9px; color: var(--ink); font-weight: 700;
  letter-spacing: .04em; white-space: nowrap; border-bottom: 1px solid var(--ink); cursor: pointer; }

/* --------------------------------------------------------------- badges */
.badge { display: inline-block; font-size: 8px; font-weight: 800; letter-spacing: .05em;
  padding: 1.5px 6px; border-radius: 2px; border: 1px solid; white-space: nowrap;
  text-transform: uppercase; }
.badge.sus { color: var(--warn-text); border-color: var(--warn-border); }
.badge.tra { color: var(--alarm); border-color: var(--alarm); }
.badge.state { color: var(--s-turbine); border-color: var(--s-turbine); }
.badge.agree { color: var(--live); border-color: var(--live); }
.badge.warn { color: var(--warn-text); border-color: var(--warn-border); }
.badge.ev { color: var(--warn-text); border-color: var(--warn-border); margin-left: 5px;
  vertical-align: 1.5px; font-size: 7px; padding: 0 4px; }

/* ------------------------------------------------------------ transport */
.transport { position: sticky; bottom: 0; z-index: 30; display: flex; align-items: center;
  gap: 12px; margin-top: 14px; background: var(--panel); padding: 10px 18px;
  border-top: 1px solid var(--hair); box-shadow: 0 -3px 10px rgba(24,32,42,.06); flex-wrap: wrap; }
.transport .btn-play { background: var(--ink); color: #fff; font-weight: 800; font-size: 12px;
  padding: 7px 17px; border-radius: 3px; border: 0; cursor: pointer; min-width: 76px; }
.transport .btn-speed { font: 700 10px/1 var(--font-mono); padding: 5px 9px;
  border: 1px solid var(--hair); border-radius: 3px; color: var(--dim); background: var(--panel); cursor: pointer; }
.transport .btn-speed.active { background: var(--hair-3); color: var(--ink);
  border-color: var(--faint); }
.transport .sep { width: 1px; align-self: stretch; background: var(--hair); }
.transport .listen-lb { font-size: 8.5px; letter-spacing: .12em; color: var(--faint);
  font-weight: 700; }
.transport .mic { font-size: 10.5px; padding: 4px 10px; border: 1px solid var(--hair);
  border-radius: 999px; color: var(--dim); background: var(--panel); cursor: pointer; }
.transport .mic.active { background: var(--live); border-color: var(--live); color: #fff; font-weight: 700; }
.transport .vol { display: flex; align-items: center; gap: 5px; font-size: 9px; color: var(--faint); }
.transport .vol input[type="range"] { width: 70px; accent-color: var(--ink); }
.transport .tm { margin-left: auto; font-size: 11px; color: var(--dim); }

/* -------------------------------------------- audio & review page bits */
.seg-tabs { display: flex; border: 1px solid var(--hair); border-radius: 5px;
  overflow: hidden; background: var(--panel); }
.seg-tabs button { font: 700 11px/1 var(--font-ui); padding: 8px 13px; color: var(--dim);
  border: 0; border-right: 1px solid var(--hair); background: var(--panel); cursor: pointer; }
.seg-tabs button:last-child { border-right: 0; }
.seg-tabs button.active { background: var(--ink); color: #fff; }
.seg-tabs button .n { font-weight: 500; opacity: .65; margin-left: 4px; }
.pill { font-size: 10px; font-weight: 600; color: var(--dim); border: 1px solid var(--hair);
  background: var(--panel); padding: 4px 10px; border-radius: 999px; cursor: pointer; }
.pill.active { color: var(--ink); border-color: var(--faint); background: var(--hair-3); font-weight: 700; }
.toolbar { display: flex; align-items: center; gap: 10px; margin: 0 18px 10px; flex-wrap: wrap; }
.toolbar .right { margin-left: auto; display: flex; align-items: center; gap: 10px; }
.progress-note { font-size: 10.5px; color: var(--dim); }
.progress-note b { color: var(--live); }
.btn-export { font-size: 10px; font-weight: 800; letter-spacing: .06em; color: #fff;
  background: var(--ink); padding: 7px 13px; border-radius: 3px; border: 0; cursor: pointer; }
.clip-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px; margin: 0 18px; }
@media (max-width: 980px) { .clip-grid { grid-template-columns: 1fr; } }
.clip-card { background: var(--panel); border: 1px solid var(--hair); border-radius: var(--radius);
  padding: 10px 12px 11px; box-shadow: var(--shadow); }
.clip-card .ct { display: flex; justify-content: space-between; gap: 8px; align-items: baseline;
  margin-bottom: 6px; }
.clip-card .ct .n2 { font-size: 11.5px; font-weight: 800; }
.clip-card audio { width: 100%; height: 30px; margin-top: 4px; }
.clip-card .meta { font-size: 8.5px; color: var(--faint); margin-top: 5px; }

/* ------------------------------------------------------ index / sensors */
.cards { display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px; margin: 0 18px; }
@media (max-width: 980px) { .cards { grid-template-columns: 1fr; } }
a.card { display: block; background: var(--panel); border: 1px solid var(--hair);
  border-radius: var(--radius); padding: 14px 15px 15px; text-decoration: none;
  color: var(--ink); box-shadow: var(--shadow); }
a.card:hover { border-color: var(--faint); }
a.card h2 { font-size: 14px; margin-bottom: 5px; }
a.card p { color: var(--dim); font-size: 12px; margin: 0; }
a.card .go { display: block; margin-top: 10px; font-size: 10px; font-weight: 700;
  color: var(--live); letter-spacing: .04em; text-transform: uppercase; }
.lede { margin: 0 18px; font-size: 13px; color: var(--dim); max-width: 86ch; line-height: 1.6; }

.site-footer { max-width: 1200px; margin: 0 auto; padding: 22px 18px 44px;
  color: var(--faint); font-size: 11px; border-top: 1px solid var(--hair); }

/* state colors as utilities (closed vocabulary) */
.st-turbine { color: var(--s-turbine); } .bg-turbine { background: var(--s-turbine); }
.st-pump { color: var(--s-pump); } .bg-pump { background: var(--s-pump); }
.st-phase-shifter { color: var(--s-phase); } .bg-phase-shifter { background: var(--s-phase); }
.st-standstill { color: var(--s-standstill); } .bg-standstill { background: var(--s-standstill); }
.st-transition { color: var(--s-transition); } .bg-transition { background: var(--s-transition); }
.st-unknown { color: var(--s-unknown); } .bg-unknown { background: var(--s-unknown); }
.text-live { color: var(--live); } .text-warn { color: var(--warn-text); }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_build_site.py::test_design_css_v7_tokens_and_no_legacy_look -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add docs/site/assets/design.css tests/test_build_site.py
git commit -m "feat(site): v7 design system stylesheet (light monitoring)"
```

Note: from this commit until Task 8/10 land, the existing pages reference classes that no longer exist — acceptable mid-branch state; the site is rebuilt page by page.

---

### Task 3: site_common.py — v7 chrome helpers

**Files:**
- Modify: `scripts/site_common.py` (color constants at lines 34–68, `topbar_html` at line 90, `FOOTER_HTML` at 104)
- Test: `tests/test_build_site.py` (append)

**Interfaces:**
- Produces (consumed by Tasks 4, 8, 11):
  - `NAV_ITEMS: list[tuple[str, str]]` = `[("index.html", "Overview"), ("sensors.html", "Sensors"), ("live.html", "Live replay"), ("audio_review.html", "Audio & review")]`
  - `app_bar_html(active_href: str, *, status_html: str = "") -> str` — replaces `topbar_html` (delete the old name; update all callers in the same commit).
  - `group_label_html(label: str, caption: str = "") -> str`
  - Color constants updated to v7 (`PAPER = "#e9ecf0"`, `PANEL = "#ffffff"`, `INK = "#18202a"`, `DIM = "#5c6b7d"`, `HAIR = "#d3d9e0"`, `LIVE = "#0e8f6f"`, `ALARM = "#c73a1d"`, `WARN = "#a16207"`, `FONT_UI = '"Helvetica Neue",Helvetica,Arial,sans-serif'`). `state_color()` values unchanged.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_build_site.py`:

```python
def test_app_bar_html_nav_and_active() -> None:
    import site_common as sc
    html = sc.app_bar_html("live.html", status_html='<span class="rep">● REPLAY</span>')
    assert 'class="app-bar"' in html
    assert html.count("<a ") == 4
    for label in ["Overview", "Sensors", "Live replay", "Audio &amp; review"]:
        assert label in html
    assert 'href="live.html" class="active"' in html
    assert "● REPLAY" in html
    assert "snippets.html" not in html and 'href="review.html"' not in html


def test_group_label_html() -> None:
    import site_common as sc
    html = sc.group_label_html("Session timeline", "full day · click or drag to seek")
    assert 'class="group-label"' in html
    assert "SESSION TIMELINE" in html.upper()
    assert "full day · click or drag to seek" in html
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_build_site.py::test_app_bar_html_nav_and_active tests/test_build_site.py::test_group_label_html -v`
Expected: FAIL (`AttributeError: module 'site_common' has no attribute 'app_bar_html'`).

- [ ] **Step 3: Implement**

In `scripts/site_common.py`, update the color/font constants to the v7 values listed in Interfaces, then replace `topbar_html` with:

```python
NAV_ITEMS: list[tuple[str, str]] = [
    ("index.html", "Overview"),
    ("sensors.html", "Sensors"),
    ("live.html", "Live replay"),
    ("audio_review.html", "Audio &amp; review"),
]


def app_bar_html(active_href: str, *, status_html: str = "") -> str:
    """v7 app bar: brand + tab nav + optional right-side status block.

    `active_href` marks the active tab. Any `live-*.html` page counts as
    `live.html` so per-session replay pages highlight the Live tab."""
    active = "live.html" if active_href.startswith("live") else active_href
    links = "".join(
        f'<a href="{href}"{" class=\"active\"" if href == active else ""}>{label}</a>'
        for href, label in NAV_ITEMS
    )
    status = f'<div class="app-status">{status_html}</div>' if status_html else ""
    return (
        '<header class="app-bar">'
        '<div class="app-brand"><b>ROWII</b><span>&thinsp;MONITOR</span></div>'
        f'<nav class="app-tabs">{links}</nav>{status}</header>'
    )


def group_label_html(label: str, caption: str = "") -> str:
    """v7 group label row: uppercase label + hairline + optional right caption."""
    cap = f'<span class="cap">{caption}</span>' if caption else ""
    return (
        f'<div class="group-label"><span class="t">{label}</span>'
        f'<span class="ln"></span>{cap}</div>'
    )
```

Then update every `topbar_html(` caller found via `grep -rn "topbar_html" scripts/` to `app_bar_html(` (same argument). `FOOTER_HTML` keeps its wording but swap its class to `site-footer` if not already.

- [ ] **Step 4: Run the full build_site test file**

Run: `.venv/bin/python -m pytest tests/test_build_site.py -v`
Expected: the two new tests PASS. If existing tests assert old topbar markup (`grep -n "topbar" tests/test_build_site.py`), update those assertions to the app-bar equivalents in this step — they are look-assertions, not behavior changes.

- [ ] **Step 5: Commit**

```bash
git add scripts/site_common.py tests/test_build_site.py
git commit -m "feat(site): shared v7 app-bar and group-label chrome"
```

---

### Task 4: build_site — index + sensors restyle, nav rename, redirect stubs

**Files:**
- Modify: `scripts/build_site.py` (`_page_shell` line 551, `render_index` line 772, `render_sensors` line 821, the build-pages entry that writes the html files)
- Test: `tests/test_build_site.py` (append)

**Interfaces:**
- Consumes: `site_common.app_bar_html`, `site_common.group_label_html` (Task 3).
- Produces: `render_redirect_stub(target: str, title: str) -> str`; `render_index()` / `render_sensors()` emitting v7 markup; build-pages now writes `snippets.html` and `review.html` as redirect stubs to `audio_review.html` and **stops** writing `review_static.html`.

- [ ] **Step 1: Write the failing tests**

```python
def test_render_index_v7() -> None:
    import build_site as bs
    html = bs.render_index()
    assert 'class="app-bar"' in html and 'class="group-label"' in html
    assert "audio_review.html" in html
    assert "snippets.html" not in html and 'href="review.html"' not in html
    assert "Research prototype, not a certified product" in html
    assert bs.find_external_resource_urls(html) == []


def test_render_redirect_stub() -> None:
    import build_site as bs
    html = bs.render_redirect_stub("audio_review.html", "Audio & review")
    assert '<meta http-equiv="refresh" content="0; url=audio_review.html">' in html
    assert '<a href="audio_review.html">' in html
    assert bs.find_external_resource_urls(html) == []
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_build_site.py::test_render_index_v7 tests/test_build_site.py::test_render_redirect_stub -v`
Expected: FAIL (`render_redirect_stub` missing; index still emits old markup).

- [ ] **Step 3: Implement**

1. `_page_shell(...)`: emit `app_bar_html(active_file)` instead of the old topbar; body wraps in `<main class="page">`; keep the single `design.css` link and the zero-external contract.
2. `render_index()`: replace the hero/cards body with v7: `group_label_html("ROWII Monitor", "acoustic condition monitoring · Rodundwerk II · measurement campaign 2026")`, a `<p class="lede">` holding the existing system description (English wording from the current hero, trimmed to ~4 sentences), the existing research-prototype `<div class="notice">` text verbatim, then `<div class="cards">` with **three** `a.card` links: Sensors, Live replay (`live.html`), Audio & review (`audio_review.html`) — reuse the existing card copy, replacing the two old audio cards with one: title "Audio & review", text "Hammer-strike tests, per-mode audio, and every model-flagged candidate with SCADA context and the expert assessment form.".
3. `render_sensors()`: wrap the existing SVG blocks in `<div class="panel panel-pad">` sections introduced by `group_label_html("Vertical section")` / `group_label_html("Plan view rings")`; drop the sentence that references `live.html`'s sensor panel (that panel is removed by Task 8); keep all marker/hover markup unchanged.
4. Add:

```python
def render_redirect_stub(target: str, title: str) -> str:
    """Tiny self-contained redirect page so old bookmarks keep working."""
    return (
        "<!doctype html>\n<html lang=\"en\">\n<head>\n<meta charset=\"utf-8\">\n"
        f"<title>{title} — ROWII Monitor</title>\n"
        f'<meta http-equiv="refresh" content="0; url={target}">\n'
        "</head>\n<body>\n"
        f'<p>This page moved to <a href="{target}">{target}</a>.</p>\n'
        "</body>\n</html>\n"
    )
```

5. In the build-pages writer: write `snippets.html` = `render_redirect_stub("audio_review.html", "Listening library")`, `review.html` = `render_redirect_stub("audio_review.html", "Candidate review")`; delete the `review_static.html` write if build-pages produced it (if it is produced elsewhere, leave that caller to Task 11). Keep writing `sensors.html`/`index.html`; **stop** rendering the old snippets body here (its clip-card fragment functions stay — Task 11 imports them).

- [ ] **Step 4: Run tests**

Run: `.venv/bin/python -m pytest tests/test_build_site.py -v`
Expected: all PASS (fix any old-markup assertions the same way as Task 3 Step 4).

- [ ] **Step 5: Commit**

```bash
git add scripts/build_site.py tests/test_build_site.py
git commit -m "feat(site): v7 index/sensors, audio_review nav, redirect stubs"
```

---

### Task 5: feature_labels.py — humanize feature names

**Files:**
- Create: `scripts/feature_labels.py`
- Test: create `tests/test_feature_labels.py`

**Interfaces:**
- Consumes: raw cache names like `RAWGeneratorMic__0::ch0_log_rms` (`results/cache/<run>--audio.npz` key `feature_names`; verified format `"<stream>::ch<N>_<feature>"`).
- Produces: `humanize_feature_name(raw: str) -> str` — used by Task 6 to ship human labels in the payload.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_feature_labels.py`:

```python
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from feature_labels import humanize_feature_name


def test_mic_channel_features() -> None:
    assert humanize_feature_name("RAWGeneratorMic__0::ch0_log_rms") == "Generator mic 1 · loudness (log RMS)"
    assert humanize_feature_name("RAWGeneratorMic__0::ch2_octave_2000") == "Generator mic 3 · octave band 2 kHz"
    assert humanize_feature_name("RAWTurbineMic__1::ch0_octave_500") == "Turbine mic 1 · octave band 500 Hz"
    assert humanize_feature_name("RAWGeneratorMic__0::ch3_spectral_centroid") == "Generator mic 4 · spectral centroid"
    assert humanize_feature_name("RAWTurbineMic__1::ch0_rolloff95") == "Turbine mic 1 · spectral rolloff (95 %)"


def test_machine_bands() -> None:
    assert humanize_feature_name("RAWGeneratorMic__0::ch1_band_shaft") == "Generator mic 2 · shaft band"
    assert humanize_feature_name("RAWGeneratorMic__0::ch0_band_blade_pass") == "Generator mic 1 · blade-pass band"
    assert humanize_feature_name("RAWGeneratorMic__0::ch0_band_guide_vane_pass") == "Generator mic 1 · guide-vane-pass band"


def test_vibration_features() -> None:
    assert humanize_feature_name("RAWGeneratorVib__2::ch1_log_rms") == "Generator vibration ch 2 · level (log RMS)"
    assert humanize_feature_name("RAWTurbineVib__3::ch0_kurtosis") == "Turbine vibration ch 1 · impulsiveness (kurtosis)"
    assert humanize_feature_name("RAWTurbineVib__3::ch2_band_shaft") == "Turbine vibration ch 3 · shaft band"


def test_unknown_falls_back_to_raw() -> None:
    assert humanize_feature_name("weird") == "weird"
    assert humanize_feature_name("RAWGeneratorMic__0::ch0_totally_new") == "Generator mic 1 · totally new"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_feature_labels.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'feature_labels'`.

- [ ] **Step 3: Implement `scripts/feature_labels.py`**

```python
"""Human-readable labels for handcrafted feature names (spec D9).

Raw names come from the feature caches (`results/cache/<run>--{audio,vibration}
.npz`, key `feature_names`) in the form `"<stream>::ch<N>_<feature>"`, e.g.
`RAWGeneratorMic__0::ch0_octave_2000`. The site shows these as
`"Generator mic 1 · octave band 2 kHz"`. Unknown parts fall back to a cleaned
version of the raw text — never raise for a new feature name.
"""
from __future__ import annotations

import re

_STREAM_LABELS: dict[str, tuple[str, str]] = {
    # stream-name substring -> (display base, channel noun)
    "GeneratorMic": ("Generator mic", ""),
    "TurbineMic": ("Turbine mic", ""),
    "GeneratorVib": ("Generator vibration", "ch "),
    "TurbineVib": ("Turbine vibration", "ch "),
}

_FEATURE_LABELS: dict[str, str] = {
    "log_rms": "loudness (log RMS)",
    "spectral_centroid": "spectral centroid",
    "rolloff95": "spectral rolloff (95 %)",
    "kurtosis": "impulsiveness (kurtosis)",
    "band_shaft": "shaft band",
    "band_blade_pass": "blade-pass band",
    "band_guide_vane_pass": "guide-vane-pass band",
}

_VIB_FEATURE_OVERRIDES: dict[str, str] = {
    "log_rms": "level (log RMS)",
}

_OCTAVE_RE = re.compile(r"^octave_(\d+)$")
_CH_RE = re.compile(r"^ch(\d+)_(.+)$")


def _octave_label(center_hz: int) -> str:
    if center_hz >= 1000:
        khz = center_hz / 1000.0
        num = f"{khz:.1f}".rstrip("0").rstrip(".")
        return f"octave band {num} kHz"
    return f"octave band {center_hz} Hz"


def _band_label(feature: str, *, vib: bool) -> str:
    m = _OCTAVE_RE.match(feature)
    if m:
        return _octave_label(int(m.group(1)))
    if vib and feature in _VIB_FEATURE_OVERRIDES:
        return _VIB_FEATURE_OVERRIDES[feature]
    if feature in _FEATURE_LABELS:
        return _FEATURE_LABELS[feature]
    if feature.startswith("band_"):
        return feature.removeprefix("band_").replace("_", "-") + " band"
    return feature.replace("_", " ")


def humanize_feature_name(raw: str) -> str:
    if "::" not in raw:
        return raw
    stream, feat = raw.split("::", 1)
    m = _CH_RE.match(feat)
    if not m:
        return raw
    ch_index = int(m.group(1)) + 1  # 1-based for humans
    feature = m.group(2)
    for key, (base, ch_noun) in _STREAM_LABELS.items():
        if key in stream:
            vib = "Vib" in key
            return f"{base} {ch_noun}{ch_index} · {_band_label(feature, vib=vib)}"
    return f"{stream} ch {ch_index} · {_band_label(feature, vib=False)}"
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_feature_labels.py -v`
Expected: PASS (all 4).

If any assertion mismatches reality (e.g. an octave center like `31` for 31.5 Hz appears in real names), adjust the label function — not the honest raw name — and extend the test with the real-world case:

```bash
.venv/bin/python -c "import numpy as np; print(list(np.load('results/cache/290626-tu--audio.npz', allow_pickle=True)['feature_names'][:40]))"
```

- [ ] **Step 5: Commit**

```bash
git add scripts/feature_labels.py tests/test_feature_labels.py
git commit -m "feat(site): humanized feature-name labels for stage-1 panel"
```

---

### Task 6: build_live_replay payload v2 (names, Q/KS, session meta)

**Files:**
- Modify: `scripts/build_live_replay.py` (`load_scada` line 314, `load_feature_snapshot` line 405, `build_payload` line 551)
- Test: create `tests/test_build_live_replay.py`

**Interfaces:**
- Consumes: `feature_labels.humanize_feature_name` (Task 5); `ck.build_extended_readout_series` (`scripts/candidate_kit.py:1551`) which already resamples `flow_net_m3s` / `ks_valve` to 1 Hz alongside power/speed.
- Produces payload keys (consumed by Task 9's JS): 
  - `payload["features"]["audio_names"]: list[str]` (135 humanized), `payload["features"]["vib_names"]: list[str]` (96 humanized)
  - `payload["scada"]["flow_1hz"]: list[float|None]`, `payload["scada"]["ks_1hz"]: list[float|None]` (same length/alignment as existing `power_1hz`/`speed_1hz` — mirror however those two are named in the existing `load_scada` return; reuse the exact existing naming pattern for the two new keys, suffix `_1hz` shown here as the pattern)
  - `payload["session"] = {"id", "display_name", "blurb", "date_label", "duration_s", "n_episodes", "events": bool}`
  - pure helper `session_summary(run: str, *, duration_s: float, n_episodes: int) -> dict` and module constant `LIVE_SESSIONS` (defined in this task, extended for pages in Task 7):

```python
LIVE_SESSIONS: dict[str, dict[str, object]] = {
    "290626-tu": {"out": "live.html", "display_name": "Turbine day",
                  "blurb": "quiet reference day", "events": False},
    "080726-pu_strikes": {"out": "live-080726-pu-strikes.html", "display_name": "Hammer-strike day",
                          "blurb": "controlled tap tests at each sensor", "events": True},
    "010726-tu1-morning": {"out": "live-010726-tu1-morning.html", "display_name": "Morning session",
                           "blurb": "turbine start-up in the morning", "events": False},
    "270626-pu_ph_pu_ph_pu_ph-1": {"out": "live-270626-cycles.html", "display_name": "Cycling day",
                                   "blurb": "pump ⇄ phase-shifter cycles", "events": False},
}
```

- [ ] **Step 1: Write the failing tests**

Create `tests/test_build_live_replay.py`:

```python
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import build_live_replay as blr


def test_live_sessions_registry() -> None:
    assert set(blr.LIVE_SESSIONS) == {
        "290626-tu", "080726-pu_strikes", "010726-tu1-morning", "270626-pu_ph_pu_ph_pu_ph-1",
    }
    assert blr.LIVE_SESSIONS["290626-tu"]["out"] == "live.html"
    outs = [s["out"] for s in blr.LIVE_SESSIONS.values()]
    assert len(outs) == len(set(outs))


def test_session_summary_shape() -> None:
    s = blr.session_summary("080726-pu_strikes", duration_s=3600.0, n_episodes=7)
    assert s["id"] == "080726-pu_strikes"
    assert s["display_name"] == "Hammer-strike day"
    assert s["events"] is True
    assert s["duration_s"] == 3600.0
    assert s["n_episodes"] == 7
    assert s["date_label"].startswith("WED · 08 JUL 2026")


def test_humanized_names_helper() -> None:
    names = blr.humanize_names(["RAWGeneratorMic__0::ch0_log_rms", "bogus"])
    assert names[0] == "Generator mic 1 · loudness (log RMS)"
    assert names[1] == "bogus"
```

`date_label`: derive weekday+date from the session id prefix `DDMMYY` (`290626` → `MON · 29 JUN 2026`) via `datetime.strptime(run[:6], "%d%m%y")` — no hardcoded weekday table.

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_build_live_replay.py -v`
Expected: FAIL (`LIVE_SESSIONS` / `session_summary` / `humanize_names` missing).

- [ ] **Step 3: Implement**

In `scripts/build_live_replay.py`:

```python
from feature_labels import humanize_feature_name  # with the other script imports


def humanize_names(raw_names: list[str]) -> list[str]:
    return [humanize_feature_name(n) for n in raw_names]


def session_summary(run: str, *, duration_s: float, n_episodes: int) -> dict[str, object]:
    meta = LIVE_SESSIONS[run]
    day = datetime.strptime(run[:6], "%d%m%y").replace(tzinfo=UTC)
    date_label = day.strftime("%a · %d %b %Y").upper()
    return {
        "id": run,
        "display_name": meta["display_name"],
        "blurb": meta["blurb"],
        "events": bool(meta["events"]),
        "date_label": date_label,
        "duration_s": float(duration_s),
        "n_episodes": int(n_episodes),
    }
```

Then wire the payload:
1. `load_feature_snapshot()` already opens `audio.npz` / `vibration.npz`: read their `feature_names` arrays and return `audio_names = humanize_names([...])`, `vib_names = humanize_names([...])` alongside the existing matrices; `build_payload()` puts them under `payload["features"]["audio_names"]` / `["vib_names"]`.
2. `load_scada(...)`: after the existing power/speed extraction, produce flow/ks 1-Hz series the same way `candidate_kit.build_extended_readout_series` does (call it, or reuse its resampling helpers `resample_channel_to_seconds` on the `flow_net_m3s` / `ks_valve` window-mean columns of the same `session_scada`); add the two lists to the returned dict using the same key style as power/speed (open the function and mirror the existing names exactly; nulls where no data).
3. `build_payload()`: add `payload["session"] = session_summary(RUN, duration_s=<the payload's existing duration value>, n_episodes=len(alerts))` where `alerts` is the existing `load_alerts(...)` result.

- [ ] **Step 4: Run tests**

Run: `.venv/bin/python -m pytest tests/test_build_live_replay.py tests/test_feature_labels.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/build_live_replay.py tests/test_build_live_replay.py
git commit -m "feat(live): payload v2 — humanized feature names, Q/KS series, session meta"
```

---

### Task 7: build_live_replay — one page per session (two-pass build)

**Files:**
- Modify: `scripts/build_live_replay.py` (module constants lines 83–102, `main` line 596, every `RUN`-reading function)
- Test: `tests/test_build_live_replay.py` (append)

**Interfaces:**
- Consumes: Task 6's `LIVE_SESSIONS`, `session_summary`.
- Produces:
  - CLI: `.venv/bin/python scripts/build_live_replay.py [--run RUN ...]` (default: all four registry runs).
  - `pages_nav(summaries: dict[str, dict], timelines: dict[str, list]) -> list[dict]` — the cross-session switcher payload: for each registry session, `{"id", "href", "display_name", "date_label", "duration_s", "n_episodes", "events", "blurb", "ribbon": [{"start_frac": float, "end_frac": float, "state": str}...], "ticks": [float fracs]}`.
  - Each emitted page's payload gains `payload["sessions_nav"]` (all four, current one included) — consumed by Task 9.
  - Emits `docs/site/<out>` per `LIVE_SESSIONS[...]["out"]`.

- [ ] **Step 1: Write the failing test for the nav builder**

Append to `tests/test_build_live_replay.py`:

```python
def test_pages_nav_fractions_and_href() -> None:
    summaries = {
        "290626-tu": {"id": "290626-tu", "display_name": "Turbine day", "blurb": "quiet reference day",
                      "events": False, "date_label": "MON · 29 JUN 2026", "duration_s": 100.0,
                      "n_episodes": 2},
    }
    timelines = {
        "290626-tu": {
            "segments": [
                {"start_s": 0.0, "end_s": 40.0, "state_name": "standstill"},
                {"start_s": 40.0, "end_s": 100.0, "state_name": "turbine"},
            ],
            "tick_s": [50.0],
        },
    }
    nav = blr.pages_nav(summaries, timelines)
    assert nav[0]["href"] == "live.html"
    assert nav[0]["ribbon"] == [
        {"start_frac": 0.0, "end_frac": 0.4, "state": "standstill"},
        {"start_frac": 0.4, "end_frac": 1.0, "state": "turbine"},
    ]
    assert nav[0]["ticks"] == [0.5]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_build_live_replay.py::test_pages_nav_fractions_and_href -v`
Expected: FAIL (`pages_nav` missing).

- [ ] **Step 3: Implement the refactor**

1. Replace module-level `RUN`/`REGIME`/derived-path constants (lines 83–102) with a context object:

```python
@dataclasses.dataclass(frozen=True)
class RunContext:
    run: str
    regime: str
    monitor_dir: Path
    audio_meta_json: Path
    # ...one field per formerly module-level derived path

def make_run_context(run: str) -> RunContext:
    regime = ck.REGIME_BY_SESSION[run]
    ...
```

Thread `ctx: RunContext` through every function that read the old globals (`load_primary_timeline`, `load_sentinel`, `load_levels`, `load_scada`, `load_logmel_strip`, `load_feature_snapshot`, `load_alerts`, `load_audio`, `build_payload`). Mechanical parameter threading — no behavior change per session `290626-tu`.

2. Add the nav builder:

```python
def pages_nav(summaries: dict[str, dict], timelines: dict[str, dict]) -> list[dict]:
    nav = []
    for run, meta in LIVE_SESSIONS.items():
        if run not in summaries:
            continue
        s = summaries[run]
        tl = timelines[run]
        dur = float(s["duration_s"]) or 1.0
        ribbon = [
            {"start_frac": round(seg["start_s"] / dur, 4),
             "end_frac": round(seg["end_s"] / dur, 4),
             "state": seg["state_name"]}
            for seg in tl["segments"]
        ]
        ticks = [round(t / dur, 4) for t in tl["tick_s"]]
        nav.append({"id": run, "href": meta["out"], "display_name": s["display_name"],
                    "date_label": s["date_label"], "duration_s": s["duration_s"],
                    "n_episodes": s["n_episodes"], "events": s["events"],
                    "blurb": s["blurb"], "ribbon": ribbon, "ticks": ticks})
    return nav
```

(`segments` / `tick_s` are extracted from each run's already-built payload: segments from the primary timeline's state segments, ticks from the alert start times — pull the exact field names from `load_primary_timeline` / `load_alerts` when threading `ctx`.)

3. Two-pass `main`: parse `--run` (repeatable, default all registry runs; unknown run → error listing valid ids). **Preflight** per run: check the required inputs exist (`results/cache/<run>--{audio,vibration,logmel,audio-beats}.npz`, monitor dir per `make_run_context`, `candidates.csv` rows) — if any are missing, abort with a message naming the missing path and the producing script; do not build partial pages. Pass 1 builds every payload (collect `summaries`, `timelines`). Pass 2 injects `payload["sessions_nav"] = pages_nav(...)` into each and calls the existing `render_live_html(payload, template, docs/site/<out>)`.

4. Audio assets: for each run, `build_live_audio` artifacts (`docs/site/assets/live/<run>_{gen,tur}.m4a` + meta json) are part of the preflight; if missing, the abort message says to run `.venv/bin/python scripts/build_live_audio.py --run <run>` first (open that script and confirm/adjust its actual CLI once during implementation; if it has no per-run CLI, add `--run` there the same way, keeping its default behavior identical).

- [ ] **Step 4: Run tests + single-session smoke**

```bash
.venv/bin/python -m pytest tests/test_build_live_replay.py -v
.venv/bin/python scripts/build_live_replay.py --run 290626-tu
```

Expected: tests PASS; the build regenerates `docs/site/live.html` (payload now includes `session`, `sessions_nav`, feature names, Q/KS). It is fine that the page still renders with the OLD template until Task 8.

- [ ] **Step 5: Commit**

```bash
git add scripts/build_live_replay.py scripts/build_live_audio.py tests/test_build_live_replay.py
git commit -m "feat(live): per-session replay builds with cross-session nav (two-pass)"
```

---

### Task 8: live_template.html — v7 markup

Full template replacement. All styling lives in `design.css` (Task 2) — the template's `<style>` block shrinks to a few page-layout lines. Keep EXACT element ids listed below; `live.js` (Task 9) binds to them.

**Files:**
- Modify: `docs/site/live_template.html` (full replacement)

**Interfaces:**
- Consumes: `design.css` classes (Task 2); payload token `__LIVE_DATA_JSON__` (unchanged).
- Produces element ids for Task 9 — **kept from today:** `simDate`, `simClock`, `stateDotKpi`, `stateNameKpi`, `stateSinceKpi`, `stateConfKpi`, `powerKpi`, `speedKpi`, `activeAlarmsKpi`, `totalAlarmsKpi`, `farKpi`, `farKpiSub`, `ribbonWrap`, `ribbonTrackDetected`, `ribbonTrackScada`, `ribbonPlayhead`, `ribbonStart`, `ribbonEnd`, `logmelViewport`, `logmelStrip`, `pvalueChartWrap`, `s1State`, `s1Cluster`, `s1Since`, `s1Confidence`, `s1SentinelRate`, `s1SentinelThreshold`, `sentinelGaugeFill`, `sentinelGaugeMark`, `sentinelGaugeVal`, `s1Decision`, `s2Pvalue`, `s2Score`, `s2Threshold`, `s2NearTransition`, `s2Alarm`, `alarmList`, `playBtn`, `transportReadout`, `audioGen`, `audioTur`, `listenVolume`, `headRunLabel`;
  **new:** `sessionCards`, `scadaTrendRows`, `trendValP`, `trendValN`, `trendValQ`, `trendValKS`, `trendPlayhead`, `topFeatures`, `fpStripCanvas`, `listenMics` (container for the three `.mic` buttons with `data-listen="muted|gen|tur"`), `volPct`, `specListenLabel`, `agreeLine`;
  **removed (and their JS paths in Task 9):** `ringsRow`, `sensorHealth`, `listenToggle`, `listenHint`, `listenVolumeRow`, `beatsGaugeFill`, `beatsGaugeVal`, `beatsDim`, `audioDim`, `vibDim`, `fusionDim`, `featureCanvas`, `eraChip`, `sentinelChip`, `sentinelLed`, `unitName`, `s1ScadaState`, `s1LoadBin`, `s1Agree`, `scadaTrend`, `scadaTrendCaption`, `feedLegend`, `transportSizeNote`.

- [ ] **Step 1: Replace the template**

Write `docs/site/live_template.html` with exactly:

```html
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Live replay — ROWII Monitor</title>
<link rel="stylesheet" href="assets/design.css">
<style>
main.live-page { max-width: 1200px; margin: 0 auto; padding: 0 0 0; }
.spec-panel { height: 118px; }
</style>
</head>
<body>

<header class="app-bar">
  <div class="app-brand"><b>ROWII</b><span>&thinsp;MONITOR</span></div>
  <nav class="app-tabs">
    <a href="index.html">Overview</a>
    <a href="sensors.html">Sensors</a>
    <a href="live.html" class="active">Live replay</a>
    <a href="audio_review.html">Audio &amp; review</a>
  </nav>
  <div class="app-status">
    <span class="rep">● REPLAY</span>
    <span id="simDate">—</span>
    <span class="clk mono" id="simClock">--:--:-- UTC</span>
  </div>
</header>

<main class="live-page">

<div class="group-label"><span class="t">Recording</span><span class="ln"></span>
  <span class="cap">replay of recorded data — not a live feed · <span id="headRunLabel">—</span></span></div>
<div class="session-cards" id="sessionCards"><!-- injected from payload.sessions_nav --></div>

<div class="kpi-band">
  <div class="kpi">
    <div class="k">Operating state — step 1</div>
    <div class="v"><span class="state-dot" id="stateDotKpi"></span><b id="stateNameKpi">—</b></div>
    <div class="s"><span id="stateSinceKpi">since —</span> · <span id="stateConfKpi">—</span></div>
  </div>
  <div class="kpi">
    <div class="k">Active power P</div>
    <div class="v mono" id="powerKpi">— <span class="u">MW</span></div>
    <div class="s">SCADA, 1 Hz mean</div>
  </div>
  <div class="kpi">
    <div class="k">Shaft speed n</div>
    <div class="v mono" id="speedKpi">— <span class="u">rpm</span></div>
    <div class="s">turbine +, pump −</div>
  </div>
  <div class="kpi">
    <div class="k">Alarm episodes today</div>
    <div class="v mono" id="totalAlarmsKpi">— <span class="u">episodes</span></div>
    <div class="s"><span class="text-alarm" id="activeAlarmsKpi">—</span> · rows in register</div>
  </div>
  <div class="kpi">
    <div class="k">False-alarm rate</div>
    <div class="v mono" id="farKpi">— <span class="u">% of windows</span></div>
    <div class="s" id="farKpiSub">realized · budget α = 5 %</div>
  </div>
</div>

<div class="group-label"><span class="t">Session timeline</span><span class="ln"></span>
  <span class="cap">full day · click or drag to seek · top: detector, bottom: SCADA rules</span></div>
<div class="panel panel-pad" style="margin: 0 18px;">
  <div class="ribbon-row">
    <div class="ribbon-label"><b>Detector</b>step 1</div>
    <div class="ribbon" id="ribbonWrapDetected"><div class="ribbon-track" id="ribbonTrackDetected"></div></div>
  </div>
  <div class="ribbon-row">
    <div class="ribbon-label"><b>SCADA</b>rule-based</div>
    <div class="ribbon slim" id="ribbonWrapScada"><div class="ribbon-track" id="ribbonTrackScada"></div></div>
  </div>
  <div class="legend">
    <span><span class="sw bg-turbine"></span>Turbine</span>
    <span><span class="sw bg-pump"></span>Pump</span>
    <span><span class="sw bg-phase-shifter"></span>Phase-shifter</span>
    <span><span class="sw bg-standstill"></span>Standstill</span>
    <span><span class="sw bg-transition"></span>Transition</span>
    <span><span class="td"></span>Alarm episode → register</span>
    <span class="agree" id="agreeLine">—</span>
  </div>
  <div class="trend-wrap" id="scadaTrendRows">
    <div class="trend-row">
      <div class="trend-label"><span class="n">P · active power</span>
        <span class="lv mono" id="trendValP">— <span class="u">MW</span></span></div>
      <div class="trend-chart" style="height:38px" data-channel="power"></div>
    </div>
    <div class="trend-row">
      <div class="trend-label"><span class="n">n · shaft speed</span>
        <span class="lv mono" id="trendValN">— <span class="u">rpm</span></span></div>
      <div class="trend-chart" style="height:26px" data-channel="speed"></div>
    </div>
    <div class="trend-row">
      <div class="trend-label"><span class="n">Q · net flow</span>
        <span class="lv mono" id="trendValQ">— <span class="u">m³/s</span></span></div>
      <div class="trend-chart" style="height:26px" data-channel="flow"></div>
    </div>
    <div class="trend-row" style="margin-bottom:0">
      <div class="trend-label"><span class="n">KS · spherical valve</span>
        <span class="lv mono" id="trendValKS">— <span class="u">pos</span></span></div>
      <div class="trend-chart" style="height:26px" data-channel="ks"></div>
    </div>
    <div class="trend-playhead" id="trendPlayhead"></div>
  </div>
  <div class="trend-cap">values left of each curve = <b>at the playhead</b> · curves = the whole session, same time axis as the ribbons · timeline row markers: <span class="mono" id="ribbonStart">00:00:00</span>–<span class="mono" id="ribbonEnd">00:00:00</span></div>
  <div class="ribbon-playhead" id="ribbonPlayhead" style="display:none"></div>
</div>

<div class="group-label"><span class="t">Spectrogram</span><span class="ln"></span>
  <span class="cap">what you hear, as an image — bright = loud · pick mic &amp; volume in the player bar</span></div>
<div class="panel panel-pad" style="margin: 0 18px;">
  <div class="panel-head">
    <span class="t">Log-mel · ±30 s around playhead</span>
    <span class="r mono">listening: <span class="ok" id="specListenLabel">muted</span> · 64 mel bands · 0–8 kHz</span>
  </div>
  <div class="spec-panel" id="logmelViewport">
    <img id="logmelStrip" alt="Scrolling log-mel spectrogram">
    <span class="spec-axis" style="top:5px;left:5px">8 kHz</span>
    <span class="spec-axis" style="bottom:4px;left:5px">0 kHz</span>
    <span class="spec-axis" style="bottom:4px;left:44%">−30 s</span>
    <span class="spec-axis" style="bottom:4px;right:5px">+30 s</span>
    <div class="spec-playhead"></div>
  </div>
</div>

<div class="group-label"><span class="t">Pipeline diagnostics</span><span class="ln"></span>
  <span class="cap">the same second under the playhead, stage by stage</span></div>
<div class="stage-grid">
  <div class="panel stage">
    <div class="panel-head"><span class="t">Stage 1 — input features</span><span class="r mono">231 features</span></div>
    <div id="topFeatures"><!-- injected: .fp-row entries --></div>
    <div class="fp-strip-wrap"><canvas id="fpStripCanvas" height="18"></canvas></div>
    <div class="fp-note">most deviating of 231 features, in σ vs. this session's mean · full vector below — audio (135) | vibration (96), orange = deviating</div>
  </div>
  <div class="panel stage">
    <div class="panel-head"><span class="t">Stage 2 — state &amp; drift sentinel</span></div>
    <div class="kv-row"><span class="k">Detected state</span><span class="v" id="s1State">—</span></div>
    <div class="kv-row"><span class="k">Mode bank</span><span class="v" id="s1Cluster">—</span></div>
    <div class="kv-row"><span class="k">State since</span><span class="v mono" id="s1Since">—</span></div>
    <div class="kv-row"><span class="k">Confidence</span><span class="v" id="s1Confidence">—</span></div>
    <div class="kv-row"><span class="k">Windows fitting no mode</span><span class="v mono"><span id="s1SentinelRate">—</span> / budget <span id="s1SentinelThreshold">—</span></span></div>
    <div class="gauge" id="sentinelGauge"><span class="fill" id="sentinelGaugeFill"></span><span class="mark" id="sentinelGaugeMark"></span></div>
    <div class="kv-row"><span class="k">Share of day</span><span class="v mono" id="sentinelGaugeVal">—</span></div>
    <div class="verdict" id="s1Decision">—</div>
  </div>
  <div class="panel stage">
    <div class="panel-head"><span class="t">Stage 3 — anomaly verdict</span><span class="r mono"><span id="s2Pvalue">p = —</span> · <span id="s2Alarm">—</span></span></div>
    <div class="pvalue-chart-wrap" id="pvalueChartWrap" style="height:120px;border:1px solid var(--hair-2);background:var(--panel-2);border-radius:2px;overflow:hidden"><!-- svg injected --></div>
    <div class="kv-row" style="margin-top:6px"><span class="k">Current score</span><span class="v mono" id="s2Score">—</span></div>
    <div class="kv-row"><span class="k">Alarm rule</span><span class="v mono">p &lt; α = 0.05, per state</span></div>
    <div class="kv-row"><span class="k">Per-state score threshold</span><span class="v mono" id="s2Threshold">—</span></div>
    <div class="kv-row"><span class="k">Near a mode transition</span><span class="v" id="s2NearTransition">—</span></div>
    <div class="verdict">Scored against <b>its own state's</b> normal model. A firing window becomes a register episode.</div>
  </div>
</div>

<div class="group-label"><span class="t">Alarm register</span><span class="ln"></span>
  <span class="cap">every episode with its trigger criterion — click LISTEN to jump there</span></div>
<div class="panel" style="margin: 0 18px; padding: 4px 8px 2px;">
  <table class="register-table">
    <thead><tr><th>Time (UTC)</th><th>Path</th><th>State</th><th>Why flagged</th><th></th></tr></thead>
    <tbody id="alarmList"><!-- injected --></tbody>
  </table>
</div>

</main>

<div class="transport">
  <button type="button" class="btn-play" id="playBtn">▶ Play</button>
  <button type="button" class="btn-speed active" data-speed="1">1×</button>
  <button type="button" class="btn-speed" data-speed="4">4×</button>
  <button type="button" class="btn-speed" data-speed="16">16×</button>
  <span class="sep"></span>
  <span class="listen-lb">LISTEN</span>
  <span id="listenMics">
    <button type="button" class="mic active" data-listen="muted">muted</button>
    <button type="button" class="mic" data-listen="gen">generator mic</button>
    <button type="button" class="mic" data-listen="tur">turbine mic</button>
  </span>
  <span class="vol">vol <input type="range" id="listenVolume" min="0" max="1" step="0.01" value="0.85">
    <span class="mono" id="volPct">85 %</span></span>
  <span class="tm mono" id="transportReadout">t = 00:00:00 / 00:00:00 h</span>
</div>

<footer class="site-footer">Krummenacher, 2026, University of St. Gallen — see README.md and
CITATION.cff for the full citation. Sensor recordings are proprietary plant data and are not
redistributed (DATA_ACCESS.md). This replay is a retrospective simulation over already-recorded
data, not a live sensor connection.</footer>

<audio id="audioGen" preload="none"></audio>
<audio id="audioTur" preload="none"></audio>

<script id="live-data" type="application/json">__LIVE_DATA_JSON__</script>
<script src="assets/live.js"></script>
</body>
</html>
```

- [ ] **Step 2: Verify token + zero-external contract**

```bash
grep -c "__LIVE_DATA_JSON__" docs/site/live_template.html   # expect 1
.venv/bin/python -c "
import sys; sys.path.insert(0, 'scripts')
import build_site as bs
from pathlib import Path
print(bs.find_external_resource_urls(Path('docs/site/live_template.html').read_text()))
"   # expect []
```

- [ ] **Step 3: Commit**

```bash
git add docs/site/live_template.html
git commit -m "feat(live): v7 template — recording cards, legend, 4-channel trend, transport listen"
```

(live.js still targets some old ids — the page is consistent again after Task 9; both land before any rebuild is published.)

---

### Task 9: live.js — v7 bindings

**Files:**
- Modify: `docs/site/assets/live.js`

**Interfaces:**
- Consumes: template ids from Task 8; payload keys from Tasks 6–7 (`features.audio_names`, `features.vib_names`, `scada.flow_1hz`, `scada.ks_1hz` — or the exact key names Task 6 chose, read them from the build script — `session`, `sessions_nav`).
- Produces: no exports (IIFE). Register LISTEN behavior: clicking a row's LISTEN seeks the playhead to the episode start and switches listen to `gen` if currently `muted`.

Work through the file top-down; every referenced anchor is a current line/function name:

- [ ] **Step 1: Delete removed-element code paths**

Remove or guard these blocks (element gone from the template): rings init (`// static: rings`, line ~121, and `setRingLevel` ~592, `setListenMarker`/`clearListenMarkers` ~651/657), feature dim labels (~387), `featureCanvas` bar renderer (`drawFeatureCanvas` ~415 and its callers at ~572/~839), BEATs gauge writes, `eraChip`/`sentinelChip`/`unitName` writes (~111 header/status), `s1ScadaState`/`s1LoadBin`/`s1Agree` readout writes, `feedLegend` fill, `transportSizeNote`. Rule: `$(id)` on a missing id must never be reached — delete the call sites, don't null-check around dead features.

- [ ] **Step 2: Session cards + header**

Add after the DOM-refs section:

```js
// -------------------------------------------------------------- static: session cards
(function renderSessionCards() {
  var wrap = $("sessionCards");
  DATA.sessions_nav.forEach(function (s) {
    var a = document.createElement("a");
    a.className = "session-card" + (s.id === DATA.session.id ? " active" : "");
    a.href = s.href;
    var hours = fmtDuration(s.duration_s); // existing helper: "04:19 h" style — reuse
    a.innerHTML =
      '<div class="dt">' + s.date_label + " · " + hours + "</div>" +
      '<div class="nm">' + s.display_name +
        (s.events ? '<span class="badge ev">EVENTS</span>' : "") + "</div>" +
      '<div class="in">' + s.blurb + " · " + s.n_episodes + " episodes</div>" +
      '<div class="mini-ribbon">' +
        s.ribbon.map(function (seg) {
          return '<i class="bg-' + cssState(seg.state) + '" style="width:' +
            ((seg.end_frac - seg.start_frac) * 100).toFixed(2) + '%"></i>';
        }).join("") +
        s.ticks.map(function (f) {
          return '<span class="mt" style="left:' + (f * 100).toFixed(2) + '%"></span>';
        }).join("") +
      "</div>";
    wrap.appendChild(a);
  });
  $("headRunLabel").textContent = DATA.session.id;
})();
```

`cssState(name)` maps state names to the utility suffixes (`turbine`, `pump`, `phase-shifter`, `standstill`, `transition`, `unknown`) — add it next to `stateColor` (line 22) using the same lookup keys.

- [ ] **Step 3: Ribbons — label both tracks, keep seek**

The detected-track segment builder (`addRibbonSeg`, line 146) and `updateRibbonLabels` (line 194) already exist; extend the SCADA track to use the same labeled-segment path as the detected track (today it may add unlabeled segs — unify so both tracks emit `.ribbon-seg` + `.ribbon-seg-label`). Bind the existing `seekFromRibbonEvent` (line 827) to BOTH `#ribbonWrapDetected` and `#ribbonWrapScada`. Move the playhead element handling: the template now has one `#ribbonPlayhead` (hidden) — instead create a playhead div inside each ribbon wrap at init and move both in `render()` (same left %).

- [ ] **Step 4: Agreement line**

Where both ribbons' second-resolution state arrays exist (the data behind the two tracks), compute once at init:

```js
var agreeCount = 0, total = 0;
for (var i = 0; i < detectedStates.length && i < scadaStates.length; i++) {
  if (scadaStates[i] === null || scadaStates[i] === "unknown") continue;
  total++;
  if (detectedStates[i] === scadaStates[i]) agreeCount++;
}
$("agreeLine").textContent = total
  ? "detector ↔ SCADA agreement: " + (100 * agreeCount / total).toFixed(1) + " % of windows"
  : "detector ↔ SCADA agreement: n/a";
```

Use the actual per-second state arrays the ribbon builders already consume (whatever their variable names are at line ~126–216).

- [ ] **Step 5: 4-channel trend + at-playhead values**

Replace the current SCADA-trend builder (`// static: SCADA trend`, lines ~269–328, `trendExtent`/`trendPolylines` reusable) with a loop over the four channels:

```js
var TRENDS = [
  { key: "power", el: "trendValP",  digits: 1, unit: "MW",   series: DATA.scada.power_1hz },
  { key: "speed", el: "trendValN",  digits: 1, unit: "rpm",  series: DATA.scada.speed_1hz },
  { key: "flow",  el: "trendValQ",  digits: 1, unit: "m³/s", series: DATA.scada.flow_1hz },
  { key: "ks",    el: "trendValKS", digits: 1, unit: "pos",  series: DATA.scada.ks_1hz },
];
```

(Adjust the four `DATA.scada.*` property names once against the real payload — Task 6 mirrored the existing power/speed key style.) For each: build one `<svg preserveAspectRatio="none">` polyline into the matching `.trend-chart[data-channel=…]` using `trendExtent`/`trendPolylines`. In `render(playheadS)`: set each value label via existing `fmtNum(v, digits)` + `' '` + unit-span (write `innerHTML = fmtNum(v,1) + ' <span class="u">MW</span>'` pattern), and position `#trendPlayhead` across the wrap:

```js
$("trendPlayhead").style.left =
  "calc(90px + (100% - 90px) * " + (playheadS / DATA.duration_s) + ")";
```

(Use the payload's real duration field name as already used by the transport readout.)

- [ ] **Step 6: Stage 1 — named top deviations + strip canvas**

Replace `drawFeatureCanvas` with:

```js
// -------------------------------------------------------------- stage 1: top features + strip
var fpCanvas = $("fpStripCanvas");
var fpCtx = fpCanvas.getContext("2d");

function topDeviations(snapIdx, k) {
  var out = [];
  var nA = DATA.features.n_audio, nV = DATA.features.n_vibration;
  for (var i = 0; i < nA; i++) out.push({ z: audioMat[snapIdx * nA + i], name: DATA.features.audio_names[i] });
  for (var j = 0; j < nV; j++) out.push({ z: vibMat[snapIdx * nV + j], name: DATA.features.vib_names[j] });
  out.sort(function (a, b) { return Math.abs(b.z) - Math.abs(a.z); });
  return out.slice(0, k);
}

function renderTopFeatures(snapIdx) {
  var rows = topDeviations(snapIdx, 4);
  $("topFeatures").innerHTML = rows.map(function (r) {
    var mag = Math.min(Math.abs(r.z) / 4, 1) * 50; // half-bar %
    var side = r.z >= 0 ? "pos" : "neg";
    var sign = r.z >= 0 ? "+" : "−";
    return '<div class="fp-row"><span class="nm">' + r.name + "</span>" +
      '<span class="fp-bar"><i class="' + side + '" style="width:' + mag.toFixed(1) + '%"></i></span>' +
      '<span class="fp-val mono">' + sign + Math.abs(r.z).toFixed(1) + " σ</span></div>";
  }).join("");
  drawFpStrip(snapIdx);
}

function drawFpStrip(snapIdx) {
  var rect = fpCanvas.getBoundingClientRect();
  var dpr = window.devicePixelRatio || 1;
  fpCanvas.width = Math.max(1, Math.round(rect.width * dpr));
  fpCanvas.height = Math.round(18 * dpr);
  fpCtx.setTransform(dpr, 0, 0, dpr, 0, 0);
  var nA = DATA.features.n_audio, nV = DATA.features.n_vibration, total = nA + nV;
  var w = rect.width, h = 18, barW = w / (total + 2);
  for (var i = 0; i < total; i++) {
    var z = i < nA ? audioMat[snapIdx * nA + i] : vibMat[snapIdx * nV + (i - nA)];
    var x = (i < nA ? i : i + 2) * barW;
    var bh = Math.min(Math.abs(z) / 4, 1) * (h - 3) + 1;
    fpCtx.fillStyle = Math.abs(z) >= 2 ? "#c07f10" : "#9db4d0";
    fpCtx.fillRect(x, h - bh, Math.max(barW - 0.4, 0.6), bh);
  }
  fpCtx.strokeStyle = "#b9c2cd";
  fpCtx.setLineDash([2, 2]);
  fpCtx.beginPath();
  var divX = (nA + 1) * barW;
  fpCtx.moveTo(divX, 0); fpCtx.lineTo(divX, h); fpCtx.stroke();
  fpCtx.setLineDash([]);
}
```

Call `renderTopFeatures(fi)` where `drawFeatureCanvas(fi)` was called (render loop ~572 and init ~839), but only when the snapshot index changed since the last call (keep a `lastSnapIdx` var) — sorting 231 values 60×/s is wasteful.

- [ ] **Step 7: KPIs, transport, listen, register**

- KPI writes keep unit spans: `powerKpi`/`speedKpi`/`farKpi` set `innerHTML = fmtNum(...) + ' <span class="u">MW</span>'` (respective units); `totalAlarmsKpi` → `String(nToday) + ' <span class="u">episodes</span>'`; `activeAlarmsKpi` → `nActive + " active now"`.
- `s1Decision` becomes a sentence: sentinel quiet → `"Sentinel quiet — once-calibrated thresholds stay frozen. Above its budget, the day would be flagged for recalibration."`; fired → `"Sentinel fired — this day exceeded its no-mode-fits budget; thresholds were recalibrated."` Toggle `.verdict.fired` and `.gauge.fired` classes from `DATA.sentinel.s1_fired` (existing flag, line ~115/770).
- Transport: `updatePlayButton` (line 806) uses `"▶ Play"`/`"❚❚ Pause"`; readout format `"t = " + fmtHMS(playheadS) + " / " + fmtHMS(duration) + " h"`; volume input listener also sets `$("volPct").textContent = Math.round(v * 100) + " %"`.
- Listen: rebind `setListenMode` (line 726) to the `#listenMics button[data-listen]` pills (`.active` class swap) and set `$("specListenLabel").textContent = {muted: "muted", gen: "generator mic", tur: "turbine mic"}[mode]`.
- Register rows: rebuild the alarm-list renderer (~329–386) to emit `<tr>` matching the template table: time cell (`mono`, HH:MM:SS), path badge (`<span class="badge sus|tra">`), state, why-line, and `<td><span class="listen" data-cid="...">LISTEN ▸</span></td>`; on click: seek to episode start (existing seek call used by the old row click), and if listen mode is `muted`, call `setListenMode("gen")`. Keep `acked`/`future` row classes and the ack toggle if present.

- [ ] **Step 8: Rebuild + browser verification**

```bash
.venv/bin/python scripts/build_live_replay.py --run 290626-tu
```

Then open `docs/site/live.html` via the local static server and verify in the browser (console clean; each: session cards render with 4 entries and the active card marked; both ribbons labeled; legend + agreement line; 4 trends with values updating during play; top-features rows with humanized names and σ; sentinel verdict sentence; register rows with LISTEN jump; transport pills + `%` volume; clock ticking).

- [ ] **Step 9: Commit**

```bash
git add docs/site/assets/live.js
git commit -m "feat(live): v7 bindings — sessions, labeled ribbons, 4-channel trend, named features, transport listen"
```

---

### Task 10: candidate_kit — v7 candidate fragment + SVG SCADA rows

The kit's interactive page (both the standalone `results/candidate-kit/index.html` and the site copy) gets the v7 card look, and `buildScadaBlock` switches from the pre-rendered `scada_png` image to data-driven SVG rows (the 1-Hz series are already in each card's meta: `power_mw_1hz`, `speed_rpm_1hz`, `flow_net_m3s_1hz`, `ks_valve_1hz` — see `build_extended_readout_series`, `scripts/candidate_kit.py:1551`). The kit page stays fully self-contained (inline CSS/JS — file:// contract in `render_index_html`'s docstring).

**Files:**
- Modify: `scripts/candidate_kit.py` — the `_CANDIDATE_CSS`-style constant feeding `_review_page_head`, the `_CANDIDATE_JS` constant (`buildScadaBlock`/`drawScadaOverlay`/`renderScadaBlock` region — the emitted JS is at `docs/site/review.html:594–696` today), `_CANDIDATE_LEGEND_HTML`, `render_index_html` (line 3592: page head/topbar/title)
- Test: `tests/test_candidate_kit.py` (append)

**Interfaces:**
- Consumes: meta keys `power_mw_1hz`, `speed_rpm_1hz`, `flow_net_m3s_1hz`, `ks_valve_1hz`, `asset_duration_s`, existing per-card fields (`candidate_id`, `class`, `state_name`, `scada_state`, `mode_mismatch`, `near_transition`, `in_sample`, criterion sentence).
- Produces: `render_index_html(...)` unchanged signature; emitted JS gains `buildScadaRows(card, meta)` replacing `buildScadaBlock(card, meta)`; `scada_png` is no longer referenced by the page (the PNG asset may still be produced for the offline static variant — do not delete its generation in this task).
- Assessment values, localStorage keys, export CSV/JSON format: **unchanged** (existing local assessments must survive).

- [ ] **Step 1: Write the failing test**

Append to `tests/test_candidate_kit.py` (reuse that file's existing fixture pattern for building a small `results` list — copy the minimal-meta construction from its existing `render_index_html` test if present; otherwise build one candidate meta dict inline with the fields above and 20 s of synthetic series):

```python
def test_render_index_html_v7_scada_rows(tmp_path) -> None:
    # ...build `results` exactly as the nearest existing render_index_html test does...
    out = ck.render_index_html(results, tmp_path)
    html = out.read_text()
    assert "buildScadaRows" in html
    assert "scada_png" not in html          # image path no longer referenced
    assert "m³/s" in html and "MW" in html  # units in the row labels
    assert "plausible anomaly" in html      # assessment vocabulary unchanged
    assert "EXPORT" in html.upper()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_candidate_kit.py::test_render_index_html_v7_scada_rows -v`
Expected: FAIL (`buildScadaRows` not in emitted HTML).

- [ ] **Step 3: Implement**

1. **CSS constant:** restyle the card CSS to v7 (values from Task 2's design.css; keep it inline/self-contained): white cards on `#e9ecf0`, `.badge` variants, spectrogram lanes unchanged mechanically, add the SCADA row styles (copy `.ar-scada`/`.ar-trow`/`.ar-tchart`/`.ar-ribbon` rules from `docs/superpowers/specs/mockups/audio-review-v7.html`, renaming to the kit's own class prefix), assessment pills restyled (radio inputs visually hidden, labels as pills with `.active`).
2. **JS:** replace `buildScadaBlock` + `drawScadaOverlay` with:

```js
function buildScadaRows(card, meta) {
  var scada = { card: card };
  var wrap = el("div", { class: "scada-block" });
  wrap.appendChild(el("div", { class: "scada-title",
    text: "Operating data (SCADA) — playhead follows the audio" }));

  var CHANNELS = [
    { key: "power_mw_1hz",     label: "P · ACTIVE POWER",    unit: "MW",   digits: 1, h: 26 },
    { key: "speed_rpm_1hz",    label: "n · SHAFT SPEED",     unit: "rpm",  digits: 0, h: 18 },
    { key: "flow_net_m3s_1hz", label: "Q · NET FLOW",        unit: "m³/s", digits: 1, h: 18 },
    { key: "ks_valve_1hz",     label: "KS · SPH. VALVE",     unit: "pos",  digits: 1, h: 18 },
  ];
  var body = el("div", { class: "scada-rows" });
  scada.valueEls = {};
  CHANNELS.forEach(function (ch) {
    var series = meta[ch.key] || [];
    var row = el("div", { class: "scada-trow" });
    var lab = el("div", { class: "scada-tlab" });
    lab.appendChild(el("span", { class: "n", text: ch.label }));
    var lv = el("span", { class: "lv mono" });
    lab.appendChild(lv);
    scada.valueEls[ch.key] = { el: lv, unit: ch.unit, digits: ch.digits, series: series };
    var chart = el("div", { class: "scada-tchart" });
    chart.style.height = ch.h + "px";
    chart.innerHTML = scadaRowSvg(series, ch.h);
    row.appendChild(lab); row.appendChild(chart);
    body.appendChild(row);
  });
  var ph = el("div", { class: "scada-rows-playhead" });
  body.appendChild(ph);
  scada.playheadEl = ph;
  wrap.appendChild(body);
  scada.el = wrap;
  return scada;
}

function scadaRowSvg(series, h) {
  var vals = series.filter(function (v) { return v !== null && isFinite(v); });
  if (!vals.length) return '<svg viewBox="0 0 100 ' + h + '"></svg>';
  var min = Math.min.apply(null, vals), max = Math.max.apply(null, vals);
  var span = (max - min) || 1;
  var pts = [];
  for (var i = 0; i < series.length; i++) {
    var v = series[i];
    if (v === null || !isFinite(v)) continue;
    var x = (i / Math.max(series.length - 1, 1)) * 100;
    var y = h - 2 - ((v - min) / span) * (h - 4);
    pts.push(x.toFixed(2) + "," + y.toFixed(2));
  }
  return '<svg viewBox="0 0 100 ' + h + '" preserveAspectRatio="none">' +
    '<polyline points="' + pts.join(" ") + '" fill="none" stroke="#2563a8" stroke-width="1.2"/></svg>';
}
```

`renderScadaBlock(card)` keeps its role but updates: playhead `left` = label-column-width + fraction of chart width (CSS `calc` like the mockup), and each `valueEls` label to `value.toFixed(digits) + " " + unit` at the current second (reuse the existing `secIdx` logic from `updateScadaReadout`, which this replaces). Keep the mode/ribbon context: reuse the existing per-card state-ribbon builder if there is one, else add a two-row labeled ribbon above the rows from `meta.scada_state`/`meta.state_name` (single full-width segments labeled with the state names — the clip is short).
3. **Page chrome:** `_review_page_head` gains nothing external; the site copy's topbar is handled by Task 11's composer, so `render_index_html` should keep emitting its own minimal standalone header for the kit build (title + date note restyled to v7 tokens).
4. Update the two `JSON.parse` bootstrap ids and everything else **only if** renames are needed — prefer zero renames.

- [ ] **Step 4: Run tests**

Run: `.venv/bin/python -m pytest tests/test_candidate_kit.py -v`
Expected: new test PASS, all existing kit tests PASS (they pin behavior, not look; fix any that assert old CSS/JS strings).

- [ ] **Step 5: Rebuild the kit and eyeball one card**

```bash
.venv/bin/python scripts/candidate_kit.py build
```

Open `results/candidate-kit/index.html` in the browser: SCADA rows render full-width responsive with unit-labeled live values; marks/assessment/export still work (set an assessment, reload, still there).

- [ ] **Step 6: Commit**

```bash
git add scripts/candidate_kit.py tests/test_candidate_kit.py
git commit -m "feat(review): v7 candidate cards with data-driven SCADA rows (drop scada_png embed)"
```

---

### Task 11: publish_audio_review.py — the merged Audio & review page

Rename `publish_review_site.py` → `publish_audio_review.py`; it becomes the composer that emits `docs/site/audio_review.html`: v7 app-bar shell + toolbar (segmented tabs, session pills, progress, export) + candidates tab (Task 10 fragment) + strikes/per-mode tabs (clip-card fragments from `build_site`'s manifest data).

**Files:**
- Rename+modify: `scripts/publish_review_site.py` → `scripts/publish_audio_review.py`
- Modify: `scripts/candidate_kit.py` — factor the body of `render_index_html` so the composer can request a **fragment** (`render_candidates_fragment(results, asset_prefix) -> tuple[str, str, str]` returning `(css, body_html, js)`) while `render_index_html` keeps composing the standalone page from the same fragment.
- Modify: `scripts/build_site.py` — expose `render_clip_cards(manifest: dict, kind: str, asset_prefix: str = "") -> str` for `kind in {"strikes", "modes"}` reusing the existing card fragment functions (`_strike_clip_card` line 904, `_demo_clip_card` line 923) restyled to `.clip-card` markup.
- Test: `tests/test_build_site.py` (append)

**Interfaces:**
- Consumes: `ck.render_candidates_fragment`, `bs.render_clip_cards`, `sc.app_bar_html`, `sc.group_label_html`, `docs/site/assets/site_manifest.json` (existing, built by `build_site.build_site_manifest` line 496).
- Produces: `docs/site/audio_review.html`; stops producing `docs/site/review.html`/`review_static.html` (Task 4's stubs own those paths). Tab switching + session filtering + progress counter are small inline JS in the composer (show/hide `data-tab` sections, count assessed candidates from localStorage using the kit's existing storage-key prefix — read the prefix constant from the kit fragment via a token, not a copy).

- [ ] **Step 1: Write the failing test**

Append to `tests/test_build_site.py`:

```python
def test_audio_review_compose(tmp_path, monkeypatch) -> None:
    import publish_audio_review as par
    html = par.compose_audio_review_html(
        candidates_fragment=("/*css*/", "<div id='cand'></div>", "/*js*/"),
        strikes_html="<div class='clip-card'>strike</div>",
        modes_html="<div class='clip-card'>mode</div>",
        n_candidates=12, n_strikes=18, n_modes=8,
        sessions=["290626-tu", "080726-pu_strikes"],
    )
    assert 'class="app-bar"' in html
    assert "Flagged candidates" in html and "Hammer strikes" in html and "Per-mode audio" in html
    assert "EXPORT" in html.upper()
    assert "assessments stay in your browser until exported" in html
    import build_site as bs
    assert bs.find_external_resource_urls(html) == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_build_site.py::test_audio_review_compose -v`
Expected: FAIL (`ModuleNotFoundError: publish_audio_review`).

- [ ] **Step 3: Implement**

1. `git mv scripts/publish_review_site.py scripts/publish_audio_review.py`.
2. In `candidate_kit.py`, split `render_index_html` mechanically: everything between "assemble css", "assemble body", "assemble js" becomes `render_candidates_fragment(results, asset_prefix="") -> tuple[str, str, str]`; `render_index_html` = standalone head + the fragment (behavior identical — the Task 10 test must still pass unchanged).
3. In `build_site.py`, add `render_clip_cards(manifest, kind, asset_prefix="")` mapping over `manifest["strike_clips"]` / `manifest["demo_clips"]` (open `build_site_manifest` at line 496 for the exact key names and use those) through the existing card fragment functions, updated to emit `.clip-card` markup: title + state `.badge`, `<audio controls preload="none">`, `.meta` line with strike count and `… UTC` timestamp.
4. In `publish_audio_review.py`, add:

```python
def compose_audio_review_html(*, candidates_fragment, strikes_html, modes_html,
                              n_candidates, n_strikes, n_modes, sessions):
    css, cand_body, cand_js = candidates_fragment
    session_pills = "".join(
        f'<button type="button" class="pill" data-session="{s}">{s}</button>' for s in sessions
    )
    tabs_js = _TABS_JS  # module constant, below
    return (
        "<!doctype html>\n<html lang=\"en\">\n<head>\n<meta charset=\"utf-8\">\n"
        '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
        "<title>Audio &amp; review — ROWII Monitor</title>\n"
        '<link rel="stylesheet" href="assets/design.css">\n'
        f"<style>{css}</style>\n</head>\n<body>\n"
        + sc.app_bar_html("audio_review.html")
        + '<main class="page">'
        + sc.group_label_html("Audio library &amp; expert review",
                              "assessments stay in your browser until exported")
        + '<div class="toolbar">'
        + '<div class="seg-tabs">'
        + f'<button type="button" class="active" data-tab="candidates">Flagged candidates<span class="n">{n_candidates}</span></button>'
        + f'<button type="button" data-tab="strikes">Hammer strikes<span class="n">{n_strikes}</span></button>'
        + f'<button type="button" data-tab="modes">Per-mode audio<span class="n">{n_modes}</span></button>'
        + "</div>"
        + f'<div class="pills"><button type="button" class="pill active" data-session="all">all sessions</button>{session_pills}</div>'
        + '<div class="right"><span class="progress-note" id="reviewProgress">—</span>'
        + '<button type="button" class="btn-export" id="exportBtn">EXPORT (.json)</button></div>'
        + "</div>"
        + f'<section data-tab-panel="candidates">{cand_body}</section>'
        + f'<section data-tab-panel="strikes" hidden><div class="clip-grid">{strikes_html}</div></section>'
        + f'<section data-tab-panel="modes" hidden><div class="clip-grid">{modes_html}</div></section>'
        + "</main>"
        + sc.FOOTER_HTML
        + f"<script>{cand_js}</script>\n<script>{tabs_js}</script>\n</body>\n</html>\n"
    )
```

`_TABS_JS` (module constant, plain string): tab clicks toggle `.active` + matching `[data-tab-panel]` `hidden`; session pills set a `data-session-filter` attribute on the candidates section and hide non-matching cards (cards already carry their session in meta — give each card root a `data-session` attribute in the Task 10 fragment if it lacks one); progress = count of candidate ids with a non-empty stored assessment over total (storage prefix token `__STORAGE_PREFIX__` replaced at compose time with the kit's constant); `#exportBtn` clicks the kit fragment's existing export control (`document.querySelector` on its existing id) — one source of truth, no re-implementation.

5. `main()`: keep steps 1–2 of the old script (kit build + asset copy to `docs/site/assets/review/`), then render the fragment with `asset_prefix="assets/review/"`, load `docs/site/assets/site_manifest.json`, render strikes/modes card HTML, compose, write `docs/site/audio_review.html`. Delete the old `review.html`/`review_static.html` writes.

- [ ] **Step 4: Run tests**

Run: `.venv/bin/python -m pytest tests/test_build_site.py tests/test_candidate_kit.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/publish_audio_review.py scripts/candidate_kit.py scripts/build_site.py tests/test_build_site.py
git rm --cached scripts/publish_review_site.py 2>/dev/null || true
git commit -m "feat(site): merged audio_review page — tabs, progress, export, composed fragments"
```

---

### Task 12: Full rebuild, verification sweep, cleanup

**Files:**
- Modify: `docs/site/*` (regenerated), delete stale `docs/site/snippets.html` content (now a stub), `docs/site/review_static.html`
- Modify: `README.md` — the site-build section (if present; `grep -n "build_site\|publish_review" README.md`) now lists the three build commands below.

**Interfaces:**
- Consumes: everything above.

- [ ] **Step 1: Full test suite**

Run: `.venv/bin/python -m pytest tests/ -q`
Expected: all green. Fix regressions before proceeding.

- [ ] **Step 2: Preflight per-session artifacts, then build everything**

```bash
ls results/cache/ | grep -E "080726-pu_strikes|010726-tu1-morning|270626" || echo "MISSING CACHES"
.venv/bin/python scripts/build_live_audio.py --run 080726-pu_strikes
.venv/bin/python scripts/build_live_audio.py --run 010726-tu1-morning
.venv/bin/python scripts/build_live_audio.py --run 270626-pu_ph_pu_ph_pu_ph-1
.venv/bin/python scripts/build_live_replay.py
.venv/bin/python scripts/build_site.py build-pages
.venv/bin/python scripts/publish_audio_review.py
rm -f docs/site/review_static.html
```

If any session's inputs are missing, report exactly which (with the preflight's named producing script), build the sessions that ARE available, and flag the gap to Stefan — do not fake a session card.

- [ ] **Step 3: Zero-external + stub checks**

```bash
.venv/bin/python - <<'EOF'
import sys; sys.path.insert(0, "scripts")
from pathlib import Path
import build_site as bs
for p in sorted(Path("docs/site").glob("*.html")):
    urls = bs.find_external_resource_urls(p.read_text())
    assert not urls, (p, urls)
    print("ok", p.name)
EOF
grep -l "http-equiv=\"refresh\"" docs/site/snippets.html docs/site/review.html
```

Expected: every page `ok`; both stubs found.

- [ ] **Step 4: Browser verification sweep**

Serve `docs/site/` locally and walk every page against the committed mockups (`docs/superpowers/specs/mockups/`): index, sensors, live (all four session pages — switch via the recording cards, play 30 s at 16×, listen toggle, register LISTEN jump), audio_review (all three tabs, set an assessment + a mark, reload, export). Console must be error-free on every page. Screenshot live.html and audio_review.html for the report to Stefan.

- [ ] **Step 5: Commit the regenerated site**

```bash
git add -A docs/site README.md
git commit -m "build(site): regenerate all pages in v7 (4 live sessions, audio_review)"
```

---

## Self-Review (done while writing)

- **Spec coverage:** D1 (index lede + dense pages) T4/T8; D2 (merge + nav + stubs) T3/T4/T11; D3 (4 sessions) T6/T7/T12; D4 (v7 system) T2; D5 (group labels) T2/T3/T8/T11; D6 (functional color) T2 tokens + T9 badges; D7 (units) global constraint + T8 markup + T9 writes + T10 rows; D8 (English) all templates; D9 (named features) T5/T6/T9; D10 (transport listen, rings removed) T8/T9. Spec §4 removals (rings, BEATs gauge) T8/T9; §5 (SCADA rows, form unchanged, export) T10/T11; §7 honesty/reproducibility T7 preflight + T12.
- **Placeholder scan:** no TBDs; the two "mirror the existing key names" instructions in T6/T9 are deliberate read-the-code anchors (the exact names exist in the file being edited), not omissions.
- **Type consistency:** `app_bar_html`/`group_label_html` used identically in T4/T8/T11; `LIVE_SESSIONS[...]["out"]` ↔ `pages_nav` `href` ↔ T9 session cards; fragment tuple `(css, body, js)` consistent between T10 factor-out and T11 composer; `sessions_nav` schema matches between T7 producer and T9 consumer.
