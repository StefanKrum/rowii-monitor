# Demo-Site Redesign — Design Spec

**Date:** 2026-08-17
**Status:** validated with Stefan via seven mockup iterations (visual companion, `.superpowers/brainstorm/62897-1786922466/content/`, final references: `live-mockup-v7-light.html`, `audio-review-mockup.html`)
**Scope:** `docs/site/` and the generators that produce it (`scripts/build_site.py`, `scripts/candidate_kit.py`, `scripts/build_live_replay.py`, `scripts/site_common.py`, `docs/site/assets/live.js`, `docs/site/assets/design.css`, `docs/site/live_template.html`)

## 1. Problem

Stefan's review of the current site (2026-08-17):

1. Review page's SCADA panel renders a fixed-width pre-rendered PNG (`candidate_kit.py` → `scada_png`, displayed at natural pixel size) — on wide screens it fills a third of the card and looks broken.
2. "Listening Library" and "Candidate Review" look identical (cards with spectrogram + audio) although they serve different purposes; visitors cannot tell them apart.
3. Pipeline diagnostics are cryptic: the feature canvas ("audio (135) / vibration (96)") shows anonymous z-score bars nobody can read; BEATs gauge and sentinel numbers lack plain-language framing.
4. Live replay is hard-wired to one session (`290626-tu`); no way to open the hammer-strike day or other recordings.
5. Session-timeline semantics (red ticks, yellow segments, two ribbon rows) are only explained by one cryptic caption line; SCADA row has no in-segment labels.
6. Overall look is generic "AI design" (white cards, rounded corners, chip salad, eyebrow labels). Wanted: a real monitoring-software look.

## 2. Decisions (validated with Stefan)

| # | Decision | Choice |
|---|----------|--------|
| D1 | Audience | All three equally: supervisors/defense, plant-expert handover, portfolio. Explanatory entry + dense expert views, clearly separated. |
| D2 | IA | Merge Listening Library + Candidate Review into **one page `audio_review.html`** ("Audio & review") with tabs. Nav: Overview · Sensors · Live replay · Audio & review. |
| D3 | Live sessions | **All four** recordings switchable: `290626-tu`, `080726-pu_strikes`, `010726-tu1-morning`, `270626-pu_ph_pu_ph_pu_ph-1`. |
| D4 | Design direction | Iterated A→C→v6→**v7: light monitoring software**. Not the warm paper/report hybrid, not full dark — a light technical system UI. Spectrograms/measurement graphics stay dark (magma) inside light panels. |
| D5 | Headings | No essay/eyebrow headings, no section numbers. Sober uppercase **group labels** (small, gray, letterspaced) with a hairline extending right and an optional one-line caption right-aligned. Panel-internal labels stay uppercase micro-labels. |
| D6 | Color semantics | Red **exclusively** for alarm semantics (ticks, active counts, TRANSIENT). Amber = transition/caution/EVENTS. Green = ok/live/agreement. State colors keep the existing closed vocabulary. No decorative accent anywhere (reaffirms the original design contract). |
| D7 | Units | **Every number carries its unit or label**: `3 episodes`, `7.5 % of windows`, `245.9 MW`, `378.8 rpm`, `+1.8 σ`, `since 00:41:12 UTC`, `04:19 h`, `41.3 m³/s`, `2.8 pos`, `64 mel bands · 0–8 kHz`, volume `65 %`, table column `TIME (UTC)`. |
| D8 | Language | All site content in **English**. |
| D9 | Stage 1 features | Show **named** top-deviating features (e.g. "Generator mic 3 · octave band 2 kHz · +1.8 σ"), full 231-value strip as context below. Feature names exist (`AudioFeaturizer.feature_names()` / `VibFeaturizer`, `src/rowii/signals/features.py`); a human-readable mapping layer is added at build time. |
| D10 | Audio control | Listen source (muted / generator mic / turbine mic) + volume move into the **sticky transport bar**; the spectrogram panel header shows which mic is being listened to. The sensor-rings panel is removed from the live page (lives on the Sensors page); its "listening" pulse marker feature is dropped. |

## 3. Design system (v7)

Replaces the current `design.css` look. One shared stylesheet, all pages.

**Surfaces (light monitoring):**

- Page background `#e9ecf0` (cool technical gray — not warm paper)
- Panel `#ffffff`, border `#d3d9e0`, radius 4 px, shadow `0 1px 2px rgba(24,32,42,.04)`
- Panel-inner chart background `#f7f9fb` / `#fbfcfd`, chart border `#dfe4ea`
- Ink `#18202a`, dim `#5c6b7d`, faint `#7a8798` / `#97a2b0`

**Functional colors:** live/ok `#0e8f6f` · alarm `#c73a1d` · warn/transition/EVENTS `#a16207` (text) / `#c07f10` (fills). State vocabulary unchanged: turbine `#2563a8`, pump `#7c4dbc`, phase-shifter `#1d8a70`, standstill `#6b7684`, transition `#c07f10`, unknown `#aab2bc`.

**Type:** UI = Helvetica Neue stack; ALL numerals mono (`ui-monospace` stack, tabular). Group labels 8.5 px / 800 / letterspacing .16em / `#7a8798`.

**App bar (every page):** white, bottom border; left brand `ROWII MONITOR` (bold + light), tab nav (active = `#eef1f5` fill + inset 1 px ring), right side page-specific status (live: `● REPLAY` green + date + `HH:MM:SS UTC` clock).

**Group label row:** `LABEL ———————————— caption`. No other heading kinds on any page.

**Numbers rule (D7):** enforced everywhere, including KPI sub-lines, captions, table headers.

## 4. Live replay page

Layout top→bottom (reference: `live-mockup-v7-light.html`):

1. **App bar** with REPLAY badge, session date, running UTC clock.
2. **RECORDING** group: 4 session cards — weekday + date + real duration (`… h`), display name (Turbine day / Hammer-strike day / Morning session / Cycling day), one-line description, real per-session mini mode-timeline (6 px ribbon from step-1 output) with real alarm ticks; amber `EVENTS` badge on the hammer day. Active card: ink border + left ink spine. Clicking a card opens that session's replay.
3. **KPI band** (5 white panels): Operating state (dot + name, `since HH:MM:SS UTC · threshold certified` / low-confidence note), Active power `… MW` (SCADA, 1 Hz mean), Shaft speed `… rpm` (turbine +, pump −), Alarm episodes today `N episodes` + red `n active now`, False-alarm rate `… % of windows` (realized · budget α = 5 %).
4. **SESSION TIMELINE** panel: detector + SCADA ribbons, **both** with in-segment state labels (narrow segments: hover tooltip as today); full **legend** (5 state swatches + transition + red tick = "Alarm episode → register"); green `agreement today: … % of windows`; **4 stacked trend rows P / n / Q / KS**, each row label = channel name + **value at the playhead** (updates while playing), shared playhead through ribbons and trends; click/drag anywhere to seek.
5. **SPECTROGRAM** panel: log-mel strip ±30 s around playhead (dark magma), axis labels (`8 kHz`/`0 kHz`, `−30 s`/`+30 s`), white playhead; header right: `listening: <mic> · 64 mel bands · 0–8 kHz`.
6. **PIPELINE DIAGNOSTICS** (3 panels, same second as playhead):
   - **Stage 1 — Input features:** top 4 deviating features **by human-readable name** with diverging ±σ bars (amber above day-mean, blue below), full 231-bar strip below (audio | vibration split, deviating bars amber), caption with the σ-vs-day-mean definition.
   - **Stage 2 — State & drift sentinel:** detected state + since, mode bank cluster + certification, `no-mode-fits … % / budget … %` + gauge, plain-language verdict ("Sentinel quiet — once-calibrated thresholds stay frozen…" / fired wording from real decision).
   - **Stage 3 — Anomaly verdict:** p-stream chart (log scale, dashed amber α = 0.05 line, red alarm dots, ink playhead), current `p = …`, alarm rule `p < α = 0.05, per state`, near-transition flag, verdict; caption noting per-state normal models and that firing windows become register episodes.
7. **ALARM REGISTER** panel: table `TIME (UTC) · PATH · STATE · WHY FLAGGED · LISTEN ▸`; SUSTAINED badge amber, TRANSIENT red; LISTEN seeks the replay there and enables audio. Rows for the session's real candidate episodes (existing why-lines from `candidates.csv`).
8. **Transport bar** (sticky bottom, white): `▶ PLAY`, speeds 1×/4×/16×, separator, `LISTEN` mic pills (muted / generator mic / turbine mic), volume slider + `… %`, right `t = HH:MM:SS / HH:MM:SS h`.

**Removed from live page:** sensor rings panel, BEATs L2-norm gauge (BEATs stays mentioned in Overview/thesis; the live gauge communicated nothing), the old status-bar/era chips (era + sentinel info live in Stage 2).

**Sessions (D3):** one built page per session sharing template + `live.js` (e.g. `live.html` = default `290626-tu`, plus `live-080726-pu_strikes.html`, …); the RECORDING cards cross-link. Each page keeps the existing single-JSON-payload architecture (~8 MB each; audio lazy as today). Per-session regime honesty (frozen vs. recalibrated) as computed by the existing pipeline — no counterfactuals; mockup KPI/duration numbers are placeholders to be replaced by real artifact values at build time.

## 5. Audio & review page (`audio_review.html`)

Replaces `snippets.html` + `review.html` (reference: `audio-review-mockup.html`).

1. **Toolbar:** segmented tabs with counts — `Flagged candidates N · Hammer strikes N · Per-mode audio N`; session filter pills (all sessions / per session); right: progress `n / N assessed · autosaved` + `EXPORT (.json)` (existing localStorage state + existing export path).
2. **Candidates tab:** list of collapsed rows (id, timestamp UTC, path badge, state badge, near-transition / in-sample badges, `assessed: … ✓` when done, OPEN). Expanded card:
   - Header: `candidate <id>`, `YYYY-MM-DD · HH:MM:SS UTC · 20 s clip`, path badge, `STATE: <detector state>`, `SCADA ↔ DETECTOR AGREE` (green) or `⚠ SCADA/DETECTOR DISAGREE` (amber), near-transition / in-sample badges, position `k of N`.
   - `WHY FLAGGED` line (existing criterion sentence).
   - Two spectrogram lanes (generator / turbine mic), each with its own player; shift-click sets yellow mark lines; playhead per lane.
   - **OPERATING DATA (SCADA)** block — same pattern as the live page: labeled detector + SCADA ribbons for the clip context window, then P / n / Q / KS rows with at-playhead values, ink playhead synced to the active lane's audio, caption `curves span ±60 s around the clip · yellow segment = SCADA transition window`. Data source: the per-second series already embedded in card meta (`power_mw_1hz`, `speed_rpm_1hz`, `flow_net_m3s_1hz`, `ks_valve_1hz`) — **the pre-rendered `scada_png` is no longer used on the site**.
   - **EXPERT ASSESSMENT:** marks chips (removable), five pills — `plausible anomaly · operational / explained · artifact / sensor · no finding · unclear` — note field, status `saved locally · included in export`. Persistence identical to today (localStorage keyed by stable candidate id).
3. **Hammer strikes / Per-mode audio tabs:** simple cards — title, state badge, mini spectrogram strip, player, meta line (strike counts, mic, timestamp UTC). No form, no SCADA block. Content = current snippets sources.
4. The standalone offline kit under `results/candidate-kit/` (and `review_static.html`'s purpose) is **not** part of the site redesign; the site drops `review_static.html` from nav/output. The standalone kit remains available for offline handover until separately migrated.

## 6. Overview & Sensors pages

Same app bar + design system, restyled content, English:

- **Overview:** short system description (what ROWII Monitor is, pipeline in three sentences), the research-prototype/data-approval notice (kept verbatim in meaning), three link cards (Sensors / Live replay / Audio & review) in v7 card style, footer citation line.
- **Sensors:** existing section/plan-view SVGs inside white panels with group labels; stream/channel names on hover as today.

## 7. Implementation notes (for the plan)

- **`design.css`:** rewrite to v7 tokens + shared components (app bar, group label, panel, badges, seg control, pills, tables, transport). Keep the closed state-color vocabulary. Per-page `<style>` blocks shrink accordingly.
- **`site_common.py`:** shared app-bar/nav + group-label HTML helpers so all generators emit identical chrome.
- **`build_live_replay.py`:** parameterize by session (×4); add to the payload: per-session card metadata (name, duration, description, mini-ribbon segments, episode count), feature-name table + human-readable labels (mapping `ch<N>`/vib prefixes → mic/accel display names via the existing stream/channel registry), per-second SCADA Q + KS series (P/n already present). Top-deviating features are computed client-side from the already-shipped z-matrices + the new name table.
- **`live.js`:** new bindings — session cards, trend at-playhead values, both-ribbon labels, legend, stage-1 top-features render, transport listen pills + volume %, register LISTEN seek. Existing seek/audio/HMR-free architecture unchanged.
- **`candidate_kit.py` (site output):** emit `audio_review.html`; candidate card data comes from the existing candidate-kit meta, strike/per-mode clip data from the current snippets sources (exact generator ownership split is settled in the implementation plan). Replace `scada_png` usage with data-driven SVG rows (series already in meta); keep card ids/state persistence stable so existing local assessments survive.
- **`build_site.py`:** index/sensors restyle; nav updated everywhere; stop emitting `snippets.html`/`review.html` and instead emit tiny redirect stubs at those paths pointing to `audio_review.html`, so existing bookmarks keep working.
- **Honesty rule:** every number on the site comes from real artifacts (results/, candidates.csv, step-1/2 outputs). No invented values; mockup numbers were placeholders.
- **Reproducibility:** one documented build entry point regenerates all pages; no hand-edited generated HTML.

## 8. Out of scope

- New models, new data computation, changes to `results/` pipelines (except payload additions listed above).
- GitHub Pages / public publishing (still gated on Illwerke data-release approval).
- German localization; mobile-first work (pages stay usable down to ~tablet width as today).
- The standalone offline candidate kit under `results/candidate-kit/`.

## 9. Success criteria

1. Review SCADA context renders as responsive curves matching the live-page pattern — no fixed-width PNG (complaint 1).
2. One "Audio & review" page; no duplicate-feeling pages (complaint 2).
3. Stage 1 shows named features; sentinel and verdict panels carry one-sentence plain-language verdicts (complaint 3).
4. Four recordings switchable from the live page, incl. the hammer-strike day (complaint 4).
5. Timeline has a full legend, labeled SCADA segments, and an agreement line (complaint 5).
6. Site-wide v7 monitoring look: app bar, group labels, functional-only color, units on every number, English (complaint 6, D5–D8).
