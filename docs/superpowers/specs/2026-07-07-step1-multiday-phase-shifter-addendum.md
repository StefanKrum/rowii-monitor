# Step 1 Addendum — Multi-Day Ingestion + Phase-Shifter State

**Date:** 2026-07-07 · **Extends:** `2026-07-05-step1-state-detection-design.md`
**Trigger:** Bruno's re-release of the full Illwerke campaign (Nextcloud, mail 2026-07-06):
four measurement days, including phase-shifter operation. Stefan's rules: Bruno's repo and
results are READ-ONLY cross-check references; only the raw data is used by this project.

## 1. New data reality (verified from the zips' tables of contents + selective extraction)

| Day root (under `data/`) | Sessions (folder names verbatim) | SCADA | Notes |
|---|---|---|---|
| `illwerke-250526/20260626 Messung` | TU, PU | hours ..11:00 only | existing; PU gained 2 rescued files (09:20 mic, 09:32 vib) from the re-share |
| `illwerke-270626/20260627 Messung` | `PU_PH_PU_PH_PU_PH` | **none** (photo `ROWII_LeistungPU_PH.jpg` only) | ~4 h alternating pump/phase-shifter |
| `illwerke-290626/20260629 Messung` | TU, PU | full day | TU includes a ~37-min PH hold during converter-assisted start |
| `illwerke-010726/20260701 Messung` | PU, TU1, TU2, `TU_PH_TU` | full day | first day with ALL FOUR modes; Campaign-1 headline dataset |

Storage: day-by-day rotation (extract → verify file counts vs zip TOC → delete zip);
Bruno's Nextcloud shares remain the archival source. Raw trees live under
`~/AI Workspace/master-thesis/data/illwerke-<day>/`.

## 2. Discovery generalization (`rowii.io.dataset`)

- `discover(data_root)` accepts BOTH a single `<...>/YYYYMMDD Messung`-style tree (current
  behaviour, backward compatible) and a PARENT root containing multiple day trees
  (`data/illwerke-*/20260*  Messung/`).
- A **session** is any direct subfolder of a `* Messung` dir that contains burst-pattern
  files (replaces the hardcoded `("TU", "PU")` folder filter). Betriebsdaten folders are
  recognised by name, as today.
- Run naming: `<dayid>-<sanitized-session>[-<gapsplit>]`, day id = the 6-digit token from
  the day root (e.g. `010726-tu_ph_tu`, `270626-pu_ph_pu_ph_pu_ph`, `250526-pu-morning`
  keeps its legacy split suffix). Sanitize: lowercase, non-alnum → `_`. Gap-splitting
  within a session unchanged (>15 min).
- Session folder names are treated as OPERATOR HINTS only (like filename times): they
  never feed the detector; GT still comes exclusively from SCADA.

## 3. Phase-shifter ground truth (`rowii.scada.labels` + config)

- `STATES` += `"phase-shifter"`.
- Rule (own design; thresholds in `GtRules`): windows at nominal speed
  (|n| ≥ 0.95·n_nom) with |P| ≤ power_eps_mw that form a CONTIGUOUS run of at least
  `ph_min_dwell_s` (default **600 s**) become `phase-shifter`; shorter such runs remain
  `transition` (unloaded spinning during start/stop sequences). Rule ordering: base states
  → ph-dwell promotion → ramp rule → transition buffer (buffer never overwrites
  phase-shifter interiors, only its edges).
- **Verification channel `1_KS Stellung` (spherical inlet valve):** hypothesis from
  Bruno's channel audit (≈3 = closed/dewatered during PH and standstill; ≈104 = open) —
  MUST be independently verified on our own 2026-07-01 data (extend
  `scripts/verify_parameters.py`: report KS value distribution per GT state) BEFORE it is
  used. If confirmed, add as an optional conjunctive check (`ph_requires_ks_closed:
  bool = True` once verified) and record measured open/closed values in
  `results/parameter_verification.md` with provenance note (hypothesis: Bruno's SCADA
  audit; verification: own data).
- `GT_CHANNELS` += `{"reactive": "1_Q_Ist", "ks_valve": "1_KS Stellung"}` (loaded for
  verification/reporting; the PH rule itself uses speed+power+dwell±KS only).
- 27.06 (no SCADA): processed with `gt=unknown` end-to-end (existing path); the photo
  serves as qualitative cross-check in the README only.

## 4. Runs & evaluation

- Campaign-1 headline: `010726-tu_ph_tu` (4-state day) + `010726-*` siblings;
  `290626-tu` (PH hold inside a TU session) as the hard boundary case.
- Metrics unchanged: state-level majority mapping handles the extra state automatically;
  `n_states`/k defaults revisited via `--k-sweep` on the 01.07 data (no assumption).
- Cross-day questions (train day A, test day B) are Step-2/Campaign territory — out of
  scope for this addendum (Bruno's cross-day numbers stay HIS results).

## 5. Non-goals

- No use of Bruno's code, labels, features, or results in our pipeline.
- No full-quality audio archive management beyond the rotation described in §1.
- No supervised classifier; the interim milestone stays unsupervised (thesis TODO T5.3).
