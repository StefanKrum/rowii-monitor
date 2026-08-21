# Verification track: 08.07.2026 strike campaign

**Purpose.** Independent verification of the per-strike ground truth used for the 08.07.2026
controlled-event campaign. The canonical, evaluation-facing ground truth is unchanged and stays
canonical: `../080726_strikes_seconds_st.csv` and `../080726_strikes_seconds_pu.csv` (interactive
per-strike annotation by ear). Everything in this directory is a *second, independent read* of the
same audio — a reproduction of the measurement partner's impulse-detection method, run on our own
raw files — used to cross-check the annotator marks, not to replace them.

## Files

| File | Content |
|---|---|
| `strikes_register_st.csv` / `_pu.csv` | Canonical 90-slot-per-session protocol register (12 fixed positions x3 + 18 guide vanes x3). One row per intended strike; `source` tags how (or whether) it was resolved. |
| `raw_st.csv` / `_pu.csv` | Every detected impulse (k>30), uncapped. |
| `protocol_st.csv` / `_pu.csv` | Same detector, capped to <=3 strikes per position (strongest-triplet selection) and <=3 per vane cluster — the regenerated equivalent of the partner's never-delivered `strikes_{ST,PU}.json`. |
| `bearing_st_sweep.csv` | Amplitude-bearing azimuth per ST guide-vane-sweep impulse and the 18-vane grouping it implies. |
| `last_six_verdicts.csv` | Pre-registered matched-filter scan for the 6 slots that stayed unresolved after mark/detector pairing and template confirmation. |

## Method

Reproduction of the measurement partner's strike-detection method (attributed external reference,
`sensing-group/hydropower-anomaly`, cloned 2026-08-18) on our own raw audio, run 2026-08-18/19
(`results/strike-repro/repro_bruno_strikes.py`, on `data/illwerke-080726/20260708 Messung/`):
5-20 kHz Butterworth-4 bandpass -> 2 ms energy envelope -> per-channel MAD z-score over the 9
mics -> max across channels -> threshold k>30, frames <0.25 s merged into one impulse group;
strongest-triplet selection per position, 1.5 s gap clustering per vane. Timebase: Gantner
epoch-2000 wall-clock axis converted to true UTC (wall - 2 h).

## Edge-artifact correction

The causal bandpass filter's ring-in at each raw-file read-chunk start produced fake impulses at
read boundaries. This is not unique to our reproduction: the measurement partner's own per-minute
scan shares it — his reported "4th impulses" at several PU positions, and his lone under-pump
landmark detections, are the same ring-in transient, not real strikes. Fix: `EDGE_GUARD_S = 0.05`
(first 50 ms of every read chunk excluded from peak search). Validated by disabling the guard: the
unguarded reproduction matches the partner's published per-minute table exactly, 30/30 reference
values (24/24 position-minute counts, 6/6 sweep-minute counts).

## Tally (counted from the register CSVs in this directory, consolidated 2026-08-19)

| source | ST | PU | total |
|---|---:|---:|---:|
| both (mark + detector agree, <=0.3 s) | 82 | 80 | 162 |
| (statistical candidates merged into the ear-cued rows above) | – | – | – |
| annotated-only (5 of the 10 additionally template-hardened) | 6 | 4 | 10 |
| **measured subtotal** | **88** | **84** | **172** |
| ear-cued (final cued listening pass 2026-08-19; 2 also coincide with the statistical candidates) | 2 | 6 | 8 |
| slots | 90 | 90 | 180 |

Earlier packaging note (resolved): the registers initially shipped without the bearing-based
ST vane assignment and the two PU landmark template finds (measured count then 163/180);
`results/strike-repro/consolidate_register.py` merged both refinements on 2026-08-19, and the
registers here carry the consolidated state. The 6 rhythm-inferred slots correspond one-to-one
with the NOT FOUND verdicts in `last_six_verdicts.csv`.
## Full trail

- Analysis narrative: `master-thesis/research/notes/analysis_2026-08-18_bruno_180_hits_reproduction.md`
- Regeneration scripts: `master-thesis/results/strike-repro/` (`repro_bruno_strikes.py`,
  `build_strike_register.py`, `bearing_vane_assignment.py`, `confirm_missing_strikes.py`,
  `vibration_landmark_check.py`, `last_six_scan.py`)
