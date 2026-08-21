# Strike-register pipeline (08.07.2026 controlled-event campaign)

Derives per-strike timestamps for the 180-strike protocol (ST + PU sessions,
12 fixed positions x 3 + 18 guide-vane sweep strikes x 3, per session) of the
2026-07-08 controlled acoustic events. The impulse detector is reproduced
from the measurement partner's analysis tooling (attributed external
reference, `sensing-group/hydropower-anomaly`), with one addition: a 50 ms
chunk-edge guard (`EDGE_GUARD_S` in `repro_bruno_strikes.py`) that removes a
causal-filter ring-in artifact at every raw-file read boundary. See
`docs/groundtruth/verification-080726/README.md` for the full method note.

## Pipeline order

1. `repro_bruno_strikes.py [ST] [PU]` — impulse detector + protocol scan over
   the raw burst files. Writes `raw_*.csv` / `protocol_*.csv`.
2. `build_strike_register.py [ST] [PU]` — fills the 90-slot-per-session
   protocol register from the detector output + annotated marks. Writes
   `strikes_register_*.csv`.
3. `consolidate_register.py` — merges the bearing-based ST vane assignment
   and two template-confirmed PU landmark singles into the registers.
4. `bearing_vane_assignment.py` — amplitude-bearing azimuth for the ST vane
   sweep -> 18-vane grouping. Writes `bearing_st_sweep.csv` (run before step 3
   consumes it).
5. `confirm_missing_strikes.py` — template-based low-threshold search +
   confirmation for still-unresolved slots.
   `last_six_scan.py` — pre-registered matched-filter scan for the final 6
   unresolved slots; updates the registers in place.
6. `make_full_viewer.py` / `make_listening_pack.py` — HTML/WAV/PNG listening
   viewers for manual review.

## Running

```bash
.venv/bin/python scripts/strike_register/repro_bruno_strikes.py ST
.venv/bin/python scripts/strike_register/build_strike_register.py
.venv/bin/python scripts/strike_register/bearing_vane_assignment.py
.venv/bin/python scripts/strike_register/consolidate_register.py
.venv/bin/python scripts/strike_register/confirm_missing_strikes.py
.venv/bin/python scripts/strike_register/last_six_scan.py
```

Works with cwd anywhere. Config (`paths.py`): `ROWII_STRIKE_DATA` overrides
the raw-data root (default `data/illwerke-080726/20260708 Messung`, resolved
relative to the workspace); `ROWII_STRIKE_OUT` overrides the output root
(default `results/strike-register/`); groundtruth is fixed at
`docs/groundtruth/`. `bearing_vane_assignment.py`'s SRP-PHAT mode is optional
(`ROWII_BRUNO_SRC`, `ROWII_BRUNO_PYDEPS`) — level-bearing mode is the default
and needs no external package.

## Canonical outputs

The registers here are working outputs, regenerated on demand. The
versioned, citable copies live under `docs/groundtruth/verification-080726/`.
