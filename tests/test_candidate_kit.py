"""Tests for `scripts/candidate_kit.py`'s PURE logic: sustained-episode grouping from
fusion `alarms.parquet` rows, transient-window extraction from audio-beats rows,
extremity-ordered dedup/overlap suppression (one shared greedy primitive backs both),
the 080726 strike-exclusion mask (padded-interval overlap against BOTH the seconds-level
`docs/groundtruth/080726_strikes_seconds_*.csv` and the coarser minute-level
`080726_events_*.csv` -- the latter covers events with no per-strike rows yet, e.g. the
real PU event_id 07/13 gap), the build-side asset-window sizing rule, `compile`'s
validation (known candidate ids, fixed assessment vocabulary, provenance header), and the
impulse register path (criterion #3): session-chunk scheduling, cross-mic coincidence
matching, pair -> Candidate construction, transient cross-dedup, context notes.

Import convention mirrors `tests/test_annotation_kit.py`: `scripts/` is not a package,
so the module under test is imported directly by inserting `scripts/` onto `sys.path`.
Real parquet reads against `results/step2/once-calibrated/`, real WAV/PNG rendering and
`index.html` assembly are exercised by actually running the CLI against real data
instead, not by a test here (same split `test_annotation_kit.py` documents) -- EXCEPT
the impulse path's z-threshold, which real data alone can validate: see the final
`@pytest.mark.data` section below.
"""
from __future__ import annotations

import dataclasses
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pandas as pd
import pytest

SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import annotation_kit as ak  # noqa: E402
import candidate_kit as ck  # noqa: E402
import make_demo_assets as mda  # noqa: E402

from rowii.anomaly import impulse as impulse_mod  # noqa: E402
from rowii.config import load_config  # noqa: E402
from rowii.io.dataset import discover, run_utc_offset_ns  # noqa: E402


def _utc(
    year: int, month: int, day: int, hour: int, minute: int, second: int, micro: int = 0
) -> datetime:
    return datetime(year, month, day, hour, minute, second, micro, tzinfo=UTC)


def _ns(dt: datetime) -> int:
    return int(dt.timestamp() * 1e9)


def _alarms_frame(rows: list[dict[str, object]]) -> pd.DataFrame:
    """Build a minimal `alarms.parquet`-shaped frame from `(window, t_utc, p_value,
    role, near_transition, state_name)` dicts -- `role`/`near_transition`/`state_name`
    default to the common case so most test rows only spell out what they vary."""
    defaults = {"role": "scored", "near_transition": False, "state_name": "turbine"}
    out = []
    for r in rows:
        row = {**defaults, **r}
        out.append(
            {
                "window": row["window"],
                "t_utc_ns": _ns(row["t_utc"]),
                "p_value": row["p_value"],
                "role": row["role"],
                "near_transition": row["near_transition"],
                "state_name": row["state_name"],
            }
        )
    return pd.DataFrame(out)


# ---------------------------------------------------------------------------
# 1. build_sustained_episodes
# ---------------------------------------------------------------------------


def test_build_sustained_episodes_groups_contiguous_windows_above_min_duration() -> None:
    base = _utc(2026, 6, 25, 3, 57, 20)
    df = _alarms_frame(
        [
            {"window": 100, "t_utc": base, "p_value": 0.005},
            {"window": 101, "t_utc": base + timedelta(seconds=1), "p_value": 0.002},
            {"window": 102, "t_utc": base + timedelta(seconds=2), "p_value": 0.008},
        ]
    )

    out = ck.build_sustained_episodes(
        df, window_s=1.0, alpha=0.01, min_duration_s=3.0,
        session="250526-tu", modality="fusion", regime="recalibrate", alarms_path="a.parquet",
    )

    assert len(out) == 1
    ep = out[0]
    assert ep.session == "250526-tu"
    assert ep.klass == "sustained"
    assert ep.start_utc == base
    assert ep.end_utc == base + timedelta(seconds=3)  # last window (t+2s) + window_s
    assert ep.duration_s == pytest.approx(3.0)
    assert ep.min_p == pytest.approx(0.002)
    assert ep.n_windows == 3
    assert ep.modality == "fusion" and ep.regime == "recalibrate" and ep.alarms_path == "a.parquet"


def test_build_sustained_episodes_drops_episode_shorter_than_min_duration() -> None:
    base = _utc(2026, 6, 25, 3, 57, 20)
    df = _alarms_frame(
        [
            {"window": 100, "t_utc": base, "p_value": 0.005},
            {"window": 101, "t_utc": base + timedelta(seconds=1), "p_value": 0.002},
        ]
    )  # 2 windows -> 2.0s duration, below the 3.0s floor

    out = ck.build_sustained_episodes(
        df, window_s=1.0, alpha=0.01, min_duration_s=3.0,
        session="s", modality="fusion", regime="frozen", alarms_path="a.parquet",
    )

    assert out == []


def test_build_sustained_episodes_a_window_index_gap_breaks_the_episode() -> None:
    base = _utc(2026, 6, 25, 3, 57, 20)
    df = _alarms_frame(
        [
            {"window": 100, "t_utc": base, "p_value": 0.005},
            {"window": 101, "t_utc": base + timedelta(seconds=1), "p_value": 0.004},
            {"window": 102, "t_utc": base + timedelta(seconds=2), "p_value": 0.003},
            # window 103 missing (e.g. an invalid grid window) -- real gap, t jumps 2s
            {"window": 104, "t_utc": base + timedelta(seconds=4), "p_value": 0.002},
            {"window": 105, "t_utc": base + timedelta(seconds=5), "p_value": 0.001},
            {"window": 106, "t_utc": base + timedelta(seconds=6), "p_value": 0.0015},
        ]
    )

    out = ck.build_sustained_episodes(
        df, window_s=1.0, alpha=0.01, min_duration_s=3.0,
        session="s", modality="fusion", regime="frozen", alarms_path="a.parquet",
    )

    # Two separate 3-window episodes, NOT one bridged 6-window run.
    assert len(out) == 2
    assert out[0].n_windows == 3 and out[0].start_utc == base
    assert out[1].n_windows == 3 and out[1].start_utc == base + timedelta(seconds=4)


def test_build_sustained_episodes_ignores_non_scored_and_above_alpha_rows() -> None:
    base = _utc(2026, 6, 25, 3, 57, 20)
    df = _alarms_frame(
        [
            {"window": 100, "t_utc": base, "p_value": 0.5},  # above alpha
            {"window": 101, "t_utc": base + timedelta(seconds=1), "p_value": 0.001,
             "role": "consumed_for_calibration"},  # excluded by role even though p<alpha
            {"window": 102, "t_utc": base + timedelta(seconds=2), "p_value": None,
             "role": "unknown_state"},
        ]
    )

    out = ck.build_sustained_episodes(
        df, window_s=1.0, alpha=0.01, min_duration_s=3.0,
        session="s", modality="fusion", regime="frozen", alarms_path="a.parquet",
    )

    assert out == []


def test_build_sustained_episodes_state_name_and_near_transition_from_extremum_and_any() -> None:
    base = _utc(2026, 6, 25, 3, 57, 20)
    df = _alarms_frame(
        [
            {"window": 100, "t_utc": base, "p_value": 0.005, "state_name": "turbine",
             "near_transition": False},
            {"window": 101, "t_utc": base + timedelta(seconds=1), "p_value": 0.001,
             "state_name": "standstill", "near_transition": True},  # the extremum row
            {"window": 102, "t_utc": base + timedelta(seconds=2), "p_value": 0.004,
             "state_name": "turbine", "near_transition": False},
        ]
    )

    out = ck.build_sustained_episodes(
        df, window_s=1.0, alpha=0.01, min_duration_s=3.0,
        session="s", modality="fusion", regime="frozen", alarms_path="a.parquet",
    )

    assert len(out) == 1
    assert out[0].state_name == "standstill"  # from the min-p row, not the first row
    assert out[0].near_transition is True  # OR across the episode


# ---------------------------------------------------------------------------
# 2. build_transient_candidates
# ---------------------------------------------------------------------------


def test_build_transient_candidates_one_row_per_scored_window_below_alpha() -> None:
    base = _utc(2026, 7, 8, 12, 43, 1)
    df = _alarms_frame(
        [
            {"window": 10, "t_utc": base, "p_value": 0.0005},
            {"window": 20, "t_utc": base + timedelta(seconds=50), "p_value": 0.05},  # above alpha
        ]
    )

    out = ck.build_transient_candidates(
        df, window_s=1.0, alpha=0.001,
        session="080726-pu_strikes", modality="audio-beats", regime="recalibrate",
        alarms_path="b.parquet",
    )

    assert len(out) == 1
    cand = out[0]
    assert cand.klass == "transient"
    assert cand.start_utc == base
    assert cand.end_utc == base + timedelta(seconds=1.0)
    assert cand.duration_s == pytest.approx(1.0)
    assert cand.min_p == pytest.approx(0.0005)
    assert cand.n_windows == 1


# ---------------------------------------------------------------------------
# 3. dedupe_by_radius / suppress_overlaps -- shared greedy, most-extreme-first primitive
# ---------------------------------------------------------------------------


def _sustained(start: datetime, dur_s: float, min_p: float, session: str = "s") -> ck.Candidate:
    return ck.Candidate(
        session=session, klass="sustained", start_utc=start,
        end_utc=start + timedelta(seconds=dur_s),
        duration_s=dur_s, min_p=min_p, state_name="turbine", near_transition=False, n_windows=3,
        modality="fusion", regime="frozen", alarms_path="a.parquet",
    )


def _transient(start: datetime, min_p: float, session: str = "s") -> ck.Candidate:
    return ck.Candidate(
        session=session, klass="transient", start_utc=start, end_utc=start + timedelta(seconds=1.0),
        duration_s=1.0, min_p=min_p, state_name="turbine", near_transition=False, n_windows=1,
        modality="audio-beats", regime="frozen", alarms_path="b.parquet",
    )


def _impulse(start: datetime, min_p: float, session: str = "s") -> ck.Candidate:
    return ck.Candidate(
        session=session, klass="impulse", start_utc=start,
        end_utc=start + timedelta(seconds=impulse_mod.FRAME_S),
        duration_s=impulse_mod.FRAME_S, min_p=min_p, state_name="n/a", near_transition=False,
        n_windows=1, modality="impulse", regime="n/a", alarms_path="",
    )


def test_dedupe_by_radius_drops_the_less_extreme_neighbour_within_radius() -> None:
    t0 = _utc(2026, 7, 8, 12, 0, 0)
    a = _transient(t0, min_p=0.0009)
    b = _transient(t0 + timedelta(seconds=3), min_p=0.0001)  # more extreme, 3s away -> within 5s

    kept = ck.dedupe_by_radius([a, b], radius_s=5.0)

    assert kept == [b]


def test_dedupe_by_radius_keeps_both_when_outside_radius() -> None:
    t0 = _utc(2026, 7, 8, 12, 0, 0)
    a = _transient(t0, min_p=0.0009)
    b = _transient(t0 + timedelta(seconds=5.5), min_p=0.0001)  # just outside +/-5s

    kept = ck.dedupe_by_radius([a, b], radius_s=5.0)

    assert {c.start_utc for c in kept} == {a.start_utc, b.start_utc}


def test_dedupe_by_radius_boundary_exactly_at_radius_is_not_a_collision() -> None:
    t0 = _utc(2026, 7, 8, 12, 0, 0)
    a = _transient(t0, min_p=0.0009)
    b = _transient(t0 + timedelta(seconds=5.0), min_p=0.0001)  # exactly 5.0s away

    kept = ck.dedupe_by_radius([a, b], radius_s=5.0)

    # Half-open padded-span convention (codebase-wide): touching exactly at the pad
    # boundary is NOT an overlap -- both survive.
    assert {c.start_utc for c in kept} == {a.start_utc, b.start_utc}


def test_dedupe_by_radius_returns_original_unwidened_spans() -> None:
    t0 = _utc(2026, 7, 8, 12, 0, 0)
    a = _transient(t0, min_p=0.0001)

    kept = ck.dedupe_by_radius([a], radius_s=5.0)

    assert kept[0].start_utc == t0
    assert kept[0].end_utc == t0 + timedelta(seconds=1.0)


def test_suppress_overlaps_keeps_the_more_extreme_of_two_overlapping_candidates() -> None:
    t0 = _utc(2026, 7, 8, 12, 0, 0)
    sustained = _sustained(t0, dur_s=5.0, min_p=0.008)  # spans [t0, t0+5s)
    transient = _transient(t0 + timedelta(seconds=2), min_p=0.0001)  # inside the span, more extreme

    kept = ck.suppress_overlaps([sustained, transient])

    assert len(kept) == 1
    assert kept[0].min_p == pytest.approx(0.0001)
    assert kept[0].klass == "transient"


def test_suppress_overlaps_keeps_both_disjoint_candidates() -> None:
    t0 = _utc(2026, 7, 8, 12, 0, 0)
    a = _sustained(t0, dur_s=3.0, min_p=0.008)
    b = _transient(t0 + timedelta(seconds=30), min_p=0.0005)

    kept = ck.suppress_overlaps([a, b])

    assert len(kept) == 2


def test_suppress_overlaps_touching_boundary_is_not_an_overlap() -> None:
    t0 = _utc(2026, 7, 8, 12, 0, 0)
    a = _sustained(t0, dur_s=3.0, min_p=0.008)  # [t0, t0+3s)
    b = _transient(t0 + timedelta(seconds=3), min_p=0.0005)  # starts exactly at a's end

    kept = ck.suppress_overlaps([a, b])

    assert len(kept) == 2


# ---------------------------------------------------------------------------
# 4. strike-exclusion mask
# ---------------------------------------------------------------------------


def test_overlaps_any_true_inside_a_padded_interval() -> None:
    strike = _utc(2026, 7, 8, 12, 43, 1)
    intervals = [(strike - timedelta(seconds=30), strike + timedelta(seconds=30))]

    inside_start = strike + timedelta(seconds=10)
    inside_end = strike + timedelta(seconds=12)
    assert ck.overlaps_any(inside_start, inside_end, intervals)


def test_overlaps_any_false_just_outside_a_padded_interval() -> None:
    strike = _utc(2026, 7, 8, 12, 43, 1)
    intervals = [(strike - timedelta(seconds=30), strike + timedelta(seconds=30))]
    outside_start = strike + timedelta(seconds=31)

    assert not ck.overlaps_any(outside_start, outside_start + timedelta(seconds=1), intervals)


def _write_strikes_csv(path: Path, rows: list[tuple[str, str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["session,event_id,kind,strike_no,strike_utc,confidence,notes"]
    for event_id, kind, strike_utc in rows:
        lines.append(f"pu,{event_id},{kind},1,{strike_utc},,")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_events_csv(path: Path, rows: list[tuple[str, str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["start_utc,end_utc,kind"]
    for start, end, kind in rows:
        lines.append(f"{start},{end},{kind}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_load_strike_exclusion_intervals_covers_per_strike_rows(tmp_path: Path) -> None:
    strikes_csv = tmp_path / "strikes.csv"
    events_csv = tmp_path / "events.csv"
    _write_strikes_csv(strikes_csv, [("01", "plate-gen_0", "2026-07-08T12:43:01.568000+00:00")])
    _write_events_csv(
        events_csv, [("2026-07-08T12:43:00+00:00", "2026-07-08T12:44:00+00:00", "plate-gen_0")]
    )

    intervals = ck.load_strike_exclusion_intervals(strikes_csv, events_csv, pad_s=30.0)

    strike_t = _utc(2026, 7, 8, 12, 43, 1, 568000)
    assert ck.overlaps_any(strike_t, strike_t + timedelta(seconds=1), intervals)
    # 40s after the strike is inside the EVENT-derived interval (event end 12:44:00 + 30s pad)
    # even though it is well outside the strike-derived one (strike + 30s = 12:43:31.568).
    far_but_in_event_pad = _utc(2026, 7, 8, 12, 44, 20)
    far_end = far_but_in_event_pad + timedelta(seconds=1)
    assert ck.overlaps_any(far_but_in_event_pad, far_end, intervals)


def test_load_strike_exclusion_intervals_handles_mixed_fractional_second_precision(
    tmp_path: Path,
) -> None:
    """Real regression: `080726_strikes_seconds_pu.csv`/`_st.csv` each have exactly
    one row with NO fractional seconds (a strike landing on a whole second) mixed
    in among hundreds of microsecond-precision rows in the SAME column -- plain
    `pd.to_datetime(..., utc=True)` locks onto one format from the first rows and
    raises on the other; this must not crash."""
    strikes_csv = tmp_path / "strikes.csv"
    events_csv = tmp_path / "events.csv"
    _write_strikes_csv(
        strikes_csv,
        [
            ("08", "vane-sweep", "2026-07-08T12:50:57.459000+00:00"),
            ("08", "vane-sweep", "2026-07-08T12:50:58+00:00"),  # whole-second, no fraction
        ],
    )
    _write_events_csv(
        events_csv, [("2026-07-08T12:50:00+00:00", "2026-07-08T12:53:00+00:00", "vane-sweep")]
    )

    intervals = ck.load_strike_exclusion_intervals(strikes_csv, events_csv, pad_s=30.0)

    whole_second_strike = _utc(2026, 7, 8, 12, 50, 58)
    whole_second_end = whole_second_strike + timedelta(seconds=1)
    assert ck.overlaps_any(whole_second_strike, whole_second_end, intervals)


def test_load_strike_exclusion_intervals_covers_events_with_no_strike_rows(tmp_path: Path) -> None:
    """Real gap in `080726_strikes_seconds_pu.csv`: event_id 07 (landmark-A_kugelschieber,
    12:49-12:50) has zero compiled per-strike rows, but the coarser events.csv still lists
    it -- a candidate inside that minute must still be excluded."""
    strikes_csv = tmp_path / "strikes.csv"
    events_csv = tmp_path / "events.csv"
    _write_strikes_csv(strikes_csv, [("01", "plate-gen_0", "2026-07-08T12:43:01.568000+00:00")])
    _write_events_csv(
        events_csv,
        [
            ("2026-07-08T12:43:00+00:00", "2026-07-08T12:44:00+00:00", "plate-gen_0"),
            ("2026-07-08T12:49:00+00:00", "2026-07-08T12:50:00+00:00", "landmark-A_kugelschieber"),
        ],
    )

    intervals = ck.load_strike_exclusion_intervals(strikes_csv, events_csv, pad_s=30.0)

    inside_uncompiled_event = _utc(2026, 7, 8, 12, 49, 30)
    assert ck.overlaps_any(
        inside_uncompiled_event, inside_uncompiled_event + timedelta(seconds=1), intervals
    )


# ---------------------------------------------------------------------------
# 5. asset_window
# ---------------------------------------------------------------------------


def test_asset_window_pads_both_sides_when_within_bounds() -> None:
    start, end = _utc(2026, 7, 8, 12, 0, 0), _utc(2026, 7, 8, 12, 0, 5)  # 5s candidate

    w_start, w_end = ck.asset_window(start, end, pad_s=10.0, min_s=20.0, max_s=60.0)

    assert w_start == start - timedelta(seconds=10)
    assert w_end == end + timedelta(seconds=10)
    assert (w_end - w_start).total_seconds() == pytest.approx(25.0)


def test_asset_window_expands_to_the_minimum_duration() -> None:
    start, end = _utc(2026, 7, 8, 12, 0, 0), _utc(2026, 7, 8, 12, 0, 1)  # 1s transient candidate

    w_start, w_end = ck.asset_window(start, end, pad_s=1.0, min_s=20.0, max_s=60.0)
    # padded span is only 3s (1 + 2*1) -- must expand symmetrically to 20s around the midpoint
    mid = start + (end - start) / 2

    assert (w_end - w_start).total_seconds() == pytest.approx(20.0)
    assert w_start == mid - timedelta(seconds=10)
    assert w_end == mid + timedelta(seconds=10)


def test_asset_window_clips_to_the_maximum_duration() -> None:
    start, end = _utc(2026, 7, 8, 12, 0, 0), _utc(2026, 7, 8, 12, 1, 0)  # 60s candidate

    w_start, w_end = ck.asset_window(start, end, pad_s=10.0, min_s=20.0, max_s=60.0)
    # padded span would be 80s -- clip symmetrically to 60s around the midpoint
    mid = start + (end - start) / 2

    assert (w_end - w_start).total_seconds() == pytest.approx(60.0)
    assert w_start == mid - timedelta(seconds=30)
    assert w_end == mid + timedelta(seconds=30)


# ---------------------------------------------------------------------------
# 6. cap_top_n / assign_ids
# ---------------------------------------------------------------------------


def test_cap_top_n_keeps_the_most_extreme_and_reports_counts() -> None:
    t0 = _utc(2026, 7, 8, 12, 0, 0)
    candidates = [
        _transient(t0 + timedelta(seconds=i * 60), min_p=0.001 - i * 1e-5) for i in range(5)
    ]

    kept = ck.cap_top_n(candidates, 2)

    assert len(kept) == 2
    assert [c.min_p for c in kept] == sorted(c.min_p for c in candidates)[:2]


def test_assign_ids_ranks_by_extremity_and_ids_carry_the_session() -> None:
    t0 = _utc(2026, 7, 8, 12, 0, 0)
    a = _transient(t0, min_p=0.0009, session="290626-tu")
    b = _transient(t0 + timedelta(seconds=60), min_p=0.0001, session="290626-tu")

    out = ck.assign_ids([a, b])

    assert out[0].min_p == pytest.approx(0.0001) and out[0].rank == 1
    assert out[0].candidate_id == "290626-tu-01"
    assert out[1].rank == 2 and out[1].candidate_id == "290626-tu-02"


# ---------------------------------------------------------------------------
# 7. compile validation
# ---------------------------------------------------------------------------


def _candidate(cid: str, session: str = "290626-tu") -> ck.Candidate:
    t0 = _utc(2026, 6, 29, 1, 11, 20)
    return ck.Candidate(
        session=session, klass="transient", start_utc=t0, end_utc=t0 + timedelta(seconds=1.0),
        duration_s=1.0, min_p=0.0002, state_name="turbine", near_transition=False, n_windows=1,
        modality="audio-beats", regime="frozen", alarms_path="b.parquet",
        candidate_id=cid, rank=1,
    )


def test_validate_and_merge_assessments_accepts_known_ids_and_valid_vocabulary() -> None:
    candidates_by_id = {"290626-tu-01": _candidate("290626-tu-01")}
    df = pd.DataFrame(
        [
            {
                "candidate_id": "290626-tu-01", "assessment": "plausible anomaly",
                "note": "clicky transient",
            }
        ]
    )

    out = ck.validate_and_merge_assessments(df, candidates_by_id)

    assert len(out) == 1
    assert out[0]["assessment"] == "plausible anomaly"
    assert out[0]["note"] == "clicky transient"
    # session is re-derived from the reference candidate, never trusted from the export
    assert out[0]["session"] == "290626-tu"


def test_validate_and_merge_assessments_skips_unreviewed_rows() -> None:
    candidates_by_id = {"290626-tu-01": _candidate("290626-tu-01")}
    df = pd.DataFrame([{"candidate_id": "290626-tu-01", "assessment": "", "note": ""}])

    out = ck.validate_and_merge_assessments(df, candidates_by_id)

    assert out == []


def test_validate_and_merge_assessments_rejects_unknown_candidate_id() -> None:
    candidates_by_id = {"290626-tu-01": _candidate("290626-tu-01")}
    df = pd.DataFrame([{"candidate_id": "no-such-id", "assessment": "unclear", "note": ""}])

    with pytest.raises(ValueError, match="unknown candidate_id"):
        ck.validate_and_merge_assessments(df, candidates_by_id)


def test_validate_and_merge_assessments_rejects_out_of_vocabulary_assessment() -> None:
    candidates_by_id = {"290626-tu-01": _candidate("290626-tu-01")}
    df = pd.DataFrame(
        [{"candidate_id": "290626-tu-01", "assessment": "definitely a fault", "note": ""}]
    )

    with pytest.raises(ValueError, match="not one of"):
        ck.validate_and_merge_assessments(df, candidates_by_id)


def test_validate_and_merge_assessments_raises_before_writing_anything_on_a_later_bad_row() -> None:
    """Mirrors `annotation_kit.compile_template`'s all-validate-then-write contract: a bad
    row anywhere in the file must fail the WHOLE call, not just be skipped."""
    candidates_by_id = {
        "290626-tu-01": _candidate("290626-tu-01"),
        "290626-tu-02": _candidate("290626-tu-02"),
    }
    df = pd.DataFrame(
        [
            {"candidate_id": "290626-tu-01", "assessment": "unclear", "note": ""},
            {"candidate_id": "290626-tu-02", "assessment": "bogus-value", "note": ""},
        ]
    )

    with pytest.raises(ValueError, match="not one of"):
        ck.validate_and_merge_assessments(df, candidates_by_id)


def test_write_assessments_csv_writes_provenance_header_and_rows(tmp_path: Path) -> None:
    from datetime import date

    rows = [
        {
            "session": "290626-tu", "candidate_id": "290626-tu-01", "class": "transient",
            "start_utc": "2026-06-29T01:11:20+00:00", "duration_s": 1.0, "min_p": 0.0002,
            "state_name": "turbine", "near_transition": False,
            "scada_state": "pump", "scada_transition": False,
            "assessment": "plausible anomaly", "note": "clicky",
        }
    ]
    out_path = tmp_path / "candidate_assessments_2026-08-15.csv"

    ck.write_assessments_csv(
        rows, out_path, source_path=tmp_path / "export.csv", compiled_date=date(2026, 8, 15)
    )

    text = out_path.read_text(encoding="utf-8")
    assert text.startswith("#")
    assert "not ground truth" in text
    frame = pd.read_csv(out_path, comment="#")
    assert list(frame["candidate_id"]) == ["290626-tu-01"]
    assert list(frame.columns) == list(ck._ASSESSMENTS_CSV_COLUMNS)
    # scada_state is carried through even though the detector's own state_name says
    # something else -- exactly the reviewer-relevant mismatch this kit surfaces.
    assert frame.loc[0, "scada_state"] == "pump"
    assert frame.loc[0, "state_name"] == "turbine"


# ---------------------------------------------------------------------------
# 8. validate_and_merge_assessments re-derives scada_state/scada_transition too
# ---------------------------------------------------------------------------


def test_validate_and_merge_assessments_re_derives_scada_columns_from_candidate() -> None:
    t0 = _utc(2026, 6, 29, 7, 53, 49)
    candidate = ck.Candidate(
        session="290626-pu", klass="transient", start_utc=t0, end_utc=t0 + timedelta(seconds=1.0),
        duration_s=1.0, min_p=0.000472, state_name="turbine", near_transition=False, n_windows=1,
        modality="audio-beats", regime="recalibrate", alarms_path="b.parquet",
        candidate_id="290626-pu-01", rank=1, scada_state="pump", scada_transition=False,
    )
    df = pd.DataFrame(
        [{"candidate_id": "290626-pu-01", "assessment": "plausible anomaly", "note": ""}]
    )

    out = ck.validate_and_merge_assessments(df, {"290626-pu-01": candidate})

    assert out[0]["scada_state"] == "pump"
    assert out[0]["scada_transition"] is False
    assert out[0]["state_name"] == "turbine"  # detector's own guess, kept alongside SCADA


# ---------------------------------------------------------------------------
# 9. scada_majority_state
# ---------------------------------------------------------------------------


def _window_series(
    first_start: datetime, states: list[str], window_s: float = 1.0
) -> tuple[list[datetime], list[str]]:
    starts = [first_start + timedelta(seconds=i * window_s) for i in range(len(states))]
    return starts, states


def test_scada_majority_state_simple_majority_wins() -> None:
    t0 = _utc(2026, 6, 29, 7, 53, 45)
    starts, states = _window_series(t0, ["pump", "pump", "transition", "transition", "transition"])

    state, touches = ck.scada_majority_state(
        starts, 1.0, states, t0, t0 + timedelta(seconds=5)
    )

    assert state == "transition"
    assert touches is True


def test_scada_majority_state_only_counts_windows_overlapping_the_span() -> None:
    t0 = _utc(2026, 6, 29, 7, 53, 45)
    # 5 one-second windows: pump, pump, pump, pump, standstill -- candidate span only
    # covers the LAST two windows (transition into standstill).
    starts, states = _window_series(t0, ["pump", "pump", "pump", "pump", "standstill"])
    span_start = t0 + timedelta(seconds=3)
    span_end = t0 + timedelta(seconds=5)

    state, touches = ck.scada_majority_state(starts, 1.0, states, span_start, span_end)

    assert state == "pump"  # windows 3 and 4 overlap: one "pump", one "standstill" -- tie
    assert touches is False


def test_scada_majority_state_ties_broken_by_earliest_occurrence() -> None:
    # Exactly 2 "pump" and 2 "standstill" windows, "pump" occurs first chronologically.
    t0 = _utc(2026, 7, 8, 15, 16, 24)
    starts, states = _window_series(t0, ["pump", "standstill", "pump", "standstill"])

    state, _ = ck.scada_majority_state(starts, 1.0, states, t0, t0 + timedelta(seconds=4))

    assert state == "pump"


def test_scada_majority_state_no_overlapping_window_returns_unknown() -> None:
    t0 = _utc(2026, 6, 29, 7, 53, 45)
    starts, states = _window_series(t0, ["pump", "pump"])
    far_start = t0 + timedelta(hours=1)

    state, touches = ck.scada_majority_state(
        starts, 1.0, states, far_start, far_start + timedelta(seconds=1)
    )

    assert state == "unknown"
    assert touches is False


def test_scada_majority_state_touching_boundary_is_not_an_overlap() -> None:
    t0 = _utc(2026, 6, 29, 7, 53, 45)
    starts, states = _window_series(t0, ["pump"])  # window covers [t0, t0+1s)

    state, _ = ck.scada_majority_state(
        starts, 1.0, states, t0 + timedelta(seconds=1), t0 + timedelta(seconds=2)
    )

    assert state == "unknown"  # candidate span starts exactly where the window ends


# ---------------------------------------------------------------------------
# 10. scada_detector_mismatch
# ---------------------------------------------------------------------------


def test_scada_detector_mismatch_true_when_states_differ() -> None:
    assert ck.scada_detector_mismatch("pump", "turbine") is True


def test_scada_detector_mismatch_false_when_states_agree() -> None:
    assert ck.scada_detector_mismatch("turbine", "turbine") is False


def test_scada_detector_mismatch_unknown_scada_never_counts_as_mismatch() -> None:
    assert ck.scada_detector_mismatch("unknown", "turbine") is False


# ---------------------------------------------------------------------------
# 11. criterion_sentence
# ---------------------------------------------------------------------------


def test_criterion_sentence_sustained_cites_window_count_duration_and_alpha() -> None:
    cand = _sustained(_utc(2026, 6, 25, 3, 57, 20), dur_s=3.0, min_p=0.0017)

    text = ck.criterion_sentence(cand)

    assert "fusion path" in text
    assert "3 consecutive windows" in text
    assert "p < 0.01" in text
    assert "3.0 s" in text


def test_criterion_sentence_transient_cites_beats_and_alpha() -> None:
    cand = _transient(_utc(2026, 7, 8, 12, 43, 1), min_p=0.00025)

    text = ck.criterion_sentence(cand)

    assert "BEATs" in text
    assert "p < 0.001" in text


def test_criterion_sentence_unknown_class_raises() -> None:
    base = _transient(_utc(2026, 7, 8, 12, 43, 1), min_p=0.00025)
    cand = dataclasses.replace(base, klass="bogus")

    with pytest.raises(ValueError, match="unknown candidate class"):
        ck.criterion_sentence(cand)


def test_criterion_sentence_impulse_cites_band_search_and_zscore() -> None:
    cand = _impulse(_utc(2026, 6, 29, 2, 0, 15), min_p=1e-15)

    text = ck.criterion_sentence(cand)

    assert "5-20 kHz" in text
    assert "both microphones" in text
    assert "0.15 s" in text
    # z is recovered from min_p via the inverse of the norm.sf transform build_impulse_pairs uses.
    assert "z" in text.lower()


def test_criterion_sentence_impulse_at_the_min_p_floor_shows_a_finite_z() -> None:
    # A candidate built from an extreme (real, observed) z-pair has min_p floored
    # at ck._MIN_P_FLOOR (not exactly 0.0) -- the sentence must show a finite z,
    # never "inf" (norm.isf(0.0) would be infinite).
    cand = _impulse(_utc(2026, 6, 29, 2, 0, 15), min_p=ck._MIN_P_FLOOR)

    text = ck.criterion_sentence(cand)

    assert "inf" not in text.lower()


# ---------------------------------------------------------------------------
# 12. load_prior_candidate_ids / pin_stable_ids -- cross-run id stability
# ---------------------------------------------------------------------------


def test_load_prior_candidate_ids_returns_empty_dict_when_file_missing(tmp_path: Path) -> None:
    assert ck.load_prior_candidate_ids(tmp_path / "does-not-exist.csv") == {}


def test_load_prior_candidate_ids_reads_session_start_utc_keyed_map(tmp_path: Path) -> None:
    csv_path = tmp_path / "candidates.csv"
    t0 = _utc(2026, 6, 29, 1, 11, 20)
    t1 = _utc(2026, 6, 29, 1, 12, 30)
    ranked = ck.assign_ids(
        [
            _transient(t0, min_p=0.0009, session="290626-tu"),
            _transient(t1, min_p=0.0002, session="290626-tu"),
        ]
    )
    frame = pd.DataFrame([ck._candidate_to_row(c) for c in ranked])
    frame.to_csv(csv_path, index=False)

    prior = ck.load_prior_candidate_ids(csv_path)

    # t1 is more extreme (lower min_p) -> rank 1 -> "290626-tu-01"; t0 -> rank 2.
    assert prior[("290626-tu", t1.isoformat())] == "290626-tu-01"
    assert prior[("290626-tu", t0.isoformat())] == "290626-tu-02"


def test_pin_stable_ids_is_a_no_op_with_no_prior_ids() -> None:
    t0 = _utc(2026, 7, 8, 12, 0, 0)
    ranked = ck.assign_ids(
        [_transient(t0, min_p=0.0009), _transient(t0 + timedelta(seconds=60), min_p=0.0001)]
    )

    out = ck.pin_stable_ids(ranked, {})

    assert [c.candidate_id for c in out] == [c.candidate_id for c in ranked]


def test_pin_stable_ids_empty_candidates_returns_empty_list() -> None:
    assert ck.pin_stable_ids([], {}) == []


def test_pin_stable_ids_keeps_old_id_when_rank_shifts(tmp_path: Path) -> None:
    t_a = _utc(2026, 7, 8, 12, 0, 0)
    t_b = _utc(2026, 7, 8, 12, 5, 0)
    # OLD run: A was rank 1 ("s-01"), B was rank 2 ("s-02").
    prior_ids = {
        ("s", t_a.isoformat()): "s-01",
        ("s", t_b.isoformat()): "s-02",
    }
    # NEW run: B is now MORE extreme than A, so assign_ids alone would swap their
    # numbers (B -> rank 1/"s-01", A -> rank 2/"s-02").
    a = _transient(t_a, min_p=0.005, session="s")
    b = _transient(t_b, min_p=0.0001, session="s")
    ranked = ck.assign_ids([a, b])
    assert [c.candidate_id for c in ranked] == ["s-01", "s-02"]  # B first (id "s-01"), A second

    out = ck.pin_stable_ids(ranked, prior_ids)

    by_start = {c.start_utc: c.candidate_id for c in out}
    assert by_start[t_a] == "s-01"  # A keeps its OLD id ...
    assert by_start[t_b] == "s-02"  # ... and B keeps ITS old id -- no id changed meaning.
    # rank still reflects the CURRENT extremity order (B is now more extreme than A).
    by_start_rank = {c.start_utc: c.rank for c in out}
    assert by_start_rank[t_b] == 1
    assert by_start_rank[t_a] == 2


def test_pin_stable_ids_new_candidate_never_reuses_a_vanished_candidates_old_slot() -> None:
    t_a = _utc(2026, 7, 8, 12, 0, 0)
    t_b = _utc(2026, 7, 8, 12, 5, 0)  # B existed in the OLD run, gone from the NEW one
    t_c = _utc(2026, 7, 8, 12, 10, 0)  # C is brand new this run
    prior_ids = {
        ("s", t_a.isoformat()): "s-01",
        ("s", t_b.isoformat()): "s-02",
    }
    # NEW run: C is now the MOST extreme (would naturally get "s-01"), A second.
    c = _transient(t_c, min_p=0.0001, session="s")
    a = _transient(t_a, min_p=0.005, session="s")
    ranked = ck.assign_ids([a, c])
    assert [cand.candidate_id for cand in ranked] == ["s-01", "s-02"]  # C, then A

    out = ck.pin_stable_ids(ranked, prior_ids)

    by_start = {cand.start_utc: cand.candidate_id for cand in out}
    assert by_start[t_a] == "s-01"  # A's OLD id, preserved
    # C must NOT become "s-02" -- that slot is retired (belonged to vanished B), and
    # reusing it would let C silently inherit B's old localStorage assessment.
    assert by_start[t_c] not in {"s-01", "s-02"}
    assert by_start[t_c] == "s-03"
    assert len({cand.candidate_id for cand in out}) == 2  # no duplicate ids either way


def test_pin_stable_ids_only_pins_within_the_same_session() -> None:
    t0 = _utc(2026, 7, 8, 12, 0, 0)
    prior_ids = {("other-session", t0.isoformat()): "other-session-01"}
    cand = _transient(t0, min_p=0.0009, session="s")
    ranked = ck.assign_ids([cand])

    out = ck.pin_stable_ids(ranked, prior_ids)

    assert out[0].candidate_id == "s-01"  # unaffected by a different session's prior id


# ---------------------------------------------------------------------------
# 13. SCADA context panel: slicing, 1 Hz resampling, ribbon rows, readout, colors
# ---------------------------------------------------------------------------


def test_slice_window_series_keeps_only_windows_overlapping_the_span() -> None:
    t0 = _utc(2026, 6, 29, 4, 47, 20)
    starts, values = _window_series(t0, [10.0, 20.0, 30.0, 40.0])  # type: ignore[arg-type]

    sliced_starts, sliced_values = ck.slice_window_series(
        starts, 1.0, values, t0 + timedelta(seconds=1), t0 + timedelta(seconds=3)
    )

    assert sliced_starts == [t0 + timedelta(seconds=1), t0 + timedelta(seconds=2)]
    assert sliced_values == [20.0, 30.0]


def test_slice_window_series_excludes_a_window_touching_the_span_boundary() -> None:
    t0 = _utc(2026, 6, 29, 4, 47, 20)
    starts, values = _window_series(t0, [1.0, 2.0, 3.0])  # type: ignore[arg-type]

    # Span starts exactly where window 0 ([t0, t0+1)) ends -- touching, not overlapping.
    sliced_starts, _sliced_values = ck.slice_window_series(
        starts, 1.0, values, t0 + timedelta(seconds=1), t0 + timedelta(seconds=2)
    )

    assert sliced_starts == [t0 + timedelta(seconds=1)]


def test_slice_window_series_empty_when_span_is_far_outside() -> None:
    t0 = _utc(2026, 6, 29, 4, 47, 20)
    starts, values = _window_series(t0, [1.0, 2.0])  # type: ignore[arg-type]

    sliced_starts, sliced_values = ck.slice_window_series(
        starts, 1.0, values, t0 + timedelta(hours=1), t0 + timedelta(hours=1, seconds=1)
    )

    assert sliced_starts == []
    assert sliced_values == []


def test_resample_channel_to_seconds_passes_through_native_1s_windows() -> None:
    t0 = _utc(2026, 6, 29, 4, 47, 20)
    starts, values = _window_series(t0, [10.0, 12.0, 14.0])  # type: ignore[arg-type]

    out = ck.resample_channel_to_seconds(starts, 1.0, values, t0, 3)

    assert out == [10.0, 12.0, 14.0]


def test_resample_channel_to_seconds_averages_sub_second_native_windows() -> None:
    t0 = _utc(2026, 6, 29, 4, 47, 20)
    starts = [t0, t0 + timedelta(seconds=0.5)]
    values = [10.0, 20.0]

    out = ck.resample_channel_to_seconds(starts, 0.5, values, t0, 1)

    assert out == [pytest.approx(15.0)]


def test_resample_channel_to_seconds_ignores_nan_values_in_the_mean() -> None:
    t0 = _utc(2026, 6, 29, 4, 47, 20)
    starts = [t0, t0 + timedelta(seconds=0.5)]
    values = [float("nan"), 20.0]

    out = ck.resample_channel_to_seconds(starts, 0.5, values, t0, 1)

    assert out == [pytest.approx(20.0)]


def test_resample_channel_to_seconds_none_when_no_window_overlaps() -> None:
    t0 = _utc(2026, 6, 29, 4, 47, 20)
    starts, values = _window_series(t0, [5.0])  # type: ignore[arg-type]

    out = ck.resample_channel_to_seconds(starts, 1.0, values, t0 + timedelta(seconds=10), 1)

    assert out == [None]


def test_resample_channel_to_seconds_none_when_every_overlapping_value_is_nan() -> None:
    t0 = _utc(2026, 6, 29, 4, 47, 20)
    starts, values = _window_series(t0, [float("nan"), float("nan")])  # type: ignore[arg-type]

    out = ck.resample_channel_to_seconds(starts, 1.0, values, t0, 2)

    assert out == [None, None]


def test_resample_states_to_seconds_majority_per_second() -> None:
    t0 = _utc(2026, 6, 29, 4, 47, 20)
    starts, states = _window_series(t0, ["pump", "pump", "standstill"])

    out = ck.resample_states_to_seconds(starts, 1.0, states, t0, 3)

    assert out == ["pump", "pump", "standstill"]


def test_resample_states_to_seconds_unknown_when_no_window_overlaps_that_second() -> None:
    t0 = _utc(2026, 6, 29, 4, 47, 20)
    starts, states = _window_series(t0, ["pump"])

    out = ck.resample_states_to_seconds(starts, 1.0, states, t0 + timedelta(seconds=10), 1)

    assert out == ["unknown"]


def _session_scada(
    *,
    has_scada: bool = True,
    window_start_utc: list[datetime] | None = None,
    window_s: float = 1.0,
    state: list[str] | None = None,
    power_mw: list[float] | None = None,
    speed_rpm: list[float] | None = None,
    flow_net_m3s: list[float] | None = None,
) -> ck.SessionScada:
    return ck.SessionScada(
        window_start_utc=window_start_utc or [],
        window_s=window_s,
        state=state or [],
        has_scada=has_scada,
        power_mw=power_mw or [],
        speed_rpm=speed_rpm or [],
        flow_net_m3s=flow_net_m3s or [],
    )


def test_build_readout_series_blank_for_a_session_with_no_betriebsdaten() -> None:
    """The 270626-pu_ph_pu_ph_pu_ph-1 placeholder path: `has_scada=False` short-circuits
    to all-`None` series without attempting to resample anything."""
    t0 = _utc(2026, 6, 27, 10, 0, 0)
    sc = _session_scada(has_scada=False)

    power, speed = ck.build_readout_series(sc, t0, 3)

    assert power == [None, None, None]
    assert speed == [None, None, None]


def test_build_readout_series_returns_real_values_when_scada_is_present() -> None:
    t0 = _utc(2026, 6, 29, 4, 47, 20)
    starts, _ = _window_series(t0, ["_", "_", "_"])
    sc = _session_scada(
        window_start_utc=starts, state=["turbine", "turbine", "turbine"],
        power_mw=[10.0, 12.0, 14.0], speed_rpm=[300.0, 305.0, 310.0],
    )

    power, speed = ck.build_readout_series(sc, t0, 3)

    assert power == [10.0, 12.0, 14.0]
    assert speed == [300.0, 305.0, 310.0]


def test_state_color_known_scada_and_detected_states_have_fixed_distinct_colors() -> None:
    names = ("standstill", "turbine", "pump", "phase-shifter", "transition", "unknown")
    colors = [ck.state_color(name) for name in names]

    assert all(c.startswith("#") for c in colors)
    assert len(set(colors)) == len(colors)  # every known state gets its own color


def test_state_color_falls_back_to_the_shared_other_color_for_an_unnamed_cluster() -> None:
    assert ck.state_color("cluster-3") == ck.state_color("cluster-7") == ck._STATE_OTHER_COLOR
    assert ck._STATE_OTHER_COLOR not in {ck.state_color("turbine"), ck.state_color("unknown")}


# ---------------------------------------------------------------------------
# 14. impulse register path (criterion #3): session chunking, cross-mic
# coincidence matching, pair -> Candidate construction, cross-dedup against the
# transient path, raw-peak-log annotation, curated context notes.
# ---------------------------------------------------------------------------


def test_iter_session_chunks_splits_into_exact_multiples() -> None:
    t0 = _utc(2026, 6, 29, 1, 0, 0)
    chunks = list(ck.iter_session_chunks(t0, t0 + timedelta(seconds=10), chunk_s=5.0))

    assert chunks == [
        (t0, t0 + timedelta(seconds=5)),
        (t0 + timedelta(seconds=5), t0 + timedelta(seconds=10)),
    ]


def test_iter_session_chunks_clips_the_last_chunk_short() -> None:
    t0 = _utc(2026, 6, 29, 1, 0, 0)
    chunks = list(ck.iter_session_chunks(t0, t0 + timedelta(seconds=7), chunk_s=5.0))

    assert chunks == [
        (t0, t0 + timedelta(seconds=5)),
        (t0 + timedelta(seconds=5), t0 + timedelta(seconds=7)),
    ]


def test_iter_session_chunks_empty_span_yields_nothing() -> None:
    t0 = _utc(2026, 6, 29, 1, 0, 0)
    assert list(ck.iter_session_chunks(t0, t0, chunk_s=5.0)) == []


def test_iter_session_chunks_rejects_non_positive_chunk_s() -> None:
    t0 = _utc(2026, 6, 29, 1, 0, 0)
    with pytest.raises(ValueError, match="chunk_s"):
        list(ck.iter_session_chunks(t0, t0 + timedelta(seconds=10), chunk_s=0.0))


def test_iter_session_chunks_rejects_end_before_start() -> None:
    t0 = _utc(2026, 6, 29, 1, 0, 0)
    with pytest.raises(ValueError, match="t_end_utc"):
        list(ck.iter_session_chunks(t0, t0 - timedelta(seconds=1), chunk_s=5.0))


# --- match_coincident_peaks -------------------------------------------------


def test_match_coincident_peaks_matches_within_tolerance() -> None:
    t0 = _utc(2026, 6, 29, 2, 0, 15)
    gen = [(t0, 8.8)]
    tur = [(t0 + timedelta(seconds=0.1), 8.2)]

    pairs = ck.match_coincident_peaks(gen, tur, tolerance_s=0.15)

    assert pairs == [(0, 0)]


def test_match_coincident_peaks_outside_tolerance_is_not_matched() -> None:
    t0 = _utc(2026, 6, 29, 2, 0, 15)
    gen = [(t0, 8.8)]
    tur = [(t0 + timedelta(seconds=0.2), 8.2)]

    assert ck.match_coincident_peaks(gen, tur, tolerance_s=0.15) == []


def test_match_coincident_peaks_boundary_exactly_at_tolerance_matches() -> None:
    # Deliberately inclusive (unlike the codebase-wide half-open SPAN convention):
    # "within 0.15s" reads as inclusive for a point-to-point tolerance.
    t0 = _utc(2026, 6, 29, 2, 0, 15)
    gen = [(t0, 8.8)]
    tur = [(t0 + timedelta(seconds=0.15), 8.2)]

    assert ck.match_coincident_peaks(gen, tur, tolerance_s=0.15) == [(0, 0)]


def test_match_coincident_peaks_greedy_closest_pairing_avoids_cross_wiring() -> None:
    t0 = _utc(2026, 6, 29, 2, 0, 15)
    # gen has two peaks; tur has one, closer to gen[1] than gen[0].
    gen = [(t0, 9.0), (t0 + timedelta(seconds=0.1), 7.0)]
    tur = [(t0 + timedelta(seconds=0.09), 8.0)]

    pairs = ck.match_coincident_peaks(gen, tur, tolerance_s=0.15)

    assert pairs == [(1, 0)]  # gen[1] (0.01s gap) wins over gen[0] (0.09s gap)


def test_match_coincident_peaks_each_peak_used_at_most_once() -> None:
    t0 = _utc(2026, 6, 29, 2, 0, 15)
    gen = [(t0, 9.0)]
    tur = [(t0 + timedelta(seconds=0.05), 8.0), (t0 + timedelta(seconds=0.06), 7.0)]

    pairs = ck.match_coincident_peaks(gen, tur, tolerance_s=0.15)

    assert len(pairs) == 1
    assert pairs[0][0] == 0  # the single gen peak is claimed by exactly one tur peak


def test_match_coincident_peaks_empty_inputs_yield_no_pairs() -> None:
    assert ck.match_coincident_peaks([], [], tolerance_s=0.15) == []


# --- build_impulse_pairs ----------------------------------------------------


def test_build_impulse_pairs_coincident_pair_becomes_one_impulse_candidate() -> None:
    t0 = _utc(2026, 6, 29, 2, 0, 15)
    gen = [(t0, 8.8)]
    tur = [(t0 + timedelta(seconds=0.1), 8.2)]

    candidates, records = ck.build_impulse_pairs(
        "290626-tu", gen, tur, tolerance_s=0.15, regime="recalibrate"
    )

    assert len(candidates) == 1
    cand = candidates[0]
    assert cand.klass == "impulse"
    assert cand.session == "290626-tu"
    assert cand.modality == "impulse"
    assert cand.start_utc == t0 + timedelta(seconds=0.05)  # midpoint of the two peak times
    assert 0.0 < cand.min_p < 1.0
    assert len(records) == 2  # one row per raw peak, both streams
    assert all(r.coincident for r in records)


def test_build_impulse_pairs_more_extreme_pair_gets_smaller_min_p() -> None:
    t0 = _utc(2026, 6, 29, 2, 0, 15)
    weak, _ = ck.build_impulse_pairs(
        "s", [(t0, 6.5)], [(t0, 6.2)], tolerance_s=0.15, regime="recalibrate"
    )
    strong, _ = ck.build_impulse_pairs(
        "s", [(t0, 15.0)], [(t0, 14.0)], tolerance_s=0.15, regime="recalibrate"
    )

    assert strong[0].min_p < weak[0].min_p  # min_p = norm.sf(min(gz, tz)) -- decreasing in z


def test_build_impulse_pairs_extreme_z_min_p_is_floored_not_exactly_zero() -> None:
    # norm.sf underflows to a literal 0.0 well before z=100 (real, observed on the
    # strike sessions) -- min_p must stay strictly positive so criterion_
    # sentence's norm.isf round-trip stays finite instead of displaying "z = inf".
    t0 = _utc(2026, 6, 29, 2, 0, 15)
    candidates, _ = ck.build_impulse_pairs(
        "s", [(t0, 100.0)], [(t0, 90.0)], tolerance_s=0.15, regime="recalibrate"
    )

    assert candidates[0].min_p > 0.0
    assert candidates[0].min_p == pytest.approx(ck._MIN_P_FLOOR)


def test_build_impulse_pairs_non_coincident_peaks_produce_no_candidate() -> None:
    t0 = _utc(2026, 6, 29, 2, 0, 15)
    gen = [(t0, 8.8)]
    tur = [(t0 + timedelta(seconds=1.0), 8.2)]  # far outside tolerance

    candidates, records = ck.build_impulse_pairs(
        "s", gen, tur, tolerance_s=0.15, regime="recalibrate"
    )

    assert candidates == []
    assert len(records) == 2
    assert all(not r.coincident for r in records)
    assert all(r.paired_t_utc is None for r in records)


def test_build_impulse_pairs_peak_record_streams_are_tagged_correctly() -> None:
    t0 = _utc(2026, 6, 29, 2, 0, 15)
    _candidates, records = ck.build_impulse_pairs(
        "s", [(t0, 8.8)], [(t0 + timedelta(seconds=0.05), 8.2)],
        tolerance_s=0.15, regime="recalibrate",
    )

    streams = {r.stream for r in records}
    assert streams == {"gen", "tur"}


# --- dedupe_impulse_against_transient ---------------------------------------


def test_dedupe_impulse_against_transient_drops_within_radius_and_records_cross_ref() -> None:
    t0 = _utc(2026, 6, 29, 2, 0, 15)
    transient = _transient(t0, min_p=0.0005)
    imp = _impulse(t0 + timedelta(seconds=1.5), min_p=1e-15)  # 1.5s away, within 2.0s radius

    kept, cross_ref = ck.dedupe_impulse_against_transient([imp], [transient], radius_s=2.0)

    assert kept == []
    assert cross_ref == {imp.start_utc.isoformat(): transient.start_utc.isoformat()}


def test_dedupe_impulse_against_transient_keeps_outside_radius() -> None:
    t0 = _utc(2026, 6, 29, 2, 0, 15)
    transient = _transient(t0, min_p=0.0005)
    imp = _impulse(t0 + timedelta(seconds=2.5), min_p=1e-15)  # outside 2.0s radius

    kept, cross_ref = ck.dedupe_impulse_against_transient([imp], [transient], radius_s=2.0)

    assert kept == [imp]
    assert cross_ref == {}


def test_dedupe_impulse_against_transient_boundary_exactly_at_radius_drops() -> None:
    t0 = _utc(2026, 6, 29, 2, 0, 15)
    transient = _transient(t0, min_p=0.0005)
    imp = _impulse(t0 + timedelta(seconds=2.0), min_p=1e-15)  # exactly at the radius

    kept, _cross_ref = ck.dedupe_impulse_against_transient([imp], [transient], radius_s=2.0)

    assert kept == []


def test_dedupe_impulse_against_transient_no_transients_keeps_everything() -> None:
    t0 = _utc(2026, 6, 29, 2, 0, 15)
    imp = _impulse(t0, min_p=1e-15)

    kept, cross_ref = ck.dedupe_impulse_against_transient([imp], [], radius_s=2.0)

    assert kept == [imp]
    assert cross_ref == {}


# --- annotate_impulse_peak_records ------------------------------------------


def test_annotate_impulse_peak_records_fills_in_dropped_pairs() -> None:
    t0 = _utc(2026, 6, 29, 2, 0, 15)
    gen_t, tur_t = t0, t0 + timedelta(seconds=0.1)
    pair_start_iso = ck._pair_midpoint_utc(gen_t, tur_t).isoformat()
    records = [
        ck.ImpulsePeakRecord(session="s", stream="gen", t_utc=gen_t, z=8.8,
                              coincident=True, paired_t_utc=tur_t),
        ck.ImpulsePeakRecord(session="s", stream="tur", t_utc=tur_t, z=8.2,
                              coincident=True, paired_t_utc=gen_t),
    ]
    cross_ref = {pair_start_iso: "2026-06-29T02:00:14+00:00"}

    out = ck.annotate_impulse_peak_records(records, cross_ref)

    assert all(r.dedup_transient_start_utc == "2026-06-29T02:00:14+00:00" for r in out)


def test_annotate_impulse_peak_records_kept_pair_resolves_to_empty_string() -> None:
    t0 = _utc(2026, 6, 29, 2, 0, 15)
    gen_t, tur_t = t0, t0 + timedelta(seconds=0.1)
    records = [
        ck.ImpulsePeakRecord(session="s", stream="gen", t_utc=gen_t, z=8.8,
                              coincident=True, paired_t_utc=tur_t),
    ]

    out = ck.annotate_impulse_peak_records(records, {})

    assert out[0].dedup_transient_start_utc == ""


def test_annotate_impulse_peak_records_non_coincident_row_untouched() -> None:
    t0 = _utc(2026, 6, 29, 2, 0, 15)
    records = [
        ck.ImpulsePeakRecord(session="s", stream="gen", t_utc=t0, z=6.5,
                              coincident=False, paired_t_utc=None),
    ]

    out = ck.annotate_impulse_peak_records(records, {})

    assert out[0].dedup_transient_start_utc == ""


# --- apply_context_notes -----------------------------------------------------


def test_apply_context_notes_sets_note_for_matching_id() -> None:
    base = _transient(_utc(2026, 6, 29, 2, 0, 15), min_p=0.0001)
    cand = dataclasses.replace(base, candidate_id="290626-tu-10")

    out = ck.apply_context_notes([cand], {"290626-tu-10": "load-swing window"})

    assert out[0].context_note == "load-swing window"


def test_apply_context_notes_no_match_resolves_to_empty_string() -> None:
    base = _transient(_utc(2026, 6, 29, 2, 0, 15), min_p=0.0001)
    cand = dataclasses.replace(base, candidate_id="290626-tu-01")

    out = ck.apply_context_notes([cand], {"290626-tu-10": "load-swing window"})

    assert out[0].context_note == ""


def test_candidate_context_note_defaults_to_empty_string() -> None:
    cand = _transient(_utc(2026, 6, 29, 2, 0, 15), min_p=0.0001)
    assert cand.context_note == ""


# ---------------------------------------------------------------------------
# 15. Real-data validation: `rowii.anomaly.impulse.Z_REGISTER_THRESHOLD` against
# the ST-landmark strikes (module docstrings of both `rowii.anomaly.impulse` and
# `scripts/candidate_kit.py`'s own `select` section restate this claim -- this
# test re-derives it directly from real data on every run).
# ---------------------------------------------------------------------------

_DATA_ROOT = load_config().data_root
_HAS_DATA_ROOT = _DATA_ROOT.is_dir()
_DATA_SKIP_REASON = "ROWII_DATA_ROOT is unset or does not point at an existing directory"


def _st_event_window(event_id: str) -> tuple[datetime, datetime, str]:
    """`(snippet_start_utc, snippet_end_utc, kind)` for ST landmark *event_id*
    ("01"/"08"), derived from the COMMITTED minute-level ground truth
    (`docs/groundtruth/080726_events_st.csv`, read via `make_demo_assets.
    _load_events_csv` -- the SAME parser `annotation_kit.build_session` uses)
    plus `annotation_kit.snippet_window`'s own +/-15s pad -- deliberately NOT the
    generated (gitignored) `results/annotation-kit/080726/events_meta.json`, so
    this test only depends on files that are actually tracked in the repo.
    `event_id` is the event's 1-indexed row position, zero-padded
    (`annotation_kit.build_session`'s own `f"{i:02d}"` convention)."""
    events = mda._load_events_csv(ak._SESSION_CONFIG["st"].events_csv)
    for i, row in enumerate(events.itertuples(index=False), start=1):
        if f"{i:02d}" == event_id:
            start, end = ak.snippet_window(
                row.start_utc.to_pydatetime(), row.end_utc.to_pydatetime()
            )
            return start, end, str(row.kind)
    raise KeyError(f"no ST event {event_id!r}")


def _st_landmark_marks(event_id: str, snippet_start_utc: datetime) -> list[float]:
    """Offsets (seconds since *snippet_start_utc*) of every annotator mark for
    ST landmark *event_id*, from the real seconds-level ground truth
    (`docs/groundtruth/080726_strikes_seconds_st.csv`) -- `format="ISO8601"`
    for the SAME mixed-fractional-second-precision reason `load_strike_
    exclusion_intervals` already documents for this exact file family."""
    strikes_csv = (
        Path(__file__).resolve().parent.parent / "docs" / "groundtruth"
        / "080726_strikes_seconds_st.csv"
    )
    df = pd.read_csv(strikes_csv, comment="#", dtype={"event_id": str})
    rows = df[df["event_id"] == event_id]
    marks = pd.to_datetime(rows["strike_utc"], utc=True, format="ISO8601")
    return sorted((m.to_pydatetime() - snippet_start_utc).total_seconds() for m in marks)


@pytest.mark.data
def test_impulse_search_recovers_st_landmark_strikes() -> None:
    if not _HAS_DATA_ROOT:
        pytest.skip(_DATA_SKIP_REASON)

    index = discover(_DATA_ROOT)
    run = mda._get_run(index, ak._SESSION_CONFIG["st"].run_name)
    offset_ns = run_utc_offset_ns(run)

    for event_id in ("01", "08"):
        snippet_start, snippet_end, kind = _st_event_window(event_id)
        marks = _st_landmark_marks(event_id, snippet_start)
        assert marks, f"no ground-truth marks found for ST event {event_id} ({kind})"

        recovered_by_stream: dict[str, int] = {}
        for stream_key, stream_name in ak._STREAM_NAME_BY_KEY.items():
            files = run.files[stream_name]
            clip = ak.extract_stream_clip(
                files, offset_ns, mda.MONO_CHANNEL_INDEX, snippet_start, snippet_end
            )
            peaks = impulse_mod.detect_impulses(clip.samples, clip.rate_hz)
            recovered = sum(
                1 for m in marks if any(abs(m - p.time_offset_s) <= 0.3 for p in peaks)
            )
            recovered_by_stream[stream_key] = recovered
            # Every genuinely recovered mark must land inside the validated z range
            # this module's own docstring claims (z=7-17) -- not merely above z_min.
            matched_zs = [
                p.z for p in peaks if any(abs(m - p.time_offset_s) <= 0.3 for m in marks)
            ]
            assert all(z >= impulse_mod.Z_REGISTER_THRESHOLD for z in matched_zs)

        # At least one stream must recover every mark at the register threshold --
        # both-mic coincidence (candidate_kit.py's own +/-0.15s tolerance) is a
        # SEPARATE, later check; this test only validates the single-stream search
        # `rowii.anomaly.impulse` itself is responsible for.
        best = max(recovered_by_stream.values())
        assert best == len(marks), (
            f"ST event {event_id} ({kind}): only {best}/{len(marks)} marks recovered "
            f"at z >= {impulse_mod.Z_REGISTER_THRESHOLD:g} on the better stream "
            f"({recovered_by_stream})"
        )
