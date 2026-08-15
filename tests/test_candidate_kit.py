"""Tests for `scripts/candidate_kit.py`'s PURE logic: sustained-episode grouping from
fusion `alarms.parquet` rows, transient-window extraction from audio-beats rows,
extremity-ordered dedup/overlap suppression (one shared greedy primitive backs both),
the 080726 strike-exclusion mask (padded-interval overlap against BOTH the seconds-level
`docs/groundtruth/080726_strikes_seconds_*.csv` and the coarser minute-level
`080726_events_*.csv` -- the latter covers events with no per-strike rows yet, e.g. the
real PU event_id 07/13 gap), the build-side asset-window sizing rule, and `compile`'s
validation (known candidate ids, fixed assessment vocabulary, provenance header).

Import convention mirrors `tests/test_annotation_kit.py`: `scripts/` is not a package,
so the module under test is imported directly by inserting `scripts/` onto `sys.path`.
Real parquet reads against `results/step2/once-calibrated/`, real WAV/PNG rendering and
`index.html` assembly are exercised by actually running the CLI against real data
instead, not by a test here (same split `test_annotation_kit.py` documents).
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

import candidate_kit as ck  # noqa: E402


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
