import logging
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from rowii.io.dataset import (
    BurstFile,
    Run,
    _parse_burst_filename,
    betriebsdaten_utc_offset_ns,
    discover,
    run_utc_offset_ns,
)
from tests.fixtures.gantner_builder import build_gantner_file


def _touch(path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch()


def _make_tu_pu_tree(root):
    meas = root / "20260626 Messung"
    tu = meas / "TU"
    pu = meas / "PU"
    # TU: 2 segments, ~12 min apart, 2 streams -> stays one run "tu"
    for t in ("06-03-00", "06-15-00"):
        for stream in ("RAWGeneratorMic__0", "RAWTurbineMic__1"):
            _touch(tu / f"{stream}_2026-06-25_{t}_000000.dat")
    # PU: morning group (09:08, 09:20) + afternoon group (13:44...14:32, ~12 min
    # cadence like TU) -- gap between 09:20 and 13:44 is > 15 min and splits the
    # folder; the ~12-min gaps within each group stay under threshold.
    for t in ("09-08-00", "09-20-00", "13-44-00", "13-56-00", "14-08-00", "14-20-00", "14-32-00"):
        for stream in ("RAWGeneratorMic__0", "RAWTurbineMic__1"):
            _touch(pu / f"{stream}_2026-06-25_{t}_000000.dat")
    return meas


def test_tu_folder_with_gaps_under_15min_stays_one_run(tmp_path):
    _make_tu_pu_tree(tmp_path)
    index = discover(tmp_path)
    tu_runs = [r for r in index.runs if r.name == "tu"]
    assert len(tu_runs) == 1
    run = tu_runs[0]
    assert set(run.files.keys()) == {"RAWGeneratorMic__0", "RAWTurbineMic__1"}
    assert len(run.files["RAWGeneratorMic__0"]) == 2
    assert len(run.files["RAWTurbineMic__1"]) == 2


def test_pu_folder_splits_into_morning_and_afternoon_on_gap(tmp_path):
    _make_tu_pu_tree(tmp_path)
    index = discover(tmp_path)
    names = sorted(r.name for r in index.runs)
    assert names == ["pu-afternoon", "pu-morning", "tu"]

    morning = next(r for r in index.runs if r.name == "pu-morning")
    afternoon = next(r for r in index.runs if r.name == "pu-afternoon")
    assert len(morning.files["RAWGeneratorMic__0"]) == 2
    assert len(morning.files["RAWTurbineMic__1"]) == 2
    assert len(afternoon.files["RAWGeneratorMic__0"]) == 5
    assert len(afternoon.files["RAWTurbineMic__1"]) == 5

    morning_names = {p.path.name for p in morning.files["RAWGeneratorMic__0"]}
    assert morning_names == {
        "RAWGeneratorMic__0_2026-06-25_09-08-00_000000.dat",
        "RAWGeneratorMic__0_2026-06-25_09-20-00_000000.dat",
    }
    afternoon_names = {p.path.name for p in afternoon.files["RAWGeneratorMic__0"]}
    assert afternoon_names == {
        "RAWGeneratorMic__0_2026-06-25_13-44-00_000000.dat",
        "RAWGeneratorMic__0_2026-06-25_13-56-00_000000.dat",
        "RAWGeneratorMic__0_2026-06-25_14-08-00_000000.dat",
        "RAWGeneratorMic__0_2026-06-25_14-20-00_000000.dat",
        "RAWGeneratorMic__0_2026-06-25_14-32-00_000000.dat",
    }


def test_files_within_a_stream_are_time_sorted(tmp_path):
    meas = tmp_path / "20260626 Messung"
    tu = meas / "TU"
    # write the later file first on disk to prove sorting isn't accidental filesystem order
    _touch(tu / "RAWGeneratorMic__0_2026-06-25_06-15-00_000000.dat")
    _touch(tu / "RAWGeneratorMic__0_2026-06-25_06-03-00_000000.dat")

    index = discover(tmp_path)
    run = next(r for r in index.runs if r.name == "tu")
    files = run.files["RAWGeneratorMic__0"]
    assert [f.path.name for f in files] == [
        "RAWGeneratorMic__0_2026-06-25_06-03-00_000000.dat",
        "RAWGeneratorMic__0_2026-06-25_06-15-00_000000.dat",
    ]
    assert files[0].start_utc_hint < files[1].start_utc_hint


def test_start_utc_hint_converts_local_cest_june_to_utc():
    # unit-tests the filename parser directly (rather than only through discover())
    # to pin the exact CEST -> UTC offset independent of folder-grouping behavior.
    stream, start_utc = _parse_burst_filename(
        Path("RAWGeneratorMic__0_2026-06-25_06-03-00_000000.dat")
    )
    assert stream == "RAWGeneratorMic__0"
    # 2026-06-25 06:03:00 local (Europe/Vienna, CEST = UTC+2) -> 04:03:00 UTC
    assert start_utc == datetime(2026, 6, 25, 4, 3, 0, tzinfo=UTC)


def test_betriebsdaten_files_are_time_sorted(tmp_path):
    meas = tmp_path / "20260626 Messung"
    bd = meas / "Betriebsdaten"
    _touch(bd / "2026-06-25_10-00-00.dat")
    _touch(bd / "2026-06-25_08-00-00.dat")
    _touch(bd / "2026-06-25_09-00-00.dat")

    index = discover(tmp_path)
    assert [p.name for p in index.betriebsdaten] == [
        "2026-06-25_08-00-00.dat",
        "2026-06-25_09-00-00.dat",
        "2026-06-25_10-00-00.dat",
    ]


def test_betriebsdaten_duplicate_hour_prefers_larger_file_and_logs_warning(
    tmp_path, caplog
):
    meas = tmp_path / "20260626 Messung"
    bd = meas / "Betriebsdaten"
    bd.mkdir(parents=True)
    small = bd / "2026-06-25_08-00-00.dat"
    large = bd / "2026-06-25_08-00-00_1.dat"
    small.write_bytes(b"x" * 10)
    large.write_bytes(b"x" * 20)

    with caplog.at_level(logging.WARNING):
        index = discover(tmp_path)

    assert len(index.betriebsdaten) == 1
    assert index.betriebsdaten[0] == large
    assert any(
        "2026-06-25" in record.message and "08" in record.message
        for record in caplog.records
        if record.levelno == logging.WARNING
    )


def test_betriebsdaten_equal_size_tie_break_prefers_lexicographically_smaller_name(
    tmp_path, caplog, monkeypatch
):
    meas = tmp_path / "20260626 Messung"
    bd = meas / "Betriebsdaten"
    bd.mkdir(parents=True)
    winner = bd / "2026-06-25_08-00-00.dat"
    loser = bd / "2026-06-25_08-00-00_1.dat"
    # identical byte counts -> size comparison alone can't break the tie
    winner.write_bytes(b"x" * 16)
    loser.write_bytes(b"x" * 16)

    # Directory iteration order is arbitrary (Path.iterdir gives no ordering
    # guarantee). Force the "wrong" arrival order here -- loser first -- so
    # the assertion below can only pass if the tie-break is a deterministic
    # name comparison, not an accident of this filesystem's iterdir() order.
    real_iterdir = Path.iterdir

    def reversed_iterdir(self: Path):
        entries = list(real_iterdir(self))
        return iter(sorted(entries, key=lambda p: p.name, reverse=True))

    monkeypatch.setattr(Path, "iterdir", reversed_iterdir)

    with caplog.at_level(logging.WARNING):
        index = discover(tmp_path)

    assert len(index.betriebsdaten) == 1
    assert index.betriebsdaten[0] == winner
    assert any(
        "2026-06-25" in record.message and "08" in record.message
        for record in caplog.records
        if record.levelno == logging.WARNING
    )


def test_masked_per_stream_gap_within_a_run_logs_warning(tmp_path, caplog):
    # mic files every 10 min (06:00, 06:10, 06:20, 06:30) pool with vib files
    # only at the endpoints (06:00, 06:30) -- the pooled sequence never has a
    # gap > 15 min (mic fills every slot), so this stays one run, but the vib
    # stream alone has a masked 30-min gap between its two files.
    meas = tmp_path / "20260626 Messung"
    tu = meas / "TU"
    for t in ("06-00-00", "06-10-00", "06-20-00", "06-30-00"):
        _touch(tu / f"RAWGeneratorMic__0_2026-06-25_{t}_000000.dat")
    for t in ("06-00-00", "06-30-00"):
        _touch(tu / f"RAWGeneratorVib__2_2026-06-25_{t}_000000.dat")

    with caplog.at_level(logging.WARNING):
        index = discover(tmp_path)

    tu_runs = [r for r in index.runs if r.name == "tu"]
    assert len(tu_runs) == 1
    assert len(tu_runs[0].files["RAWGeneratorVib__2"]) == 2

    warnings = [r.message for r in caplog.records if r.levelno == logging.WARNING]
    assert any("RAWGeneratorVib__2" in msg and "tu" in msg for msg in warnings)


def test_burst_filename_with_unparseable_date_is_excluded_and_logs_warning(
    tmp_path, caplog
):
    meas = tmp_path / "20260626 Messung"
    tu = meas / "TU"
    # matches the burst filename *shape* (regex doesn't validate month range)
    # but month=13 fails strptime -- must be excluded, not silently dropped.
    bad_name = "RAWGeneratorMic__0_2026-13-05_06-00-00_000000.dat"
    _touch(tu / bad_name)
    _touch(tu / "RAWGeneratorMic__0_2026-06-25_06-00-00_000000.dat")

    with caplog.at_level(logging.WARNING):
        index = discover(tmp_path)

    tu_runs = [r for r in index.runs if r.name == "tu"]
    assert len(tu_runs) == 1
    files = tu_runs[0].files["RAWGeneratorMic__0"]
    assert len(files) == 1
    assert files[0].path.name == "RAWGeneratorMic__0_2026-06-25_06-00-00_000000.dat"

    warnings = [r.message for r in caplog.records if r.levelno == logging.WARNING]
    assert any(bad_name in msg for msg in warnings)


# ---------------------------------------------------------------------------
# Session discovery generalization (no TU/PU whitelist) + multi-day parent root
# ---------------------------------------------------------------------------


def test_session_folder_not_named_tu_or_pu_is_still_discovered(tmp_path):
    # Drops the hardcoded ("TU", "PU") folder-name whitelist: a session is any
    # DIRECT subfolder of a "* Messung" dir that contains burst-pattern files,
    # named however the operator chose (real example: "TU_PH_TU").
    meas = tmp_path / "20260701 Messung"
    session = meas / "TU_PH_TU"
    for t in ("15-45-00", "15-57-00"):
        _touch(session / f"RAWGeneratorMic__0_2026-07-01_{t}_000000.dat")

    index = discover(tmp_path)

    assert [r.name for r in index.runs] == ["tu_ph_tu"]
    assert len(index.runs[0].files["RAWGeneratorMic__0"]) == 2


def test_single_tree_root_keeps_legacy_unprefixed_run_names(tmp_path):
    # Backward compat: a data_root that IS a day root (contains "* Messung"
    # directly, no illwerke-<day> parent layer) must keep producing exactly the
    # unprefixed run names the pre-addendum pipeline/tests already depend on.
    _make_tu_pu_tree(tmp_path)

    index = discover(tmp_path)

    assert sorted(r.name for r in index.runs) == ["pu-afternoon", "pu-morning", "tu"]


def _make_parent_root_with_two_days(root):
    day1 = root / "illwerke-010726" / "20260701 Messung"
    day2 = root / "illwerke-270626" / "20260627 Messung"
    for t in ("15-45-00", "15-57-00"):
        _touch(day1 / "TU_PH_TU" / f"RAWGeneratorMic__0_2026-07-01_{t}_000000.dat")
    for t in ("06-41-00", "06-53-00"):
        _touch(
            day2
            / "PU_PH_PU_PH_PU_PH"
            / f"RAWGeneratorMic__0_2026-06-27_{t}_000000.dat"
        )
    _touch(day1 / "Betriebsdaten" / "2026-07-01_15-00-00.dat")
    _touch(day2 / "Betriebsdaten" / "2026-06-27_06-00-00.dat")
    return root


def test_parent_root_with_multiple_day_trees_prefixes_run_names_with_dayid(tmp_path):
    # A PARENT root containing multiple illwerke-<dayid>/<...Messung> trees (the new
    # ROWII_DATA_ROOT layout) must discover every day's sessions, each prefixed with
    # the 6-digit dayid token taken from its own "illwerke-<dayid>" directory name.
    _make_parent_root_with_two_days(tmp_path)

    index = discover(tmp_path)

    assert sorted(r.name for r in index.runs) == [
        "010726-tu_ph_tu",
        "270626-pu_ph_pu_ph_pu_ph",
    ]


def test_parent_root_betriebsdaten_is_scoped_per_day_tree(tmp_path):
    # Spec requirement: "a run only sees its own day's Betriebsdaten" -- each day
    # tree's Betriebsdaten files must be retrievable independently, not just pooled
    # into one flat list a caller could accidentally apply to the wrong day's run.
    _make_parent_root_with_two_days(tmp_path)

    index = discover(tmp_path)

    day1_root = tmp_path / "illwerke-010726"
    day2_root = tmp_path / "illwerke-270626"
    assert set(index.betriebsdaten_by_day.keys()) == {day1_root, day2_root}
    assert [p.name for p in index.betriebsdaten_by_day[day1_root]] == [
        "2026-07-01_15-00-00.dat"
    ]
    assert [p.name for p in index.betriebsdaten_by_day[day2_root]] == [
        "2026-06-27_06-00-00.dat"
    ]

    run_010726 = next(r for r in index.runs if r.name == "010726-tu_ph_tu")
    run_270626 = next(r for r in index.runs if r.name == "270626-pu_ph_pu_ph_pu_ph")
    assert run_010726.day_root == day1_root
    assert run_270626.day_root == day2_root

    # The flat, pooled field stays available (pre-addendum callers/tests keep
    # working unchanged) -- it is simply the union of every day tree's files.
    assert len(index.betriebsdaten) == 2


def test_single_tree_root_day_root_equals_data_root_itself(tmp_path):
    # In legacy single-tree mode there is no illwerke-<day> layer to read a dayid
    # from -- day_root must be data_root itself, and betriebsdaten_by_day must key
    # on exactly that path (so per-day-tree lookups work uniformly in both modes).
    _make_tu_pu_tree(tmp_path)
    _touch(tmp_path / "20260626 Messung" / "Betriebsdaten" / "2026-06-25_08-00-00.dat")

    index = discover(tmp_path)

    assert set(index.betriebsdaten_by_day.keys()) == {tmp_path}
    for run in index.runs:
        assert run.day_root == tmp_path


def test_parent_root_with_no_scada_day_has_empty_betriebsdaten_for_that_day(tmp_path):
    # 27.06 (real delivery): PU_PH_PU_PH_PU_PH session, NO Betriebsdaten folder at
    # all -- that day tree must simply have no entry (or an empty list), never
    # borrow another day's Betriebsdaten files by falling through to the pooled list.
    day_root = tmp_path / "illwerke-270626"
    meas = day_root / "20260627 Messung"
    for t in ("06-41-00", "06-53-00"):
        _touch(meas / "PU_PH_PU_PH_PU_PH" / f"RAWGeneratorMic__0_2026-06-27_{t}_000000.dat")

    index = discover(tmp_path)

    assert index.betriebsdaten_by_day.get(day_root, []) == []
    assert index.betriebsdaten == []


# ---------------------------------------------------------------------------
# run_utc_offset_ns / betriebsdaten_utc_offset_ns (DAQ epoch-2000 clock
# quirk -- every Gantner file's binary frame timestamps count seconds since
# 2000-01-01 LOCAL time but are labelled as Unix (1970) nanoseconds; both functions
# derive the constant per-file-set offset that maps the raw axis onto true UTC)
# ---------------------------------------------------------------------------


def _naive_unix_decode_ns(dt: datetime) -> int:
    """Nanoseconds since the Unix epoch that *dt* decodes to when its own digits are
    read naively AS IF it already were a Unix timestamp -- i.e. the raw on-disk
    `header.t0_ns` a quirky DAQ file would carry for a recording whose true local
    wall-clock reads *dt*'s own digits (see `rowii.io.dataset.run_utc_offset_ns`'s
    docstring for the full quirk model). Whole-second *dt* only (no microseconds) so
    this stays exact integer arithmetic, with no float-precision noise to reason
    about in the assertions below -- the function under test is robust to that noise
    anyway (its final rounding-to-the-hour step swallows anything under 30 minutes),
    but keeping the fixtures exact makes the expected constants easy to verify by
    hand.
    """
    return int((dt - datetime(1970, 1, 1, tzinfo=UTC)).total_seconds()) * 10**9


def _burst_run(
    tmp_path: Path, files_spec: list[tuple[str, int, datetime]], name: str = "quirky-run"
) -> Run:
    """A `Run` with one minimal single-channel Gantner file per `(stream, raw_t0_ns,
    start_utc_hint)` triple in *files_spec* -- channel content is irrelevant to
    `run_utc_offset_ns`, which only ever reads `header.t0_ns` (`read_header`, cheap)."""
    files: dict[str, list[BurstFile]] = {}
    for i, (stream, raw_t0_ns, hint) in enumerate(files_spec):
        path = tmp_path / f"{stream}_{i}.dat"
        build_gantner_file(
            path, ["ch0"], np.zeros((1, 1), dtype=np.float32), t0_ns=raw_t0_ns, rate_hz=1.0
        )
        burst = BurstFile(path=path, stream=stream, start_utc_hint=hint)
        files.setdefault(stream, []).append(burst)
    return Run(name=name, files=files, day_root=tmp_path)


# CEST (UTC+2) worked example, verified against the real June-2026 delivery (module
# docstring above): raw header.t0_ns decodes (naively, as Unix ns)
# to 1996-06-27T06:41:03Z; the true instant is 2026-06-27T04:41:03Z (local
# Europe/Vienna wall-clock digits 06:41:03 on 2026-06-27, CEST = UTC+2). Offset =
# 946_684_800 s epoch-2000 shift - 7_200 s CEST = 946_677_600 s -- exactly a whole
# number of hours (262,966), so the nearest-hour rounding step is a no-op here.
_CEST_RAW_DT = datetime(1996, 6, 27, 6, 41, 3, tzinfo=UTC)
_CEST_HINT = datetime(2026, 6, 27, 4, 41, 3, tzinfo=UTC)
_CEST_OFFSET_NS = 946_677_600 * 10**9


def test_run_utc_offset_ns_quirky_clock_derives_documented_offset(tmp_path):
    raw_t0_ns = _naive_unix_decode_ns(_CEST_RAW_DT)
    run = _burst_run(
        tmp_path,
        [
            ("RAWGeneratorMic__0", raw_t0_ns, _CEST_HINT),
            ("RAWTurbineMic__1", raw_t0_ns, _CEST_HINT),
        ],
    )

    assert run_utc_offset_ns(run) == _CEST_OFFSET_NS


def test_run_utc_offset_ns_correct_clock_returns_zero(tmp_path):
    # A future/already-correct-clock dataset: header.t0_ns is a genuine UTC
    # nanosecond value a few seconds off its own filename hint (ordinary DAQ
    # jitter), well under the 1-hour plausibility gate -- the offset must come out
    # as exactly 0, never some tiny rounded-to-an-hour artifact invented for data
    # that was never quirky to begin with.
    hint = datetime(2030, 3, 1, 12, 0, 0, tzinfo=UTC)
    raw_t0_ns = _naive_unix_decode_ns(hint) + 5 * 10**9  # 5 s of ordinary DAQ jitter
    run = _burst_run(
        tmp_path,
        [
            ("RAWGeneratorMic__0", raw_t0_ns, hint),
            ("RAWTurbineMic__1", raw_t0_ns, hint),
        ],
    )

    assert run_utc_offset_ns(run) == 0


def test_run_utc_offset_ns_warns_on_file_deviating_from_rounded_offset(tmp_path, caplog):
    # Two files exactly matching the CEST worked example (median stays pinned at
    # 946_677_600 s, 2 out of 3 entries agree) plus a THIRD file whose raw t0_ns is
    # shifted an extra 5 s off that same hour -- the median is untouched, but the
    # deviant file's OWN raw offset differs from the rounded value by > 2 s and must
    # be named, alongside the run, in a WARNING.
    clean_raw_t0_ns = _naive_unix_decode_ns(_CEST_RAW_DT)
    noisy_raw_t0_ns = clean_raw_t0_ns - 5 * 10**9
    run = _burst_run(
        tmp_path,
        [
            ("RAWGeneratorMic__0", clean_raw_t0_ns, _CEST_HINT),
            ("RAWGeneratorMic__0", clean_raw_t0_ns, _CEST_HINT),
            ("RAWTurbineMic__1", noisy_raw_t0_ns, _CEST_HINT),
        ],
        name="noisy-run",
    )

    with caplog.at_level(logging.WARNING):
        offset = run_utc_offset_ns(run)

    assert offset == _CEST_OFFSET_NS
    warnings = [r.message for r in caplog.records if r.levelno == logging.WARNING]
    assert any(
        "noisy-run" in w and "RAWTurbineMic__1_2.dat" in w for w in warnings
    ), warnings


def test_run_utc_offset_ns_empty_run_returns_zero(tmp_path):
    # No files at all (degenerate, but must not crash) -- nothing to derive from.
    run = Run(name="empty-run", files={}, day_root=tmp_path)

    assert run_utc_offset_ns(run) == 0


def test_betriebsdaten_utc_offset_ns_quirky_clock_derives_documented_offset(tmp_path):
    # Same CEST worked example, but via the Betriebsdaten filename convention
    # (`_BETRIEBSDATEN_RE`: "<date>_<hour>-00-00[_<n>].dat", local time, whole hours
    # only) rather than a pre-parsed `BurstFile.start_utc_hint` -- the function must
    # derive its own hint from the filename before deriving the offset.
    raw_t0_ns = _naive_unix_decode_ns(datetime(1996, 6, 27, 6, 0, 0, tzinfo=UTC))
    p1 = tmp_path / "2026-06-27_06-00-00.dat"
    p2 = tmp_path / "2026-06-27_07-00-00.dat"
    build_gantner_file(
        p1, ["ch0"], np.zeros((1, 1), dtype=np.float32), t0_ns=raw_t0_ns, rate_hz=1.0
    )
    build_gantner_file(
        p2, ["ch0"], np.zeros((1, 1), dtype=np.float32),
        t0_ns=raw_t0_ns + round(3600 * 1e9), rate_hz=1.0,
    )

    assert betriebsdaten_utc_offset_ns([p1, p2]) == _CEST_OFFSET_NS


def test_betriebsdaten_utc_offset_ns_files_not_matching_pattern_return_zero(tmp_path):
    # Existing unit-test fixtures across the codebase (tests/test_gt_labels.py etc.)
    # build Betriebsdaten-shaped `.dat` files under arbitrary names ("bd.dat",
    # "h1.dat", ...) that were never meant to model the DAQ-clock quirk at all --
    # `load_scada_window_means` must stay a safe no-op (offset 0) for those, not
    # crash or invent a bogus derived offset from an unparseable filename.
    p = tmp_path / "not-a-betriebsdaten-filename.dat"
    build_gantner_file(p, ["ch0"], np.zeros((1, 1), dtype=np.float32), t0_ns=0, rate_hz=1.0)

    assert betriebsdaten_utc_offset_ns([p]) == 0


def test_betriebsdaten_utc_offset_ns_empty_list_returns_zero() -> None:
    assert betriebsdaten_utc_offset_ns([]) == 0
