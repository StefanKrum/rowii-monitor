import logging
from datetime import UTC, datetime
from pathlib import Path

from rowii.io.dataset import _parse_burst_filename, discover


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
