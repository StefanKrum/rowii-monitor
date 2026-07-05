"""Real-data smoke guards against the June-25 Rodundwerk II delivery.

Every test here is `@pytest.mark.data`: skipped unless `ROWII_DATA_ROOT` (via
`rowii.config.load_config`, the same `.env` + process-env resolution the rest
of the pipeline uses) points at a directory that actually exists (a stale
value or a missing `.env` must skip, not fail with a FileNotFoundError from
deep inside `discover`).

These are integration guards, not unit tests of new logic -- their purpose is
to catch the moment reality (channel names, sample rates, channel counts)
drifts from what the rest of the pipeline hard-codes or assumes. Task 13's
no-legacy-assumptions constraint: every number asserted below is a property of
THIS data, not carried over from an earlier delivery or exploratory deck.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from rowii.config import load_config
from rowii.io.dataset import discover
from rowii.io.gantner import read_header
from rowii.scada.labels import GT_CHANNELS

_DATA_ROOT = load_config().data_root
_HAS_DATA_ROOT = _DATA_ROOT.is_dir()

pytestmark = pytest.mark.data

skip_reason = (
    "ROWII_DATA_ROOT is unset or does not point at an existing directory"
)


@pytest.fixture(scope="module")
def data_root() -> Path:
    if not _HAS_DATA_ROOT:
        pytest.skip(skip_reason)
    return _DATA_ROOT


def test_discover_finds_tu_run_with_four_streams_of_twelve_files_each(data_root: Path) -> None:
    index = discover(data_root)
    tu_runs = [r for r in index.runs if r.name == "tu"]
    assert len(tu_runs) == 1, f"expected exactly one 'tu' run, found {len(tu_runs)}"

    run = tu_runs[0]
    expected_streams = {
        "RAWGeneratorMic__0",
        "RAWTurbineMic__1",
        "RAWGeneratorVib__2",
        "RAWTurbineVib__3",
    }
    assert set(run.files.keys()) == expected_streams
    for stream in expected_streams:
        n_files = len(run.files[stream])
        assert n_files == 12, f"stream {stream}: expected 12 files, found {n_files}"


def test_discover_finds_at_least_one_pu_run(data_root: Path) -> None:
    index = discover(data_root)
    pu_runs = [r for r in index.runs if r.name.startswith("pu")]
    assert len(pu_runs) >= 1, "expected at least one 'pu*' run in the discovered index"


def test_tu_mic_file_header_reports_plausible_audio_rate_and_channels(data_root: Path) -> None:
    index = discover(data_root)
    run = next(r for r in index.runs if r.name == "tu")
    first_mic_file = sorted(
        run.files["RAWGeneratorMic__0"], key=lambda f: f.start_utc_hint
    )[0]

    header = read_header(first_mic_file.path)

    assert 45_000 <= header.sample_rate_hz <= 55_000, (
        f"mic sample rate {header.sample_rate_hz} Hz outside the expected ~50 kHz range"
    )
    assert len(header.channel_names) >= 4, (
        f"mic file has {len(header.channel_names)} channels, expected >= 4"
    )


def test_tu_vib_file_header_reports_plausible_vibration_rate_and_six_channels(
    data_root: Path,
) -> None:
    index = discover(data_root)
    run = next(r for r in index.runs if r.name == "tu")
    first_vib_file = sorted(
        run.files["RAWGeneratorVib__2"], key=lambda f: f.start_utc_hint
    )[0]

    header = read_header(first_vib_file.path)

    assert 9_000 <= header.sample_rate_hz <= 11_000, (
        f"vib sample rate {header.sample_rate_hz} Hz outside the expected ~10 kHz range"
    )
    assert len(header.channel_names) == 6, (
        f"vib file has {len(header.channel_names)} channels, expected exactly 6"
    )


def test_betriebsdaten_file_contains_all_gt_channel_names(data_root: Path) -> None:
    index = discover(data_root)
    assert index.betriebsdaten, "expected at least one discovered Betriebsdaten file"

    target = next(
        (p for p in index.betriebsdaten if p.name == "2026-06-25_05-00-00.dat"),
        index.betriebsdaten[0],
    )

    header = read_header(target)

    for key, channel_name in GT_CHANNELS.items():
        assert channel_name in header.channel_names, (
            f"GT_CHANNELS[{key!r}] = {channel_name!r} not found in "
            f"{target.name}; available channels: {header.channel_names}"
        )
