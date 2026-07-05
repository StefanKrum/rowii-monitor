import numpy as np
import pytest

from rowii.io.gantner import GantnerFormatError, read_gantner, read_header
from tests.fixtures.gantner_builder import build_gantner_file


def test_roundtrip_reads_names_units_rate_and_data(tmp_path) -> None:
    rng = np.random.default_rng(0)
    data = rng.normal(size=(500, 3)).astype(np.float32)
    p = build_gantner_file(tmp_path / "t.dat", ["ChA", "ChB", "ChC"], data,
                           rate_hz=100.0, units=["Pa", "", "m/s2"])
    f = read_gantner(p)
    assert f.header.channel_names == ["ChA", "ChB", "ChC"]
    assert f.header.channel_units == ["Pa", "", "m/s2"]
    assert f.header.n_frames == 500
    assert abs(f.header.sample_rate_hz - 100.0) < 0.5
    np.testing.assert_allclose(f.data, data, rtol=1e-6)
    assert f.timestamps_ns[0] == f.header.t0_ns


def test_partial_tail_frame_is_dropped(tmp_path) -> None:
    data = np.zeros((10, 2), dtype=np.float32)
    p = build_gantner_file(tmp_path / "t.dat", ["A", "B"], data)
    raw = p.read_bytes()
    p.write_bytes(raw[:-5])                    # cut into the last frame
    assert read_gantner(p).header.n_frames == 9


def test_missing_magic_raises(tmp_path) -> None:
    p = tmp_path / "bad.dat"
    p.write_bytes(b"\x00\x00not a gantner file" * 10)
    with pytest.raises(GantnerFormatError, match="magic"):
        read_gantner(p)


def test_missing_padding_raises(tmp_path) -> None:
    data = np.zeros((5, 1), dtype=np.float32)
    p = build_gantner_file(tmp_path / "t.dat", ["A"], data, corrupt_padding=True)
    with pytest.raises(GantnerFormatError, match="padding"):
        read_gantner(p)


def test_read_header_is_cheap_and_consistent(tmp_path) -> None:
    data = np.zeros((1000, 2), dtype=np.float32)
    p = build_gantner_file(tmp_path / "t.dat", ["A", "B"], data, rate_hz=50.0)
    h = read_header(p)
    assert h.n_frames == 1000 and abs(h.sample_rate_hz - 50.0) < 0.5
