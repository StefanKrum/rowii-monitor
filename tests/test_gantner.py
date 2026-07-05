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


def test_channel_name_of_length_36_roundtrips(tmp_path) -> None:
    # 36 is the length of every uuid.uuid4() string, so its own 2-byte little-endian
    # length prefix has low byte 0x24 ('$') — printable ASCII. A name of the same length
    # reproduces the same length-prefix/token collision for a *name* token, not just a UUID.
    name = "N" * 36
    data = np.zeros((5, 1), dtype=np.float32)
    p = build_gantner_file(tmp_path / "t.dat", [name], data, units=["Pa"])
    f = read_gantner(p)
    assert f.header.channel_names == [name]
    assert f.header.channel_units == ["Pa"]


def test_channel_name_of_length_50_roundtrips(tmp_path) -> None:
    # length 50 -> low byte 0x32 ('2'), a different point in the printable-ASCII collision
    # range (32-126) than the UUID-driven 36-char case.
    name = "N" * 50
    data = np.zeros((5, 1), dtype=np.float32)
    p = build_gantner_file(tmp_path / "t.dat", [name], data, units=["Pa"])
    f = read_gantner(p)
    assert f.header.channel_names == [name]
    assert f.header.channel_units == ["Pa"]


def test_channel_unit_of_length_40_roundtrips(tmp_path) -> None:
    # length 40 -> low byte 0x28 ('('), exercising the same collision class on the unit
    # token specifically, with a short (unaffected) name alongside it.
    unit = "U" * 40
    data = np.zeros((5, 1), dtype=np.float32)
    p = build_gantner_file(tmp_path / "t.dat", ["ChA"], data, units=[unit])
    f = read_gantner(p)
    assert f.header.channel_names == ["ChA"]
    assert f.header.channel_units == [unit]


def test_filler_bytes_between_tokens_are_skipped(tmp_path) -> None:
    # Simulates unknown filler bytes between name/unit tokens and the UUID token of each
    # channel (observed as a ~16-byte per-channel descriptor blob during the V2.18 reverse
    # engineering). A validating scan-tokenizer must reject-and-advance through this filler
    # rather than mis-parsing bytes inside it as spurious tokens.
    data = np.random.default_rng(1).normal(size=(20, 2)).astype(np.float32)
    p = build_gantner_file(tmp_path / "t.dat", ["ChA", "ChB"], data,
                           units=["Pa", "m/s2"], filler_bytes=16)
    f = read_gantner(p)
    assert f.header.channel_names == ["ChA", "ChB"]
    assert f.header.channel_units == ["Pa", "m/s2"]
    assert f.header.n_frames == 20
    np.testing.assert_allclose(f.data, data, rtol=1e-6)


def test_real_channel_record_gap_pattern_does_not_shadow_the_uuid_token(tmp_path) -> None:
    # Task 13 real-data finding (Betriebsdaten 2026-06-25_05-00-00.dat and TU vib files):
    # every real channel record has FIXED gap bytes between name-end and unit-start
    # (`00 00 08 00 08 00 02 00`) and between unit-end and the uuid's own length prefix
    # (`2b 00 02 00`). The second gap's tail happens to overlap the uuid length-prefix's own
    # bytes in a way that, read one byte later than intended, forms an accidental but
    # STRUCTURALLY VALID short token (length=2, one printable payload byte, NUL terminator) --
    # in the real file this shadows the genuine 36-byte uuid token entirely, dropping the
    # channel. A validating scan-tokenizer must prefer the longer, later-starting genuine
    # token over the shorter one that appears to validate first when scanning byte by byte.
    real_gap_pattern = bytes.fromhex("00000800080002002b00020002")
    p = build_gantner_file(
        tmp_path / "t.dat", ["ChA", "ChB"], np.zeros((5, 2), dtype=np.float32),
        units=["Pa", "m/s2"], raw_filler=real_gap_pattern,
    )
    f = read_gantner(p)
    assert f.header.channel_names == ["ChA", "ChB"]
    assert f.header.channel_units == ["Pa", "m/s2"]
