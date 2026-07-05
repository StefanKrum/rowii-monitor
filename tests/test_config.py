from pathlib import Path

from rowii.config import load_config


def test_defaults_without_env() -> None:
    cfg = load_config(env={})
    assert cfg.window.window_s == 1.0
    assert cfg.detect.n_states == 4
    assert cfg.detect.self_transition == 0.98
    assert cfg.beats_checkpoint is None


def test_env_overrides() -> None:
    cfg = load_config(env={"ROWII_DATA_ROOT": "/tmp/x", "ROWII_BEATS_CHECKPOINT": "/tmp/b.pt"})
    assert cfg.data_root == Path("/tmp/x")
    assert cfg.beats_checkpoint == Path("/tmp/b.pt")
