"""Tests for `scripts/scarcity_detection.py` (package-6 pillar-3, plan Task 5):
synthetic labeled wav trees only (mirroring `tests/test_corpora_labeled.py`'s
builder conventions -- no real MIMII anywhere, this repo's downloads-never-run-
in-tests rule), driven through the handcrafted representation plus the torch
representations' skip path (checkpoint envs unset -- `benchmark_inference`'s
established semantics).

The synthetic corpus makes abnormal clips LOUD high-variance broadband noise
against tonal normals: the labeled iterator per-window standardizes every
window (amendment A1.5), which erases a PURE level difference by construction,
so separability must come from spectral shape -- a 200 Hz tone vs. broadband
noise separates cleanly in the handcrafted spectral features (centroid,
rolloff, band energies) regardless of gain.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.io.wavfile import write as write_wav

SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import scarcity_detection  # noqa: E402

_RATE_HZ = 16_000

# Expected CSV schema, hardcoded (NOT read off the module constant) so a drive-by
# column rename/reorder fails loudly here -- the binding contract of plan Task 5.
_EXPECTED_COLUMNS = [
    "representation", "machine_id", "fraction", "seed",
    "n_fit_clips", "n_cal_clips", "n_test_normal_clips", "n_abnormal_clips",
    "auc_clip", "pauc_clip", "tpr_at_alpha", "realized_normal_clip_far",
    "auc_window", "degenerate",
]


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch, tmp_path):
    # chdir away from the repo root so load_config() cannot pick up the repo's own
    # .env file (deleting the env vars alone would not mask file values) --
    # tests/test_benchmark_inference.py's established convention.
    monkeypatch.chdir(tmp_path)
    for var in (
        "ROWII_BEATS_CHECKPOINT", "ROWII_BEATS_INT8_CHECKPOINT",
        "ROWII_TFC_AUDIO_CHECKPOINT", "ROWII_STUDENT_CHECKPOINT",
    ):
        monkeypatch.delenv(var, raising=False)


def _write_wav_file(path: Path, samples: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    int16 = np.clip(samples * 32767, -32768, 32767).astype(np.int16)
    write_wav(path, _RATE_HZ, int16)


def _build_separable_tree(
    root: Path,
    *,
    machine_ids: tuple[str, ...] = ("id_00", "id_02"),
    n_normal: int = 6,
    n_abnormal: int = 4,
    duration_s: float = 2.0,
) -> None:
    """`root/pump/id_*/{normal,abnormal}/<NNNNNNNN>.wav` (MIMII's real layout,
    the Task-4 builder pattern): normal = 200 Hz tone + light noise, abnormal =
    loud broadband noise (module docstring on why spectral, not level,
    separation is required)."""
    t = np.arange(int(round(duration_s * _RATE_HZ))) / _RATE_HZ
    seed = 0
    for mid in machine_ids:
        for i in range(n_normal):
            rng = np.random.default_rng(seed)
            seed += 1
            samples = 0.4 * np.sin(2 * np.pi * 200.0 * t) + rng.normal(0.0, 0.01, t.shape)
            _write_wav_file(root / "pump" / mid / "normal" / f"{i:08d}.wav", samples)
        for i in range(n_abnormal):
            rng = np.random.default_rng(seed)
            seed += 1
            _write_wav_file(
                root / "pump" / mid / "abnormal" / f"{i:08d}.wav",
                rng.normal(0.0, 0.5, t.shape),
            )


def _run(root: Path, out: Path, overrides: dict[str, str] | None = None) -> int:
    """`main()` with the plan's smoke arguments (handcrafted, fractions 0.5/1.0,
    seed 7, cap 4); *overrides* replaces individual flag values -- an empty-string
    value passes the flag bare (for store_true flags like `--dry-run`)."""
    values = {
        "--representations": "handcrafted",
        "--fractions": "0.5,1.0",
        "--seeds": "7",
        "--limit-clips-per-class": "4",
        **(overrides or {}),
    }
    argv = ["--root", str(root), "--out", str(out)]
    for flag, value in values.items():
        argv.append(flag)
        if value:
            argv.append(value)
    return scarcity_detection.main(argv)


def _read_table(out: Path) -> pd.DataFrame:
    return pd.read_csv(out / "scarcity_detection.csv", comment="#")


class TestEndToEnd:
    def test_exit_zero_with_all_outputs_present(self, tmp_path):
        root, out = tmp_path / "corpus", tmp_path / "out"
        _build_separable_tree(root)

        assert _run(root, out) == 0

        assert (out / "scarcity_detection.csv").is_file()
        assert (out / "scarcity_detection.md").is_file()
        assert (out / "scarcity_curve.png").is_file()
        assert list((out / "cache").glob("*.npz"))  # one embedding cache per machine

    def test_csv_schema_exact_and_pauc_definition_named_in_header_comment(self, tmp_path):
        root, out = tmp_path / "corpus", tmp_path / "out"
        _build_separable_tree(root)

        assert _run(root, out) == 0

        first_line = (out / "scarcity_detection.csv").read_text().splitlines()[0]
        assert first_line.startswith("#")
        assert "McClish" in first_line  # A1.4: definition named in a CSV header comment
        assert "max_fpr=0.1" in first_line
        table = _read_table(out)
        assert list(table.columns) == _EXPECTED_COLUMNS
        assert len(table) == 4  # 2 machine ids x 2 fractions x 1 seed
        assert set(table["representation"]) == {"handcrafted"}
        assert set(table["machine_id"]) == {"id_00", "id_02"}

    def test_n_fit_clips_monotone_in_fraction(self, tmp_path):
        root, out = tmp_path / "corpus", tmp_path / "out"
        _build_separable_tree(root)

        assert _run(root, out) == 0

        table = _read_table(out)
        for _, group in table.groupby("machine_id"):
            by_fraction = group.sort_values("fraction")["n_fit_clips"].tolist()
            assert by_fraction == sorted(by_fraction)
            assert by_fraction[0] < by_fraction[-1]  # 0.5 vs 1.0 differ on this corpus

    def test_auc_clip_in_range_and_separates_the_synthetic_classes(self, tmp_path):
        root, out = tmp_path / "corpus", tmp_path / "out"
        _build_separable_tree(root)

        assert _run(root, out) == 0

        table = _read_table(out)
        scored = table[~table["degenerate"]]
        assert len(scored) == 4  # no degenerate cells at cap 4 with fractions 0.5/1.0
        assert ((scored["auc_clip"] >= 0.0) & (scored["auc_clip"] <= 1.0)).all()
        assert (scored["auc_clip"] > 0.5).all()
        assert ((scored["auc_window"] >= 0.0) & (scored["auc_window"] <= 1.0)).all()

    def test_determinism_second_run_byte_identical_csv_and_cache_hit_logged(
        self, tmp_path, caplog
    ):
        root, out = tmp_path / "corpus", tmp_path / "out"
        _build_separable_tree(root)
        assert _run(root, out) == 0
        first_bytes = (out / "scarcity_detection.csv").read_bytes()

        with caplog.at_level(logging.INFO):
            caplog.clear()
            assert _run(root, out) == 0

        assert (out / "scarcity_detection.csv").read_bytes() == first_bytes
        hits = [r.getMessage() for r in caplog.records if "cache HIT" in r.getMessage()]
        assert len(hits) == 2  # one per machine id -- extraction ran ONCE, first run only

    def test_md_restates_binding_framings(self, tmp_path):
        root, out = tmp_path / "corpus", tmp_path / "out"
        _build_separable_tree(root)

        assert _run(root, out) == 0

        md = (out / "scarcity_detection.md").read_text()
        assert "McClish" in md  # A1.4: pAUC definition named in the md
        assert "per-window standardized" in md  # A1.5: standardization caveat
        # A1.5: student-transferability framing, verbatim from the spec.
        assert "The student on MIMII measures TRANSFERABILITY of a PSHP-distilled encoder" in md
        assert "public-proxy" in md  # spec section 4 honesty framing
        assert "never window-calibrated/clip-applied" in md  # A1.4 coherence rule


class TestProtocol:
    def test_degenerate_tiny_fraction_cell_flagged_not_crashed(self, tmp_path, caplog):
        root, out = tmp_path / "corpus", tmp_path / "out"
        _build_separable_tree(root)

        with caplog.at_level(logging.INFO):
            # pool = 3 clips -> fraction 0.05 draws ceil(0.15) = 1 clip -> the 80/20
            # split leaves the fit side EMPTY -- must flag, never crash.
            assert _run(root, out, {"--fractions": "0.05,1.0"}) == 0

        table = _read_table(out)
        tiny = table[table["fraction"] == 0.05]
        assert len(tiny) == 2
        assert tiny["degenerate"].all()
        assert tiny["auc_clip"].isna().all()
        assert tiny["tpr_at_alpha"].isna().all()
        full = table[table["fraction"] == 1.0]
        assert not full["degenerate"].any()
        assert any("degenerate" in r.getMessage() for r in caplog.records)

    def test_test_split_shared_across_fraction_and_seed_cells(self, tmp_path):
        root, out = tmp_path / "corpus", tmp_path / "out"
        _build_separable_tree(root)

        assert _run(root, out, {"--seeds": "7,8"}) == 0

        table = _read_table(out)
        for _, group in table.groupby("machine_id"):
            assert len(group) == 4  # 2 fractions x 2 seeds
            # The seed-7 TEST draw happens ONCE per machine and is shared by every
            # (fraction, seed) cell: identical held-out count everywhere ...
            assert group["n_test_normal_clips"].nunique() == 1
            n_test = int(group["n_test_normal_clips"].iloc[0])
            # ... and the fraction-1.0 draw uses the ENTIRE remaining pool, so
            # fit + cal + test must reconstruct all 4 kept normal clips exactly.
            full = group[group["fraction"] == 1.0]
            assert ((full["n_fit_clips"] + full["n_cal_clips"] + n_test) == 4).all()
            assert group["n_abnormal_clips"].nunique() == 1

    def test_standardized_pauc_definition_pinned_by_hand_computed_case(self):
        # Hand derivation (A1.4: the pAUC definition is pinned against a hand-
        # computed case). 10 normals scored 0..9, abnormals scored 8.5 and 20:
        # ROC points with FPR <= 0.1 are (0,0) -> (0,0.5) [threshold 20] ->
        # (0.1,0.5) [normal 9 admitted] -> (0.1,1.0) [threshold 8.5]. Raw partial
        # area = 0.5 * 0.1 = 0.005... trapezoid: 0.05; McClish standardization
        # with min = 0.5 * 0.1^2 = 0.005 and max = 0.1 gives
        # 0.5 * (1 + (0.05 - 0.005) / (0.1 - 0.005)) = 0.5 * (1 + 9/19) = 14/19.
        y = np.array([0] * 10 + [1] * 2)
        scores = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 8.5, 20.0])

        assert scarcity_detection._standardized_pauc(y, scores) == pytest.approx(14 / 19)


class TestCliContract:
    def test_unknown_representation_exits_2(self, tmp_path):
        root = tmp_path / "corpus"
        _build_separable_tree(root)

        with pytest.raises(SystemExit) as exc_info:
            _run(root, tmp_path / "out", {"--representations": "handcrafted,bogus"})

        assert exc_info.value.code == 2

    def test_beats_skipped_with_log_when_checkpoint_env_unset(self, tmp_path, caplog):
        root, out = tmp_path / "corpus", tmp_path / "out"
        _build_separable_tree(root)

        with caplog.at_level(logging.INFO):
            assert _run(root, out, {"--representations": "handcrafted,beats"}) == 0

        skips = [
            r.getMessage() for r in caplog.records
            if "skipping" in r.getMessage() and "beats" in r.getMessage()
        ]
        assert skips  # skip-with-log, never an error (benchmark_inference semantics)
        table = _read_table(out)
        assert set(table["representation"]) == {"handcrafted"}  # no beats rows

    def test_dry_run_prints_cell_matrix_and_writes_nothing(self, tmp_path, capsys):
        root, out = tmp_path / "corpus", tmp_path / "out"
        _build_separable_tree(root)

        assert _run(root, out, {"--dry-run": ""}) == 0

        assert not out.exists()  # writes NOTHING, not even the out directory
        printed = capsys.readouterr().out
        assert "dry-run" in printed
        assert "handcrafted" in printed
        assert "limit_clips_per_class" in printed  # caps printed
        assert "id_00" in printed and "id_02" in printed
        assert "cells" in printed

    def test_machine_ids_filters_and_unknown_id_exits_2(self, tmp_path):
        root, out = tmp_path / "corpus", tmp_path / "out"
        _build_separable_tree(root)

        assert _run(root, out, {"--machine-ids": "id_00"}) == 0
        table = _read_table(out)
        assert set(table["machine_id"]) == {"id_00"}

        assert _run(root, tmp_path / "out2", {"--machine-ids": "id_99"}) == 2
