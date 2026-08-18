"""Tests for scripts/make_thesis_figures.py: pure data-loading/aggregation helpers on
tiny synthetic fixtures (never the committed `results/` artifacts -- hermetic and fast,
mirroring `tests/test_analyze_days.py`'s own precedent), plus one end-to-end smoke test
that the four PDFs actually get written. Pixel content is verified separately, by hand,
via `pdftoppm` + visual inspection (not something a unit test can usefully assert)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import pytest

SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import make_thesis_figures as m  # noqa: E402

# ---------------------------------------------------------------------------
# session_x_positions / session_tick_label
# ---------------------------------------------------------------------------


def test_session_x_positions_are_unit_spaced_within_an_era() -> None:
    xs = m.session_x_positions()
    era_a = [xs[i] for i, s in enumerate(m.SESSIONS) if s.era == "A"]
    assert era_a[1] - era_a[0] == pytest.approx(1.0)
    assert era_a[2] - era_a[1] == pytest.approx(1.0)


def test_session_x_positions_inserts_era_gap_at_each_boundary() -> None:
    xs = m.session_x_positions()
    eras = [s.era for s in m.SESSIONS]
    boundaries = [i for i in range(1, len(eras)) if eras[i] != eras[i - 1]]
    assert len(boundaries) == 2  # A->B and B->C
    for i in boundaries:
        step = xs[i] - xs[i - 1]
        assert step == pytest.approx(1.0 + m._ERA_GAP)


def test_session_tick_label_appends_asterisk_only_for_in_sample() -> None:
    in_sample = next(s for s in m.SESSIONS if s.in_sample)
    not_in_sample = next(s for s in m.SESSIONS if not s.in_sample)
    assert m.session_tick_label(in_sample).endswith("*")
    assert not m.session_tick_label(not_in_sample).endswith("*")


# ---------------------------------------------------------------------------
# F1/F2 loaders: fusion_regimes.csv, fusion_trigger_log.csv, fusion.json
# ---------------------------------------------------------------------------


def _write_fusion_tree(tmp_path: Path) -> Path:
    fusion_dir = tmp_path / "step2" / "once-calibrated" / "fusion"
    fusion_dir.mkdir(parents=True)
    regimes = pd.DataFrame(
        [
            {
                "day": 250526,
                "era": "A",
                "run": "250526-tu",
                "tags": "()",
                "always_frozen_far": 0.056,
                "always_recalibrate_far": 0.027,
                "once_triggered_far": 0.027,
                "frozen_far_full_population": 0.105,
                "far_basis": "common-window",
                "decision": "recalibrate",
            },
            {
                "day": 290626,
                "era": "B",
                "run": "290626-tu",
                "tags": "()",
                "always_frozen_far": 0.075,
                "always_recalibrate_far": 0.044,
                "once_triggered_far": 0.075,
                "frozen_far_full_population": 0.044,
                "far_basis": "common-window",
                "decision": "frozen",
            },
        ]
    )
    regimes.to_csv(fusion_dir / "fusion_regimes.csv", index=False)

    trigger_rows = []
    for s in m.SESSIONS:
        trigger_rows.append(
            {
                "day": s.run.split("-")[0],
                "era": s.era,
                "tags": "()",
                "s1_rate": 0.5 if s.run == "270626-pu_ph_pu_ph_pu_ph-1" else 0.03,
                "s1_threshold": 0.0805,
                "s1_fired": s.run == "270626-pu_ph_pu_ph_pu_ph-1",
                "low_confidence_modes": "()",
                "s2_mic_median": -2.8,
                "s2_vib_median": -5.3,
                "s2_anchor": -2.8,
                "s2_mad": 0.3,
                "s2_fired": False,
                "s2_attribution": "machine",
                "decision": "frozen",
                "run": s.run,
            }
        )
    pd.DataFrame(trigger_rows).to_csv(fusion_dir / "fusion_trigger_log.csv", index=False)

    (fusion_dir / "fusion.json").write_text(
        json.dumps({"alpha": 0.05, "s1": {"threshold": 0.0805}})
    )
    return tmp_path


def test_load_fusion_regimes_indexes_rows_by_run(tmp_path: Path) -> None:
    results_dir = _write_fusion_tree(tmp_path)
    regimes = m.load_fusion_regimes(results_dir)
    assert regimes.loc["290626-tu", "once_triggered_far"] == pytest.approx(0.075)
    assert "270626-pu_ph_pu_ph_pu_ph-1" not in regimes.index  # sentinel-only, no FAR row


def test_load_fusion_trigger_log_parses_s1_fired_as_bool(tmp_path: Path) -> None:
    results_dir = _write_fusion_tree(tmp_path)
    trigger = m.load_fusion_trigger_log(results_dir)
    assert bool(trigger.loc["270626-pu_ph_pu_ph_pu_ph-1", "s1_fired"]) is True
    assert bool(trigger.loc["290626-tu", "s1_fired"]) is False


def test_load_fusion_meta_reads_alpha_and_threshold(tmp_path: Path) -> None:
    results_dir = _write_fusion_tree(tmp_path)
    meta = m.load_fusion_meta(results_dir)
    assert meta["alpha"] == pytest.approx(0.05)
    assert meta["s1"]["threshold"] == pytest.approx(0.0805)


def test_make_f2_sentinel_requires_every_session_in_the_trigger_log(tmp_path: Path) -> None:
    """F2 has no "n/a" branch (unlike F1): every one of the 8 sessions must have a
    sentinel rate, so a trigger log missing one session is a hard failure, not a
    silently blank bar."""
    results_dir = _write_fusion_tree(tmp_path)
    fusion_dir = results_dir / "step2" / "once-calibrated" / "fusion"
    trigger = pd.read_csv(fusion_dir / "fusion_trigger_log.csv")
    trigger = trigger[trigger["run"] != "080726-pu_strikes"]
    trigger.to_csv(fusion_dir / "fusion_trigger_log.csv", index=False)
    with pytest.raises(AssertionError):
        m.make_f2_sentinel(results_dir, tmp_path / "out")


# ---------------------------------------------------------------------------
# F3: scarcity_detection.csv
# ---------------------------------------------------------------------------


def test_scarcity_curve_means_averages_over_machine_id_and_seed() -> None:
    df = pd.DataFrame(
        [
            {"representation": "beats", "fraction": 0.05, "auc_clip": 0.90},
            {"representation": "beats", "fraction": 0.05, "auc_clip": 0.92},
            {"representation": "beats", "fraction": 0.10, "auc_clip": 0.95},
        ]
    )
    means = m.scarcity_curve_means(df)
    assert means["beats"].loc[0.05] == pytest.approx(0.91)
    assert means["beats"].loc[0.10] == pytest.approx(0.95)


def _write_scarcity_csv(tmp_path: Path) -> Path:
    results_dir = tmp_path
    scarcity_dir = results_dir / "scarcity-detection"
    scarcity_dir.mkdir(parents=True)
    rows = []
    for rep in m._SCARCITY_REPR_ORDER:
        for frac in m._SCARCITY_FRACTIONS:
            for machine in ("id_00", "id_02"):
                rows.append(
                    {
                        "representation": rep,
                        "machine_id": machine,
                        "fraction": frac,
                        "seed": 7,
                        "n_fit_clips": 8,
                        "n_cal_clips": 3,
                        "n_test_normal_clips": 90,
                        "n_abnormal_clips": 100,
                        "auc_clip": 0.8,
                        "pauc_clip": 0.7,
                        "tpr_at_alpha": 0.0,
                        "realized_normal_clip_far": 0.0,
                        "auc_window": 0.8,
                        "degenerate": False,
                    }
                )
    # one degenerate row that must be dropped
    rows.append(
        {
            "representation": "beats",
            "machine_id": "id_00",
            "fraction": 0.05,
            "seed": 9,
            "n_fit_clips": 0,
            "n_cal_clips": 0,
            "n_test_normal_clips": 90,
            "n_abnormal_clips": 100,
            "auc_clip": float("nan"),
            "pauc_clip": float("nan"),
            "tpr_at_alpha": float("nan"),
            "realized_normal_clip_far": float("nan"),
            "auc_window": float("nan"),
            "degenerate": True,
        }
    )
    path = scarcity_dir / "scarcity_detection.csv"
    with path.open("w") as f:
        f.write("# pauc_clip = ... (a leading comment line, like the real artifact)\n")
    pd.DataFrame(rows).to_csv(path, mode="a", index=False)
    return results_dir


def test_load_scarcity_curve_skips_the_comment_line_and_degenerate_rows(tmp_path: Path) -> None:
    results_dir = _write_scarcity_csv(tmp_path)
    df = m.load_scarcity_curve(results_dir)
    assert not df["degenerate"].any()
    assert len(df) == len(m._SCARCITY_REPR_ORDER) * len(m._SCARCITY_FRACTIONS) * 2


# ---------------------------------------------------------------------------
# F4: latency.csv
# ---------------------------------------------------------------------------


def _write_latency_csv(tmp_path: Path) -> Path:
    results_dir = tmp_path
    pillar_dir = results_dir / "pillar3-perstrike"
    pillar_dir.mkdir(parents=True)
    rows = []
    for session in ("st", "pu"):
        for rep in m._LATENCY_REPR_ORDER:
            for k in range(3):
                missed = rep == "vibration"
                rows.append(
                    {
                        "row_type": "detail",
                        "level": "physical_strike",
                        "session": session,
                        "representation": rep,
                        "regime": "once-calibrated/recalibrate",
                        "alarms_path": "x",
                        "alpha": 0.05,
                        "search_horizon_s": 5.0,
                        "n_total": "",
                        "n_detected": "",
                        "n_missed": "",
                        "median_s": "",
                        "iqr_low_s": "",
                        "iqr_high_s": "",
                        "event_id": f"{k:02d}",
                        "kind": "plate-gen",
                        "kind_group": "plate-gen",
                        "strike_no": k + 1,
                        "n_impulses": 3,
                        "ref_utc": "2026-07-08 10:15:00+00:00",
                        "latency_s": float("nan") if missed else 0.3 + 0.1 * k,
                        "missed": missed,
                    }
                )
        # one summary row per session/representation, must be excluded by the loader
        rows.append(
            {
                "row_type": "summary",
                "level": "physical_strike",
                "session": session,
                "representation": "fusion",
                "regime": "once-calibrated/recalibrate",
                "alarms_path": "x",
                "alpha": 0.05,
                "search_horizon_s": 5.0,
                "n_total": 3,
                "n_detected": 3,
                "n_missed": 0,
                "median_s": 0.3,
                "iqr_low_s": 0.3,
                "iqr_high_s": 0.4,
                "event_id": "",
                "kind": "",
                "kind_group": "",
                "strike_no": "",
                "n_impulses": "",
                "ref_utc": "",
                "latency_s": "",
                "missed": "",
            }
        )
    pd.DataFrame(rows).to_csv(pillar_dir / "latency.csv", index=False)
    return results_dir


def test_load_latency_detail_keeps_only_detail_physical_strike_rows(tmp_path: Path) -> None:
    results_dir = _write_latency_csv(tmp_path)
    df = m.load_latency_detail(results_dir)
    assert (df["row_type"] == "detail").all()
    assert (df["level"] == "physical_strike").all()


def test_latency_group_counts_splits_detected_from_missed(tmp_path: Path) -> None:
    results_dir = _write_latency_csv(tmp_path)
    df = m.load_latency_detail(results_dir)
    detected, n_det, n_total = m.latency_group_counts(df, "st", "audio-beats")
    assert n_total == 3
    assert n_det == 3
    assert list(detected) == pytest.approx([0.3, 0.4, 0.5])


def test_latency_group_counts_handles_zero_detections(tmp_path: Path) -> None:
    results_dir = _write_latency_csv(tmp_path)
    df = m.load_latency_detail(results_dir)
    detected, n_det, n_total = m.latency_group_counts(df, "st", "vibration")
    assert n_total == 3
    assert n_det == 0
    assert len(detected) == 0


# ---------------------------------------------------------------------------
# F5 loaders: cross-day rotation far_table.csv + within-day pooled aggregation
# ---------------------------------------------------------------------------

_CROSS_DAY_COLS = "label,n_calibration,n_scored,n_alarms,realized_far,nominal_alpha\n"


def _write_transfer_tree(results_dir: Path) -> Path:
    """A synthetic cross-day + within-day tree over the three real day ids, with
    made-up rates: the loaders are what is under test, never the plotted values."""
    runs = [s.run for s in m.TRANSFER_DAYS]
    for i, src in enumerate(runs):
        for j, dst in enumerate(runs):
            if i == j:
                continue
            d = (
                results_dir / "step2" / "cross-day" / "fusion-detected"
                / f"{src}__to__{dst}" / "knn-pooled"
            )
            d.mkdir(parents=True)
            far = 0.1 * (i + 1) + 0.01 * (j + 1)
            (d / "far_table.csv").write_text(
                _CROSS_DAY_COLS + f"pooled,100,200,{round(far * 200)},{far},0.05\n"
            )
    for k, run in enumerate(runs):
        d = results_dir / "step2" / "within-day" / run / "fusion-detected" / "pooled-knn"
        d.mkdir(parents=True)
        # Two per-state rows plus a stray non-numeric label: the aggregate must be
        # sum(alarms)/sum(scored) over the per-state rows only.
        (d / "far_table.csv").write_text(
            _CROSS_DAY_COLS
            + f"0,50,100,{10 + k},{(10 + k) / 100},0.05\n"
            + f"1,50,300,{20 + k},{(20 + k) / 300},0.05\n"
        )
    return results_dir


def test_load_transfer_cell_reads_the_pooled_row(tmp_path: Path) -> None:
    results_dir = _write_transfer_tree(tmp_path / "results")
    far, n_scored = m.load_transfer_cell(results_dir, "250526-tu", "290626-tu")
    assert far == pytest.approx(0.12)
    assert n_scored == 200


def test_load_own_day_cell_aggregates_alarms_over_scored(tmp_path: Path) -> None:
    results_dir = _write_transfer_tree(tmp_path / "results")
    far, n_scored = m.load_own_day_cell(results_dir, "250526-tu")
    assert n_scored == 400
    assert far == pytest.approx((10 + 20) / 400)


def test_transfer_matrix_puts_own_day_values_on_the_diagonal(tmp_path: Path) -> None:
    results_dir = _write_transfer_tree(tmp_path / "results")
    far, scored = m.transfer_matrix(results_dir)
    n = len(m.TRANSFER_DAYS)  # 4 since the 30 June extension (2026-08-18)
    assert far.shape == (n, n)
    for k in range(n):
        assert far[k, k] == pytest.approx((10 + k + 20 + k) / 400)
        assert scored[k, k] == 400
    assert far[0, 1] == pytest.approx(0.12)
    assert scored[0, 1] == 200


def test_load_transfer_cell_rejects_a_foreign_nominal_alpha(tmp_path: Path) -> None:
    results_dir = _write_transfer_tree(tmp_path / "results")
    path = (
        results_dir / "step2" / "cross-day" / "fusion-detected"
        / "250526-tu__to__290626-tu" / "knn-pooled" / "far_table.csv"
    )
    path.write_text(_CROSS_DAY_COLS + "pooled,100,200,24,0.12,0.10\n")
    with pytest.raises(AssertionError):
        m.load_transfer_cell(results_dir, "250526-tu", "290626-tu")


# ---------------------------------------------------------------------------
# End-to-end smoke test: all five PDFs get written and are non-trivially sized
# ---------------------------------------------------------------------------


def test_make_all_writes_five_nonempty_pdfs(tmp_path: Path) -> None:
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    _write_fusion_tree(results_dir)
    scarcity_src = _write_scarcity_csv(tmp_path / "scarcity_src")
    (scarcity_src / "scarcity-detection").rename(results_dir / "scarcity-detection")
    latency_src = _write_latency_csv(tmp_path / "latency_src")
    (latency_src / "pillar3-perstrike").rename(results_dir / "pillar3-perstrike")
    _write_transfer_tree(results_dir)

    out_dir = tmp_path / "out"
    paths = m.make_all(results_dir, out_dir)

    assert len(paths) == 5
    for p in paths:
        assert p.is_file()
        assert p.suffix == ".pdf"
        assert p.stat().st_size > 2_000  # a blank/broken PDF page is far smaller
