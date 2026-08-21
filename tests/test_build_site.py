"""Tests for `scripts/build_site.py`'s PURE helpers only (candidate curation/pinning,
clip-window arithmetic for both the candidate-kit and annotation-kit sources, the
as-built sensor-geometry table, and the external-resource-URL scanner used by this
task's own "zero external requests" verification step) -- synthetic fixtures
throughout, no real `results/candidate-kit`/`results/annotation-kit` data anywhere
(mirrors `tests/test_make_demo_assets.py`'s "pure-math, no real data" posture). The
IO-touching parts of that script (reading the real candidate/annotation registers,
trimming/writing WAVs, writing the four site HTML pages) are exercised by actually
running the CLI against the real repo data as part of this task, not by a unit test
here.

Import convention mirrors `tests/test_make_demo_assets.py`: `scripts/` is not a
package, so the module under test is imported directly by inserting `scripts/` onto
`sys.path`.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import build_site as bs  # noqa: E402

# ---------------------------------------------------------------------------
# 1. select_top_candidates -- per-class strength ranking + pinning
# ---------------------------------------------------------------------------


def _row(candidate_id: str, klass: str, min_p: float) -> dict[str, str]:
    """A minimal `candidates.csv`-shaped row -- only the two columns
    `select_top_candidates` actually reads (`class`, `min_p`) plus the id."""
    return {"candidate_id": candidate_id, "class": klass, "min_p": str(min_p)}


def test_select_top_candidates_keeps_the_lowest_min_p_per_class() -> None:
    rows = [
        _row("s-1", "sustained", 0.02),
        _row("s-2", "sustained", 0.01),
        _row("s-3", "sustained", 0.03),
        _row("t-1", "transient", 0.05),
        _row("t-2", "transient", 0.04),
    ]
    kept = bs.select_top_candidates(rows, n_per_class=2)
    ids = [r["candidate_id"] for r in kept]
    assert ids == ["s-2", "s-1", "t-2", "t-1"]  # sustained (sorted class name) first


def test_select_top_candidates_orders_classes_alphabetically_and_rows_by_min_p() -> None:
    rows = [
        _row("t-1", "transient", 0.09),
        _row("s-1", "sustained", 0.09),
    ]
    kept = bs.select_top_candidates(rows, n_per_class=1)
    assert [r["candidate_id"] for r in kept] == ["s-1", "t-1"]


def test_select_top_candidates_pin_outside_natural_topn_is_included() -> None:
    rows = [
        _row("t-1", "transient", 0.001),
        _row("t-2", "transient", 0.002),
        _row("t-3", "transient", 0.003),
        _row("t-4", "transient", 0.004),
        _row("t-5", "transient", 0.005),
        _row("t-pin", "transient", 0.900),  # far weaker than the natural top 5
    ]
    kept = bs.select_top_candidates(rows, n_per_class=5, pinned_ids=frozenset({"t-pin"}))
    ids = {r["candidate_id"] for r in kept}
    assert "t-pin" in ids
    assert len(kept) == 5
    # the pin displaces exactly the single weakest natural member (t-5)
    assert "t-5" not in ids
    assert {"t-1", "t-2", "t-3", "t-4"} <= ids


def test_select_top_candidates_pin_already_in_natural_topn_is_not_duplicated() -> None:
    rows = [_row("t-1", "transient", 0.001), _row("t-2", "transient", 0.002)]
    kept = bs.select_top_candidates(rows, n_per_class=2, pinned_ids=frozenset({"t-1"}))
    assert [r["candidate_id"] for r in kept] == ["t-1", "t-2"]


def test_select_top_candidates_more_pins_than_n_per_class_keeps_every_pin() -> None:
    rows = [_row(f"t-{i}", "transient", float(i)) for i in range(5)]
    pins = frozenset({"t-0", "t-1", "t-2"})
    kept = bs.select_top_candidates(rows, n_per_class=2, pinned_ids=pins)
    ids = {r["candidate_id"] for r in kept}
    assert pins <= ids  # every pin survives even though n_per_class == 2


def test_select_top_candidates_unknown_pin_raises() -> None:
    rows = [_row("t-1", "transient", 0.01)]
    with pytest.raises(ValueError, match="t-does-not-exist"):
        bs.select_top_candidates(rows, n_per_class=1, pinned_ids=frozenset({"t-does-not-exist"}))


def test_select_top_candidates_n_per_class_larger_than_available_returns_all() -> None:
    rows = [_row("s-1", "sustained", 0.01), _row("s-2", "sustained", 0.02)]
    kept = bs.select_top_candidates(rows, n_per_class=10)
    assert len(kept) == 2


def test_select_top_candidates_rejects_non_positive_n_per_class() -> None:
    with pytest.raises(ValueError, match="n_per_class"):
        bs.select_top_candidates([_row("s-1", "sustained", 0.01)], n_per_class=0)


# ---------------------------------------------------------------------------
# 2. centered_or_leading_window -- candidate-kit clip trim-window arithmetic
# ---------------------------------------------------------------------------


def test_centered_window_for_an_event_shorter_than_the_clip() -> None:
    # 1 s event at offset 10.0 s, clipped to a 10 s window -> centered: starts
    # (10 - 1) / 2 = 4.5 s before the event.
    start = bs.centered_or_leading_window(
        event_start_s=10.0, event_duration_s=1.0, clip_duration_s=10.0, available_duration_s=21.0
    )
    assert start == pytest.approx(5.5)


def test_leading_window_for_an_event_at_least_as_long_as_the_clip() -> None:
    start = bs.centered_or_leading_window(
        event_start_s=8.0, event_duration_s=12.0, clip_duration_s=10.0, available_duration_s=40.0
    )
    assert start == pytest.approx(8.0)  # starts exactly at the anomaly, no centering


def test_leading_window_at_exactly_the_clip_duration_starts_at_event() -> None:
    start = bs.centered_or_leading_window(
        event_start_s=3.0, event_duration_s=10.0, clip_duration_s=10.0, available_duration_s=40.0
    )
    assert start == pytest.approx(3.0)


def test_centered_window_clamped_when_it_would_start_before_zero() -> None:
    start = bs.centered_or_leading_window(
        event_start_s=1.0, event_duration_s=1.0, clip_duration_s=10.0, available_duration_s=21.0
    )
    assert start == 0.0


def test_centered_window_clamped_when_it_would_run_past_the_available_span() -> None:
    start = bs.centered_or_leading_window(
        event_start_s=19.5, event_duration_s=1.0, clip_duration_s=10.0, available_duration_s=21.0
    )
    assert start == pytest.approx(11.0)  # 21 - 10


def test_centered_window_raises_if_the_clip_cannot_fit_at_all() -> None:
    with pytest.raises(ValueError, match="available_duration_s"):
        bs.centered_or_leading_window(
            event_start_s=1.0, event_duration_s=1.0, clip_duration_s=10.0, available_duration_s=5.0
        )


# ---------------------------------------------------------------------------
# 3. strike_window_start_s -- annotation-kit (hammer-strike) trim-window arithmetic
# ---------------------------------------------------------------------------


def test_strike_window_anchors_just_before_the_earliest_strike() -> None:
    start = bs.strike_window_start_s(
        strike_offsets_s=[16.568, 17.248, 18.023],
        clip_duration_s=10.0,
        available_duration_s=90.0,
        lead_pad_s=2.0,
    )
    assert start == pytest.approx(14.568)


def test_strike_window_uses_the_earliest_strike_regardless_of_input_order() -> None:
    start = bs.strike_window_start_s(
        strike_offsets_s=[18.023, 16.568, 17.248],
        clip_duration_s=10.0,
        available_duration_s=90.0,
        lead_pad_s=2.0,
    )
    assert start == pytest.approx(14.568)


def test_strike_window_defaults_to_the_start_of_the_clip_when_no_strikes_are_known() -> None:
    # e.g. `landmark-A_kugelschieber`, which has zero compiled per-strike rows
    # (docs/groundtruth/080726_strikes_seconds_pu.csv's own coverage gap).
    start = bs.strike_window_start_s(
        strike_offsets_s=[], clip_duration_s=10.0, available_duration_s=90.0
    )
    assert start == 0.0


def test_strike_window_clamped_when_the_lead_pad_would_go_negative() -> None:
    start = bs.strike_window_start_s(
        strike_offsets_s=[0.5], clip_duration_s=10.0, available_duration_s=90.0, lead_pad_s=2.0
    )
    assert start == 0.0


def test_strike_window_clamped_when_the_strike_is_near_the_end() -> None:
    start = bs.strike_window_start_s(
        strike_offsets_s=[89.0], clip_duration_s=10.0, available_duration_s=90.0, lead_pad_s=2.0
    )
    assert start == pytest.approx(80.0)  # 90 - 10


def test_strike_window_raises_if_the_clip_cannot_fit_at_all() -> None:
    with pytest.raises(ValueError, match="available_duration_s"):
        bs.strike_window_start_s(
            strike_offsets_s=[], clip_duration_s=10.0, available_duration_s=5.0
        )


# ---------------------------------------------------------------------------
# 4. sensor_layout -- the as-built sensor-geometry table behind sensors.html
# ---------------------------------------------------------------------------


def test_sensor_layout_has_nine_microphones_and_four_accelerometers() -> None:
    points = bs.sensor_layout()
    mics = [p for p in points if p.group in ("genmic", "turmic")]
    accels = [p for p in points if p.group in ("genvib", "turvib")]
    assert len(mics) == 9  # 4 generator ring + 4 turbine ring + 1 turbine bottom
    assert len(accels) == 4  # 2 per level (0 deg / 180 deg), each tri-axial


def test_sensor_layout_labels_match_the_documented_naming_convention() -> None:
    labels = {p.label for p in bs.sensor_layout()}
    expected = {
        "GenMic0", "GenMic90", "GenMic180", "GenMic270",
        "TurMic0", "TurMic90", "TurMic180", "TurMic270", "TurMicBottom",
        "GenVib0", "GenVib180", "TurVib0", "TurVib180",
    }
    assert labels == expected


def test_sensor_layout_has_no_duplicate_labels() -> None:
    labels = [p.label for p in bs.sensor_layout()]
    assert len(labels) == len(set(labels))


def test_sensor_layout_accelerometers_each_carry_three_axis_channels() -> None:
    for p in bs.sensor_layout():
        if p.group in ("genvib", "turvib"):
            assert p.channels == (f"{p.label}X", f"{p.label}Y", f"{p.label}Z")


def test_sensor_layout_microphones_carry_exactly_one_channel_matching_their_label() -> None:
    for p in bs.sensor_layout():
        if p.group in ("genmic", "turmic"):
            assert p.channels == (p.label,)


def test_sensor_layout_turbine_bottom_mic_has_no_ring_angle() -> None:
    bottom = next(p for p in bs.sensor_layout() if p.label == "TurMicBottom")
    assert bottom.angle_deg is None


# ---------------------------------------------------------------------------
# 5. find_external_resource_urls -- the "zero external requests" scanner
# ---------------------------------------------------------------------------


def test_find_external_resource_urls_empty_on_plain_local_html() -> None:
    html = "<html><body><a href=\"snippets.html\">Listening library</a></body></html>"
    assert bs.find_external_resource_urls(html) == []


def test_find_external_resource_urls_flags_an_external_script_src() -> None:
    html = '<script src="https://cdn.example.com/lib.js"></script>'
    hits = bs.find_external_resource_urls(html)
    assert hits == ["https://cdn.example.com/lib.js"]


def test_find_external_resource_urls_flags_an_external_stylesheet_and_href() -> None:
    html = (
        '<link rel="stylesheet" href="https://fonts.example.com/a.css">'
        '<a href="http://example.com/page">plain link</a>'
    )
    hits = bs.find_external_resource_urls(html)
    assert set(hits) == {"https://fonts.example.com/a.css", "http://example.com/page"}


def test_find_external_resource_urls_ignores_bare_text_mentions() -> None:
    # A URL that appears only as visible prose text, not inside a src/href/action
    # attribute, is not a resource load and must not be flagged.
    html = "<p>Data courtesy of the plant operator (see https://example.org/policy).</p>"
    assert bs.find_external_resource_urls(html) == []


def test_design_css_v7_tokens_and_no_legacy_look() -> None:
    css_path = Path(__file__).resolve().parents[1] / "docs" / "site" / "assets" / "design.css"
    css = css_path.read_text()
    # v7 tokens (spec §3)
    for token in [
        "--paper: #e9ecf0", "--panel: #ffffff", "--ink: #18202a", "--hair: #d3d9e0",
        "--live: #0e8f6f", "--alarm: #c73a1d", "--warn-text: #a16207", "--warn-fill: #c07f10",
        "--s-turbine: #2563a8", "--s-pump: #7c4dbc", "--s-phase: #1d8a70",
        "--s-standstill: #6b7684", "--s-transition: #c07f10", "--s-unknown: #aab2bc",
    ]:
        assert token in css, token
    # v7 components exist
    for cls in [".app-bar", ".group-label", ".session-card", ".kpi-band", ".trend-row",
                ".stage-grid", ".register-table", ".transport", ".seg-tabs", ".clip-card",
                ".pills"]:
        assert cls in css, cls
    # legacy look must be gone
    assert "Avenir Next" not in css
    assert "--paper: #eceef0" not in css
    assert ".topbar" not in css


# ---------------------------------------------------------------------------
# 6. site_common v7 chrome helpers -- app_bar_html / group_label_html
# ---------------------------------------------------------------------------


def test_app_bar_html_nav_and_active() -> None:
    import site_common as sc
    html = sc.app_bar_html("live.html", status_html='<span class="rep">● REPLAY</span>')
    assert 'class="app-bar"' in html
    assert html.count("<a ") == 4
    for label in ["Overview", "Sensors", "Live replay", "Audio &amp; review"]:
        assert label in html
    assert 'href="live.html" class="active"' in html
    assert "● REPLAY" in html
    assert "snippets.html" not in html and 'href="review.html"' not in html


def test_group_label_html() -> None:
    import site_common as sc
    html = sc.group_label_html("Session timeline", "full day · click or drag to seek")
    assert 'class="group-label"' in html
    assert "SESSION TIMELINE" in html.upper()
    assert "full day · click or drag to seek" in html


# ---------------------------------------------------------------------------
# 7. build_site v7 page renderers -- render_index / render_redirect_stub
# ---------------------------------------------------------------------------


def test_render_index_v7() -> None:
    import build_site as bs
    html = bs.render_index()
    assert 'class="app-bar"' in html and 'class="group-label"' in html
    assert "audio_review.html" in html
    assert "snippets.html" not in html and 'href="review.html"' not in html
    assert "Research prototype, not a certified product" in html
    assert bs.find_external_resource_urls(html) == []


def test_render_redirect_stub() -> None:
    import build_site as bs
    html = bs.render_redirect_stub("audio_review.html", "Audio & review")
    assert '<meta http-equiv="refresh" content="0; url=audio_review.html">' in html
    assert '<a href="audio_review.html">' in html
    assert bs.find_external_resource_urls(html) == []


# ---------------------------------------------------------------------------
# 8. render_clip_cards -- v7 .clip-card markup for the Hammer strikes / Per-mode
#    audio tabs of the merged audio_review.html page (Task 11)
# ---------------------------------------------------------------------------


def _strike_manifest(**overrides: object) -> dict[str, object]:
    """A minimal `site_manifest.json`-shaped dict -- only the `"strikes"` list
    `render_clip_cards(..., "strikes")` reads (real key names verified against
    `build_site_manifest`/`curate_strike_clips`'s own real output)."""
    clip: dict[str, object] = {
        "event_id": "01", "kind": "plate-gen_0", "session": "pu", "n_strikes": 3,
        "clip_utc": "2026-07-08T12:42:59.568000+00:00", "side": "gen",
        "wav": "strikes/pu-01_plate-gen_0_gen.wav",
    }
    clip.update(overrides)
    return {"strikes": [clip]}


def test_render_clip_cards_strikes_shape_and_units() -> None:
    html = bs.render_clip_cards(_strike_manifest(), "strikes")
    assert 'class="clip-card"' in html
    assert "Reference plate" in html  # _KIND_LABELS lookup, not the raw "plate-gen_0" key
    assert "3 strikes logged" in html
    assert "2026-07-08 12:42:59 UTC" in html
    assert '<audio controls preload="none" src="assets/strikes/pu-01_plate-gen_0_gen.wav">' in html
    assert bs.find_external_resource_urls(html) == []


def test_render_clip_cards_strikes_zero_count_uses_fallback_note() -> None:
    html = bs.render_clip_cards(_strike_manifest(n_strikes=0), "strikes")
    assert "no strike timestamp" in html
    assert "0 strikes logged" not in html


def test_render_clip_cards_strikes_singular_count_grammar() -> None:
    html = bs.render_clip_cards(_strike_manifest(n_strikes=1), "strikes")
    assert "1 strike logged" in html


def test_render_clip_cards_strikes_respects_asset_prefix() -> None:
    html = bs.render_clip_cards(_strike_manifest(), "strikes", asset_prefix="assets/review/")
    assert 'src="assets/review/assets/strikes/pu-01_plate-gen_0_gen.wav"' in html


def _demo_manifest(**overrides: object) -> dict[str, object]:
    """A minimal `docs/demo/assets/manifest.json`-shaped dict (real key names
    verified against the real committed file) -- one `"state"`-kind clip
    (what `render_clip_cards(..., "modes")` selects) plus one `"strike"`-kind
    clip (must be excluded: that family belongs to the Hammer strikes tab)."""
    state_clip: dict[str, object] = {
        "file": "state_cluster0.wav", "kind": "state", "label": "0",
        "start_utc": "2026-07-08T14:19:42.410000+00:00", "duration_s": 10.0,
        "source_run": "080726-pu_strikes",
        "description": "Unsupervised-detected state 0 ...",
    }
    state_clip.update(overrides)
    strike_clip = {
        "file": "strike_pump_plate_tur_0.wav", "kind": "strike", "label": "plate-tur_0",
        "start_utc": "2026-07-08T12:54:00+00:00", "duration_s": 10.0,
        "source_run": "080726-pu_strikes", "description": "Hammer strike ...",
    }
    return {"clips": [state_clip, strike_clip]}


def test_render_clip_cards_modes_selects_state_kind_only() -> None:
    html = bs.render_clip_cards(_demo_manifest(), "modes")
    assert 'class="clip-card"' in html
    assert "State 0" in html
    # Site-local copy (docs/site/assets/modes/), NOT the docs/demo/ sibling
    # directory: a `../demo/assets/...` reference 404s once docs/site/ itself
    # is the served web root (the real local setup, and there is no GitHub
    # Pages workflow implying otherwise) -- RFC 3986 dot-segment removal
    # collapses `..` above the root rather than escaping it.
    assert '<audio controls preload="none" src="assets/modes/state_cluster0.wav">' in html
    assert "../demo/" not in html
    assert "2026-07-08 14:19:42 UTC" in html
    assert "080726-pu_strikes" in html
    assert "plate-tur_0" not in html  # the strike-kind clip must not leak into "modes"
    assert bs.find_external_resource_urls(html) == []


def test_render_clip_cards_modes_respects_asset_prefix() -> None:
    html = bs.render_clip_cards(_demo_manifest(), "modes", asset_prefix="assets/review/")
    assert 'src="assets/review/assets/modes/state_cluster0.wav"' in html
    assert "../demo/" not in html


def test_render_clip_cards_unknown_kind_raises() -> None:
    with pytest.raises(ValueError, match="kind"):
        bs.render_clip_cards({"strikes": [], "clips": []}, "bogus")


# ---------------------------------------------------------------------------
# 9. compose_audio_review_html -- the merged audio_review.html shell (Task 11)
# ---------------------------------------------------------------------------


def test_audio_review_compose() -> None:
    import publish_audio_review as par
    html = par.compose_audio_review_html(
        candidates_fragment=("/*css*/", "<div id='cand'></div>", "/*js*/"),
        strikes_html="<div class='clip-card'>strike</div>",
        modes_html="<div class='clip-card'>mode</div>",
        n_candidates=12, n_strikes=18, n_modes=8,
        sessions=["290626-tu", "080726-pu_strikes"],
    )
    assert 'class="app-bar"' in html
    assert "Flagged candidates" in html and "Hammer strikes" in html and "Per-mode audio" in html
    assert "EXPORT" in html.upper()
    assert "assessments stay in your browser until exported" in html
    import build_site as bs
    assert bs.find_external_resource_urls(html) == []
