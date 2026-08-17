"""Build the public-facing demo site under `docs/site/`: a landing page, an
as-built sensor-geometry schematic, and (curation + card-rendering only --
`scripts/publish_audio_review.py` owns the actual page) the hammer-strike and
per-operating-mode halves of the merged `audio_review.html` listening/review
page (`curate_strike_clips`/`curate_candidate_clips`/`build_site_manifest` for
curation, `render_clip_cards` for markup; the model-flagged-candidate half is
entirely `candidate_kit.py`'s own, never this module's).
Nothing here re-extracts audio from raw data or re-runs any model -- every byte comes
from already-computed local artifacts (`results/annotation-kit/080726`,
`results/candidate-kit`, `docs/demo/assets`), matching this task's own instruction to
reuse those, not rebuild them. The site never makes an external request: every
stylesheet/script/audio/image is inline or a same-repo relative path.

Two subcommands:

    curate-clips   Trim short (`CLIP_DURATION_S` = 10 s) WAV clips from the
                   already-extracted `results/annotation-kit/080726` (Schonhammer
                   strikes, one representative clip per PU-session
                   event -- `curate_strike_clips`) and `results/candidate-kit`
                   (the top `CANDIDATES_PER_CLASS` model-flagged candidates per
                   class, `PINNED_CANDIDATE_IDS` guaranteed included --
                   `select_top_candidates`/`curate_candidate_clips`) registers.
                   NEVER touches `ROWII_DATA_ROOT`/raw Gantner burst files --
                   only already-derived local WAVs already sitting in
                   `results/`, sliced to a short public excerpt and
                   re-peak-normalized (`make_demo_assets._write_clip_wav`,
                   reused, not duplicated). Writes `docs/site/assets/
                   {strikes,candidates}/*.wav` plus `docs/site/assets/
                   site_manifest.json` (the curated selection + every field
                   `build-pages` needs, so that step touches no `results/`
                   data at all).
    build-pages    Renders `docs/site/index.html`/`sensors.html` (v7 markup,
                   no manifest data needed) plus tiny same-repo redirect
                   stubs at `docs/site/snippets.html`/`review.html`
                   pointing to the merged `audio_review.html` page, so old
                   bookmarks to the former Listening Library / Candidate
                   Review pages still land somewhere useful. Touches no
                   `results/` data, so this step alone is fast and safe to
                   rerun after any HTML/CSS wording change. `live.html` is
                   built separately by `scripts/build_live_replay.py`.

Pure helpers (candidate curation/pinning, the two clip-trim-window arithmetic
variants, the as-built sensor-geometry table, and the "did this page make an
external request" scanner) are unit-tested with synthetic fixtures in
`tests/test_build_site.py`, mirroring `tests/test_make_demo_assets.py`'s own
"pure logic gets a unit test, IO-touching code is exercised for real" split.
"""
from __future__ import annotations

import argparse
import csv
import html
import json
import logging
import math
import re
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
from scipy.io import wavfile

SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS_DIR))
import make_demo_assets as mda  # noqa: E402
import site_common as sc  # noqa: E402

REPO_ROOT = SCRIPTS_DIR.parent
DEFAULT_ANNOTATION_KIT_DIR = REPO_ROOT / "results" / "annotation-kit" / "080726"
DEFAULT_GROUNDTRUTH_DIR = REPO_ROOT / "docs" / "groundtruth"
DEFAULT_CANDIDATE_KIT_DIR = REPO_ROOT / "results" / "candidate-kit"
DEFAULT_SITE_DIR = REPO_ROOT / "docs" / "site"
DEFAULT_ASSETS_DIR = DEFAULT_SITE_DIR / "assets"
DEFAULT_MANIFEST_PATH = DEFAULT_ASSETS_DIR / "site_manifest.json"
DEFAULT_DEMO_MANIFEST_PATH = REPO_ROOT / "docs" / "demo" / "assets" / "manifest.json"

CLIP_DURATION_S = 10.0
"""Every curated public clip (strike or candidate) is trimmed to this length --
matches `docs/demo/assets/manifest.json`'s own existing clips exactly, so every
audio control on the site behaves the same way."""
STRIKE_LEAD_PAD_S = 2.0
STRIKE_SESSION = "pu"
"""`results/annotation-kit/080726` has both a `pu` (pump-operation) and an `st`
(standstill) session, each covering the same 13 strike/landmark/vane-sweep kinds.
The curated strike library uses `pu` throughout for one consistent operating
condition across all 13 clips; `docs/demo/assets`' existing 2 standstill clips are
additionally linked in the same section for contrast, not copied again."""
CANDIDATES_PER_CLASS = 5
PINNED_CANDIDATE_IDS = frozenset({"290626-tu-11"})
"""Guaranteed-included candidate id(s) regardless of natural `min_p` ranking --
see `select_top_candidates`."""


# ---------------------------------------------------------------------------
# Pure helpers (no disk I/O) -- unit-tested in tests/test_build_site.py
# ---------------------------------------------------------------------------


def select_top_candidates(
    rows: Sequence[Mapping[str, Any]],
    n_per_class: int,
    pinned_ids: frozenset[str] = frozenset(),
) -> list[Mapping[str, Any]]:
    """The public "N strongest per class" curation rule behind snippets.html's
    unverified-candidates section: within each `class` value present in *rows*
    (e.g. `"sustained"`/`"transient"`), keep the `n_per_class` rows with the
    LOWEST `min_p` (the more extreme/strongest a candidate, the lower its
    minimum scored p-value -- same convention `results/candidate-kit/
    candidates.csv`'s own `rank` column uses), classes returned in
    alphabetical-class-name order and each class's rows sorted ascending by
    `min_p`.

    Every id in *pinned_ids* is GUARANTEED present in the output for its own
    class, even if its `min_p` would not naturally place it in the top
    `n_per_class` -- it displaces that class's weakest natural member (or, if
    more ids are pinned for one class than `n_per_class` allows, every pin
    still survives and that class's output grows past `n_per_class` rather
    than silently dropping a pin the caller explicitly asked for).

    Raises:
        ValueError: if `n_per_class <= 0`, or if any id in *pinned_ids* does
            not appear in *rows* at all (a caller bug -- e.g. a typo'd
            candidate_id -- must fail loudly, not silently omit the pin).
    """
    if n_per_class <= 0:
        raise ValueError(f"n_per_class must be positive, got {n_per_class}")
    by_class: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        by_class.setdefault(str(row["class"]), []).append(row)

    seen_pins: set[str] = set()
    out: list[Mapping[str, Any]] = []
    for klass in sorted(by_class):
        group = by_class[klass]
        pinned = [r for r in group if r["candidate_id"] in pinned_ids]
        pinned_ids_in_class = {r["candidate_id"] for r in pinned}
        seen_pins.update(pinned_ids_in_class)
        rest = sorted(
            (r for r in group if r["candidate_id"] not in pinned_ids_in_class),
            key=lambda r: float(r["min_p"]),
        )
        n_rest = max(0, n_per_class - len(pinned))
        keep = [*pinned, *rest[:n_rest]]
        keep.sort(key=lambda r: float(r["min_p"]))
        out.extend(keep)

    missing = pinned_ids - seen_pins
    if missing:
        raise ValueError(f"pinned candidate id(s) not found in rows: {sorted(missing)}")
    return out


def centered_or_leading_window(
    event_start_s: float,
    event_duration_s: float,
    clip_duration_s: float,
    available_duration_s: float,
) -> float:
    """The trim-window start offset (seconds, relative to the source WAV's own
    start) for a *clip_duration_s*-long public clip covering one candidate-kit
    event `[event_start_s, event_start_s + event_duration_s)` inside an
    already-extracted asset WAV spanning `[0, available_duration_s)`.

    An event at least as long as the clip starts the clip exactly at the
    event (there is no room to add lead-in, and every extra second would just
    cut off more of the anomaly); a shorter event gets a clip CENTERED on it,
    so the surrounding few seconds of normal audio give context on both
    sides. Either way the result is clamped into
    `[0, available_duration_s - clip_duration_s]` so the returned window
    always lies fully inside the asset.

    Raises:
        ValueError: if `available_duration_s < clip_duration_s` -- the asset
            is too short to hold one clip at all, a caller/data bug, not a
            condition to silently paper over.
    """
    if available_duration_s < clip_duration_s:
        raise ValueError(
            f"available_duration_s ({available_duration_s}) is shorter than "
            f"clip_duration_s ({clip_duration_s}) -- cannot fit one clip"
        )
    if event_duration_s >= clip_duration_s:
        raw_start = event_start_s
    else:
        raw_start = event_start_s - (clip_duration_s - event_duration_s) / 2.0
    return min(max(raw_start, 0.0), available_duration_s - clip_duration_s)


def strike_window_start_s(
    strike_offsets_s: Sequence[float],
    clip_duration_s: float,
    available_duration_s: float,
    lead_pad_s: float = STRIKE_LEAD_PAD_S,
) -> float:
    """The trim-window start offset for a *clip_duration_s*-long public clip
    covering one annotation-kit hammer-strike event, anchored just before the
    EARLIEST physical hammer strike in *strike_offsets_s* (seconds, relative
    to the event's own 90 s/210 s snippet WAV) -- *lead_pad_s* of lead-in so
    the strike is audible a moment into the clip rather than at frame zero.
    *strike_offsets_s* need not be sorted (the earliest one always wins).

    An event with NO compiled per-strike timestamps yet (a real, documented
    gap: `docs/groundtruth/080726_strikes_seconds_pu.csv` has zero rows for
    `landmark-A_kugelschieber`/`landmark-C_EG` as of this task) falls back to
    the start of the snippet (offset 0.0) -- the only defensible default
    without a real anchor. Clamped into
    `[0, available_duration_s - clip_duration_s]` exactly like
    `centered_or_leading_window`.

    Raises:
        ValueError: if `available_duration_s < clip_duration_s`.
    """
    if available_duration_s < clip_duration_s:
        raise ValueError(
            f"available_duration_s ({available_duration_s}) is shorter than "
            f"clip_duration_s ({clip_duration_s}) -- cannot fit one clip"
        )
    anchor = (min(strike_offsets_s) - lead_pad_s) if strike_offsets_s else 0.0
    return min(max(anchor, 0.0), available_duration_s - clip_duration_s)


@dataclass(frozen=True)
class SensorPoint:
    """One hoverable marker in the sensors.html as-built schematic."""

    label: str
    group: str
    """`"genmic"` | `"turmic"` | `"genvib"` | `"turvib"`."""
    stream: str
    """The raw Gantner stream this marker's channel(s) belong to."""
    angle_deg: float | None
    """Ring position in degrees, clockwise from top; `None` for the
    off-ring turbine bottom microphone."""
    channels: tuple[str, ...]
    description: str


def sensor_layout() -> list[SensorPoint]:
    """The as-built sensor geometry (9 microphones + 2 tri-axial accelerometer
    pairs) behind sensors.html: a generator-level ring of 4 microphones at 90
    degree spacing, a turbine-level ring of 4 microphones at 90 degree spacing
    plus one additional bottom microphone (9 microphones total, ~50 kHz), and
    one tri-axial accelerometer pair (0/180 degrees) per level (12 channels
    total, ~10 kHz) -- a plain hardware-inventory fact from on-site
    verification, NOT a claim about which raw channel INDEX maps to which
    ring position (that exact channel-to-azimuth correspondence is not
    independently confirmed in any delivered specification; the pipeline
    itself only ever consumes channel 0 of each stream -- README's "Audio
    contract" / `make_demo_assets.MONO_CHANNEL_INDEX`). Labels are
    descriptive POSITION names for this diagram, not verified hardware
    channel-index identifiers.
    """
    points: list[SensorPoint] = []
    for angle in (0, 90, 180, 270):
        points.append(
            SensorPoint(
                label=f"GenMic{angle}",
                group="genmic",
                stream="RAWGeneratorMic__0",
                angle_deg=float(angle),
                channels=(f"GenMic{angle}",),
                description=f"Generator-level microphone ring, {angle}° position.",
            )
        )
    for angle in (0, 90, 180, 270):
        points.append(
            SensorPoint(
                label=f"TurMic{angle}",
                group="turmic",
                stream="RAWTurbineMic__1",
                angle_deg=float(angle),
                channels=(f"TurMic{angle}",),
                description=f"Turbine-level microphone ring, {angle}° position.",
            )
        )
    points.append(
        SensorPoint(
            label="TurMicBottom",
            group="turmic",
            stream="RAWTurbineMic__1",
            angle_deg=None,
            channels=("TurMicBottom",),
            description="Turbine-level microphone, bottom position (off-ring, below the casing).",
        )
    )
    for prefix, stream, group, level in (
        ("GenVib", "RAWGeneratorVib__2", "genvib", "Generator"),
        ("TurVib", "RAWTurbineVib__3", "turvib", "Turbine"),
    ):
        for angle in (0, 180):
            marker = f"{prefix}{angle}"
            points.append(
                SensorPoint(
                    label=marker,
                    group=group,
                    stream=stream,
                    angle_deg=float(angle),
                    channels=(f"{marker}X", f"{marker}Y", f"{marker}Z"),
                    description=(
                        f"{level}-level tri-axial accelerometer, {angle}° position."
                    ),
                )
            )
    return points


_RESOURCE_ATTR_RE = re.compile(
    r'\b(?:src|href|action|data-src)\s*=\s*["\'](https?://[^"\']+)["\']', re.IGNORECASE
)
_CSS_URL_RE = re.compile(r'url\(\s*["\']?(https?://[^"\')]+)["\']?\s*\)', re.IGNORECASE)


def find_external_resource_urls(html_text: str) -> list[str]:
    """Every `http(s)://` URL in *html_text* that sits inside a
    RESOURCE-LOADING attribute (`src`/`href`/`action`/`data-src`) or a CSS
    `url(...)` -- i.e. something a browser would fetch or navigate to, as
    opposed to a URL that only appears as plain visible text (a citation like
    "data courtesy of ... (see https://...)"), which this task's own
    self-containment rule explicitly allows. Used both as this module's own
    pure-logic unit test subject and, for real, against every generated
    docs/site/*.html file as this task's zero-external-requests verification
    (deliberately independent of, and stricter/more automatable than, an ad
    hoc manual grep).
    """
    hits = [m.group(1) for m in _RESOURCE_ATTR_RE.finditer(html_text)]
    hits += [m.group(1) for m in _CSS_URL_RE.finditer(html_text)]
    return hits


# ---------------------------------------------------------------------------
# curate-clips: IO-touching (reads results/, writes docs/site/assets/)
# ---------------------------------------------------------------------------


def _write_trimmed_clip(src_path: Path, dst_path: Path, start_s: float, duration_s: float) -> None:
    """Read *src_path* (an already-extracted, already-16kHz/int16 local WAV --
    NEVER a raw Gantner burst file), slice out `[start_s, start_s+duration_s)`,
    and write it to *dst_path* re-peak-normalized to
    `make_demo_assets.TARGET_DBFS` (mirrors that module's own
    `_write_clip_wav`, reused here rather than duplicated) -- a short slice of
    an already-normalized clip can otherwise sit quieter than its own peak
    suggests.

    Raises:
        ValueError: if *src_path*'s sample rate is not
            `make_demo_assets.TARGET_SAMPLE_RATE_HZ`, or if the requested
            window falls even partially outside its sample range.
    """
    sample_rate_hz, pcm16 = wavfile.read(src_path)
    if sample_rate_hz != mda.TARGET_SAMPLE_RATE_HZ:
        raise ValueError(
            f"{src_path.name} sample rate {sample_rate_hz} Hz != expected "
            f"{mda.TARGET_SAMPLE_RATE_HZ} Hz -- refusing to write a mismatched-rate clip"
        )
    start_idx = round(start_s * sample_rate_hz)
    end_idx = start_idx + round(duration_s * sample_rate_hz)
    if start_idx < 0 or end_idx > len(pcm16):
        raise ValueError(
            f"[{start_s:.3f}, {start_s + duration_s:.3f}) s outside {src_path.name} "
            f"({len(pcm16) / sample_rate_hz:.3f} s available)"
        )
    full_scale = float(-np.iinfo(np.int16).min)
    clip_float = pcm16[start_idx:end_idx].astype(np.float64) / full_scale
    mda._write_clip_wav(dst_path, clip_float)  # noqa: SLF001


def _load_strike_times(groundtruth_dir: Path, session: str) -> dict[str, list[datetime]]:
    """`event_id -> [strike_utc, ...]` from the seconds-level ground truth
    (`docs/groundtruth/080726_strikes_seconds_<session>.csv`), skipping that
    file's leading `#`-prefixed provenance-comment lines. An `event_id` with
    no physical strikes compiled yet is simply absent (never an empty list),
    matching the CSV's own row-per-strike shape."""
    path = groundtruth_dir / f"080726_strikes_seconds_{session}.csv"
    lines = [ln for ln in path.read_text(encoding="utf-8").splitlines() if not ln.startswith("#")]
    by_event: dict[str, list[datetime]] = {}
    for row in csv.DictReader(lines):
        by_event.setdefault(row["event_id"], []).append(datetime.fromisoformat(row["strike_utc"]))
    return by_event


def curate_strike_clips(
    annotation_kit_dir: Path,
    groundtruth_dir: Path,
    out_dir: Path,
    session: str = STRIKE_SESSION,
    clip_duration_s: float = CLIP_DURATION_S,
    lead_pad_s: float = STRIKE_LEAD_PAD_S,
) -> list[dict[str, Any]]:
    """Trim one `clip_duration_s`-long clip per *session* event in
    `<annotation_kit_dir>/events_meta.json` (13 events for the `pu` session:
    the 4 generator-ring + 4 turbine-ring + 1 turbine-bottom hammer strikes,
    3 landmark strikes, and the vane sweep), anchored on that event's real
    strike time(s) via `strike_window_start_s`, into `<out_dir>/*.wav`. Picks
    the turbine-mic recording for `plate-tur_*`/`landmark`/`vane-sweep` kinds
    that mention "tur", the generator-mic recording otherwise (every kind
    string is checked against this project's own fixed vocabulary --
    `tests/test_per_strike.py`'s `_KIND_GROUPS_ALL`).
    """
    events = json.loads((annotation_kit_dir / "events_meta.json").read_text(encoding="utf-8"))
    events = [e for e in events if e["session"] == session]
    strike_times = _load_strike_times(groundtruth_dir, session)
    out_dir.mkdir(parents=True, exist_ok=True)

    clips: list[dict[str, Any]] = []
    for event in sorted(events, key=lambda e: str(e["event_id"])):
        snippet_start = datetime.fromisoformat(event["snippet_start_utc"])
        available_s = float(event["duration_s"])
        offsets_s = [
            (t - snippet_start).total_seconds() for t in strike_times.get(event["event_id"], [])
        ]
        clip_start_s = strike_window_start_s(offsets_s, clip_duration_s, available_s, lead_pad_s)
        side = "tur" if "tur" in event["kind"] else "gen"
        src = annotation_kit_dir / event[f"{side}_wav"]
        dst_name = f"{session}-{event['event_id']}_{event['kind']}_{side}.wav"
        _write_trimmed_clip(src, out_dir / dst_name, clip_start_s, clip_duration_s)
        clips.append(
            {
                "event_id": event["event_id"],
                "kind": event["kind"],
                "session": session,
                "n_strikes": len(offsets_s),
                "clip_utc": (snippet_start + timedelta(seconds=clip_start_s)).isoformat(),
                "side": side,
                "wav": f"strikes/{dst_name}",
            }
        )
    return clips


def load_candidate_rows(candidate_kit_dir: Path) -> list[dict[str, Any]]:
    """`candidates.csv` rows merged with their `candidates_meta.json` sibling
    (asset paths + the human-readable `criterion_text`), keyed by
    `candidate_id`. Meta fields win on overlap (e.g. `min_p`/`class` are
    duplicated across both files; the JSON copy carries real
    float/bool types instead of the CSV's strings)."""
    rows = list(csv.DictReader((candidate_kit_dir / "candidates.csv").open(encoding="utf-8")))
    meta_list = json.loads((candidate_kit_dir / "candidates_meta.json").read_text(encoding="utf-8"))
    meta_by_id = {m["candidate_id"]: m for m in meta_list}
    merged = []
    for row in rows:
        meta = meta_by_id.get(row["candidate_id"])
        if meta is None:
            raise ValueError(f"candidate {row['candidate_id']!r} has no candidates_meta.json entry")
        merged.append({**row, **meta})
    return merged


def curate_candidate_clips(
    candidate_kit_dir: Path,
    out_dir: Path,
    n_per_class: int = CANDIDATES_PER_CLASS,
    pinned_ids: frozenset[str] = PINNED_CANDIDATE_IDS,
    clip_duration_s: float = CLIP_DURATION_S,
) -> list[dict[str, Any]]:
    """`select_top_candidates` over the real candidate register, then trim
    BOTH mic recordings (generator + turbine -- mirrors the two independent
    playback lanes `results/candidate-kit/index.html` itself offers a
    reviewer) of every selected candidate into `<out_dir>/*.wav`, windowed by
    `centered_or_leading_window` against that candidate's own
    `asset_start_utc`/`asset_duration_s` span."""
    rows = load_candidate_rows(candidate_kit_dir)
    selected = select_top_candidates(rows, n_per_class=n_per_class, pinned_ids=pinned_ids)
    out_dir.mkdir(parents=True, exist_ok=True)

    clips: list[dict[str, Any]] = []
    for row in selected:
        asset_start = datetime.fromisoformat(row["asset_start_utc"])
        event_start = datetime.fromisoformat(row["start_utc"])
        event_offset_s = (event_start - asset_start).total_seconds()
        clip_start_s = centered_or_leading_window(
            event_offset_s,
            float(row["duration_s"]),
            clip_duration_s,
            float(row["asset_duration_s"]),
        )
        for side in ("gen", "tur"):
            src = candidate_kit_dir / row[f"{side}_wav"]
            dst_name = f"{row['candidate_id']}_{side}.wav"
            _write_trimmed_clip(src, out_dir / dst_name, clip_start_s, clip_duration_s)
        clips.append(
            {
                "candidate_id": row["candidate_id"],
                "class": row["class"],
                "session": row["session"],
                "start_utc": row["start_utc"],
                "criterion_text": row["criterion_text"],
                "state_name": row["state_name"],
                "scada_state": row["scada_state"],
                "in_sample": bool(row["in_sample"]),
                "gen_wav": f"candidates/{row['candidate_id']}_gen.wav",
                "tur_wav": f"candidates/{row['candidate_id']}_tur.wav",
            }
        )
    return clips


def build_site_manifest(
    annotation_kit_dir: Path = DEFAULT_ANNOTATION_KIT_DIR,
    groundtruth_dir: Path = DEFAULT_GROUNDTRUTH_DIR,
    candidate_kit_dir: Path = DEFAULT_CANDIDATE_KIT_DIR,
    assets_dir: Path = DEFAULT_ASSETS_DIR,
) -> dict[str, Any]:
    """Run both curation passes and write `<assets_dir>/site_manifest.json` --
    the single file `build-pages` needs to render snippets.html without ever
    touching `results/` again."""
    strikes = curate_strike_clips(annotation_kit_dir, groundtruth_dir, assets_dir / "strikes")
    candidates = curate_candidate_clips(candidate_kit_dir, assets_dir / "candidates")
    manifest: dict[str, Any] = {
        "clip_duration_s": CLIP_DURATION_S,
        "strike_session": STRIKE_SESSION,
        "strikes": strikes,
        "candidates": candidates,
    }
    assets_dir.mkdir(parents=True, exist_ok=True)
    (assets_dir / "site_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


# ---------------------------------------------------------------------------
# build-pages: HTML assembly (reads site_manifest.json + docs/demo/assets/
# manifest.json, writes docs/site/*.html; touches no results/ data)
# ---------------------------------------------------------------------------

_KIND_LABELS = {
    "plate-gen_0": "Reference plate — generator ring, 0°",
    "plate-gen_90": "Reference plate — generator ring, 90°",
    "plate-gen_180": "Reference plate — generator ring, 180°",
    "plate-gen_270": "Reference plate — generator ring, 270°",
    "plate-tur_0": "Reference plate — turbine ring, 0°",
    "plate-tur_90": "Reference plate — turbine ring, 90°",
    "plate-tur_180": "Reference plate — turbine ring, 180°",
    "plate-tur_270": "Reference plate — turbine ring, 270°",
    "plate-tur_bottom": "Reference plate — turbine bottom microphone",
    "landmark-A_kugelschieber": "Landmark A — ball valve (Kugelschieber)",
    "landmark-B_11TG": "Landmark B — 11TG",
    "landmark-C_EG": "Landmark C — EG",
    "vane-sweep": "Guide-vane cover sweep (structure-borne excitation)",
}


_NOTICE_HTML = (
    '<div class="notice"><strong>Research prototype, not a certified product.</strong> '
    "Data courtesy of the plant operator, used under a research data-sharing agreement "
    "for an HSG master thesis. Audio is published with the operator's data-release "
    "approval.</div>"
)


def _page_shell(
    *, title: str, active_file: str, body_html: str, extra_css: str = "", wide: bool = False
) -> str:
    """Every `docs/site/*.html` page shares ONE stylesheet link
    (`assets/design.css`, the v7 light-monitoring design system) plus
    `site_common`'s app-bar/footer markup -- `extra_css` is a small,
    page-scoped `<style>` block for layout that genuinely differs per page
    (the sensor rings, the clip grid, ...), never a re-declaration of a
    shared token."""
    page_cls = "page wide" if wide else "page"
    style_tag = f"<style>{extra_css}</style>\n" if extra_css else ""
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(title)}</title>
<link rel="stylesheet" href="assets/design.css">
{style_tag}</head>
<body>
{sc.app_bar_html(active_file)}
<main class="{page_cls}">
{body_html}
</main>
{sc.FOOTER_HTML}
</body>
</html>
"""


_SENSORS_CSS = """
.sensor-layout { display: flex; gap: 22px; align-items: flex-start; flex-wrap: wrap; }
.sensor-diagram { flex: 0 0 auto; background: var(--panel); border: 1px solid var(--hair);
  border-radius: var(--radius); padding: 14px; }
.sensor-diagram svg { width: 300px; height: auto; display: block; }
.sensor rect, .sensor circle, .sensor line { transition: fill .15s ease, stroke .15s ease; }
.sensor.mic circle { fill: var(--live); fill-opacity: .3; stroke: var(--ink); stroke-width: 1;
  cursor: pointer; }
.sensor.mic:hover circle, .sensor.mic:focus circle { fill: var(--live); fill-opacity: .85; }
.sensor.vib rect { fill: var(--panel); stroke: var(--warn-border); stroke-width: 1.1;
  cursor: pointer; }
.sensor.vib line { stroke: var(--warn-border); stroke-width: 1; }
.sensor.vib:hover rect, .sensor.vib:focus rect { fill: var(--warn-fill); fill-opacity: .28; }
.m-block { fill: var(--panel-2); stroke: var(--hair); stroke-width: 1.4; }
.m-shaft { fill: var(--hair-2); }
.m-label { fill: var(--dim); font-size: 10.5px; font-weight: 700; letter-spacing: .03em; }
.sensor-readout { flex: 1 1 260px; background: var(--panel-2); border: 1px solid var(--hair);
  border-radius: var(--radius); padding: 16px 18px; min-height: 84px; font-size: 13.5px;
  color: var(--dim); }
.sensor-readout .rlabel { display: block; color: var(--ink); font-weight: 800; font-size: 15px;
  margin-bottom: 4px; }
.sensor-readout code { display: block; margin-top: 6px; }
table.sensor-table { width: 100%; border-collapse: collapse; margin-top: 22px; font-size: 13px; }
table.sensor-table th, table.sensor-table td { text-align: left; padding: 7px 10px;
  border-bottom: 1px solid var(--hair-2); }
table.sensor-table th { color: var(--dim); font-weight: 700; text-transform: uppercase;
  font-size: 10.5px; letter-spacing: .05em; }
.callout { background: var(--panel); border: 1px solid var(--hair);
  border-left: 3px solid var(--warn-border);
  border-radius: 8px; padding: 12px 16px; margin-top: 22px; font-size: 13.5px; color: var(--dim);
  line-height: 1.55; }
.callout strong { color: var(--ink); }
.plan-rings { display: flex; gap: 22px; flex-wrap: wrap; margin-top: 10px; }
.plan-ring-cell { background: var(--panel); border: 1px solid var(--hair);
  border-radius: var(--radius);
  padding: 12px; }
.plan-ring-cell svg { display: block; }
"""

def _sensor_readout_script() -> str:
    return (
        "<script>"
        'document.querySelectorAll(".sensor").forEach(function(el){'
        'function show(){var r=document.getElementById("sensorReadout");'
        'r.innerHTML="<span class=\\"rlabel\\">"+el.dataset.label+"</span>"+el.dataset.desc'
        '+"<code>"+el.dataset.stream+"</code>";}'
        'el.addEventListener("pointerenter", show);'
        'el.addEventListener("focus", show);'
        "});"
        "</script>"
    )


def _mic_marker_svg(x: float, y: float, p: SensorPoint) -> str:
    lbl = html.escape(p.label)
    desc = html.escape(p.description)
    title = html.escape(f"{p.label} — {p.description} ({p.stream})")
    return (
        f'<g class="sensor mic" tabindex="0" data-label="{lbl}" data-desc="{desc}" '
        f'data-stream="{html.escape(p.stream)}"><title>{title}</title>'
        f'<circle cx="{x:.1f}" cy="{y:.1f}" r="7"/></g>'
    )


def _vib_marker_svg(x: float, y: float, p: SensorPoint) -> str:
    lbl = html.escape(p.label)
    chans = ", ".join(p.channels)
    desc = html.escape(f"{p.description} Channels: {chans}.")
    title = html.escape(f"{p.label} — {p.description} Channels: {chans}. ({p.stream})")
    return (
        f'<g class="sensor vib" tabindex="0" data-label="{lbl}" data-desc="{desc}" '
        f'data-stream="{html.escape(p.stream)}"><title>{title}</title>'
        f'<rect x="{x - 8:.1f}" y="{y - 8:.1f}" width="16" height="16" rx="3"/>'
        f'<line x1="{x:.1f}" y1="{y:.1f}" x2="{x + 11:.1f}" y2="{y - 5:.1f}"/>'
        f'<line x1="{x:.1f}" y1="{y:.1f}" x2="{x - 4:.1f}" y2="{y - 11:.1f}"/>'
        f'<line x1="{x:.1f}" y1="{y:.1f}" x2="{x:.1f}" y2="{y + 11:.1f}"/>'
        "</g>"
    )


def _sensor_svg(points: Sequence[SensorPoint]) -> str:
    """Static (build-time-computed, no runtime layout JS) SVG schematic of the
    as-built sensor geometry: a generator block (top) and a turbine/pump block
    (bottom) on a shared shaft, each ringed by its microphones at their
    labelled angular position (0 degrees = top, clockwise, drawn as a
    shallow ellipse to suggest a plan-view ring), the turbine ring's
    additional bottom microphone drawn distinctly below the casing, and the
    two tri-axial accelerometer pairs as small 3-axis glyphs beside each
    block. Every marker carries a native SVG `<title>` (no-JS hover tooltip +
    screen-reader text) plus `data-*` attributes `_sensor_readout_script`'s
    tiny page script mirrors into the always-visible readout panel.
    """
    width, height = 320, 560
    gen_cx, gen_cy = 160, 128
    tur_cx, tur_cy = 160, 420
    ring_rx, ring_ry = 82, 34

    def ring_xy(cx: float, cy: float, angle_deg: float) -> tuple[float, float]:
        a = math.radians(angle_deg - 90.0)
        return cx + ring_rx * math.cos(a), cy + ring_ry * math.sin(a)

    parts: list[str] = [
        f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="As-built sensor geometry">'
    ]
    parts.append(
        f'<rect class="m-shaft" x="{gen_cx - 9}" y="{gen_cy + 46}" width="18" '
        f'height="{tur_cy - gen_cy - 92}"/>'
    )
    parts.append(
        f'<rect class="m-block" x="{gen_cx - 82}" y="{gen_cy - 46}" '
        f'width="164" height="92" rx="10"/>'
    )
    parts.append(
        f'<text class="m-label" x="{gen_cx}" y="{gen_cy + 4}" '
        f'text-anchor="middle">GENERATOR</text>'
    )
    parts.append(
        f'<rect class="m-block" x="{tur_cx - 74}" y="{tur_cy - 42}" '
        f'width="148" height="84" rx="10"/>'
    )
    parts.append(
        f'<text class="m-label" x="{tur_cx}" y="{tur_cy + 4}" '
        f'text-anchor="middle">TURBINE / PUMP</text>'
    )

    for p in points:
        if p.angle_deg is None:
            continue
        if p.group == "genmic":
            x, y = ring_xy(gen_cx, gen_cy, p.angle_deg)
            parts.append(_mic_marker_svg(x, y, p))
        elif p.group == "turmic":
            x, y = ring_xy(tur_cx, tur_cy, p.angle_deg)
            parts.append(_mic_marker_svg(x, y, p))
    bottom = next(p for p in points if p.label == "TurMicBottom")
    parts.append(_mic_marker_svg(tur_cx, tur_cy + 70, bottom))

    accel_xy = {
        "GenVib0": (gen_cx + 108, gen_cy - 20),
        "GenVib180": (gen_cx - 108, gen_cy - 20),
        "TurVib0": (tur_cx + 100, tur_cy - 18),
        "TurVib180": (tur_cx - 100, tur_cy - 18),
    }
    for p in points:
        if p.label in accel_xy:
            x, y = accel_xy[p.label]
            parts.append(_vib_marker_svg(x, y, p))

    parts.append("</svg>")
    return "".join(parts)


def render_redirect_stub(target: str, title: str) -> str:
    """Tiny self-contained redirect page so old bookmarks keep working."""
    return (
        "<!doctype html>\n<html lang=\"en\">\n<head>\n<meta charset=\"utf-8\">\n"
        f"<title>{title} — ROWII Monitor</title>\n"
        f'<meta http-equiv="refresh" content="0; url={target}">\n'
        "</head>\n<body>\n"
        f'<p>This page moved to <a href="{target}">{target}</a>.</p>\n'
        "</body>\n</html>\n"
    )


def render_index() -> str:
    body = f"""
{sc.group_label_html(
        "ROWII Monitor",
        "acoustic condition monitoring · Rodundwerk II · measurement campaign 2026",
    )}
<p class="lede">ROWII Monitor is a research-prototype condition-monitoring pipeline for
the Rodundwerk&nbsp;II pump-turbine, built as part of an HSG master thesis. From nine
microphones and two tri-axial accelerometers it detects the machine's operating state
second by second &mdash; unsupervised, validated against SCADA-derived ground truth
&mdash; and flags one-second windows that look unusual for the current mode, using a
calibrate-once-per-instrumentation-era design with a label-free drift sentinel. This
site replays four recorded sessions end to end: where the sensors sit on the machine,
a control-room replay of state, score and alarms, a library of real audio clips, and
the candidate-review tool used to hand model-flagged windows to a plant expert.</p>
{_NOTICE_HTML}
<div class="cards">
  <a class="card" href="sensors.html">
    <h2>Sensors</h2>
    <p>The as-built microphone and accelerometer layout on the machine &mdash; vertical
    section and plan-view rings &mdash; with every stream/channel name on hover.</p>
    <span class="go">Open the sensor map &rarr;</span>
  </a>
  <a class="card" href="live.html">
    <h2>Live replay</h2>
    <p>Replays of four recorded sessions — operating state, features, sentinel and
    alarms in a control-room view.</p>
    <span class="go">Open the control room &rarr;</span>
  </a>
  <a class="card" href="audio_review.html">
    <h2>Audio &amp; review</h2>
    <p>Hammer-strike tests, per-mode audio, and every model-flagged candidate with SCADA
    context and the expert assessment form.</p>
    <span class="go">Open audio &amp; review &rarr;</span>
  </a>
</div>
"""
    return _page_shell(title="ROWII Monitor", active_file="index.html", body_html=body)


def render_sensors() -> str:
    points = sensor_layout()
    rows = []
    for p in points:
        angle = f"{p.angle_deg:g}°" if p.angle_deg is not None else "bottom (off-ring)"
        rows.append(
            f"<tr><td>{html.escape(p.label)}</td><td>{html.escape(p.stream)}</td>"
            f"<td>{angle}</td><td>{html.escape(', '.join(p.channels))}</td>"
            f"<td>{html.escape(p.description)}</td></tr>"
        )
    body = f"""
{sc.group_label_html("Sensor layout")}
<p class="lede">The as-built acoustic and vibration sensor geometry on the machine set, in two
views: a vertical section (how the sensors sit on the machine) and a plan view
(looking straight down each ring). Hover or keyboard-focus any marker for its
stream/channel name; the full table is below for reference.</p>

{sc.group_label_html("Vertical section")}
<div class="panel panel-pad">
  <div class="sensor-layout">
    <div class="sensor-diagram">{_sensor_svg(points)}</div>
    <div class="sensor-readout" id="sensorReadout">Hover or focus a sensor marker to see
    its name here.</div>
  </div>
</div>

{sc.group_label_html("Plan view rings")}
<div class="panel panel-pad">
  <p>Each ring drawn from above: the same 4&times;90&deg; microphone spacing (plus the
  turbine ring's off-ring bottom microphone) and the two tri-axial accelerometers per
  level. Markers share the same hover readout as the section view above.</p>
  {sc.render_plan_rings_block(size=210, interactive=True)}
</div>

<table class="sensor-table">
  <thead><tr><th>Label</th><th>Stream</th><th>Ring position</th>
  <th>Channel(s)</th><th>Description</th></tr></thead>
  <tbody>{"".join(rows)}</tbody>
</table>
<div class="callout">
  <strong>As-built vs. planned.</strong> An earlier planning draft specified a different
  microphone-ring layout (5 positions per ring, 72&deg; spacing). The geometry shown
  here is the as-installed configuration verified on site: 4 microphones at 90&deg;
  spacing on the generator ring, 4 at 90&deg; spacing plus one additional bottom
  microphone on the turbine ring, and one tri-axial accelerometer pair (0&deg;/180&deg;)
  per level. The discrepancy between the planned and as-built ring layout is not
  reconciled in any delivered specification.
</div>
{_sensor_readout_script()}
"""
    return _page_shell(
        title="Sensors — ROWII Monitor", active_file="sensors.html", body_html=body,
        extra_css=_SENSORS_CSS,
    )


def _format_clip_timestamp(iso_utc: str) -> str:
    """`"2026-07-08T14:19:42.410000+00:00"` -> `"2026-07-08 14:19:42 UTC"` -- a
    clip-card `.meta` line (v7 spec D7: units on every number, a timestamp's own
    "unit" being its zone), not the raw ISO offset the source manifests carry."""
    return datetime.fromisoformat(iso_utc).strftime("%Y-%m-%d %H:%M:%S") + " UTC"


def _strike_clip_card(clip: Mapping[str, Any], asset_prefix: str = "") -> str:
    """v7 `.clip-card` (design.css: `.ct`/`.n2` title row, `<audio>`, `.meta` --
    replaces the pre-v7 `.clip-head`/`.clip-title`/`.clip-tag`/`.clip-note`
    markup `_SNIPPETS_CSS` used to style). `asset_prefix` mirrors `candidate_kit.
    _prefixed_metas`'s convention (prepended to the already-relative `assets/...`
    path); empty by default since `build_site_manifest` already writes strike
    clips straight into `<site>/assets/strikes/`, needing no relocation for
    `docs/site/audio_review.html` itself."""
    label = _KIND_LABELS.get(clip["kind"], clip["kind"])
    n = int(clip["n_strikes"])
    strikes_note = (
        f"{n} strike{'s' if n != 1 else ''} logged"
        if n
        else "no strike timestamp yet (clip starts at the top of the logged minute)"
    )
    return f"""<div class="clip-card">
  <div class="ct"><span class="n2">{html.escape(label)}</span>
  <span class="badge">{html.escape(clip['session'].upper())}</span></div>
  <audio controls preload="none" src="{asset_prefix}assets/{html.escape(clip['wav'])}"></audio>
  <p class="meta">{strikes_note} &middot; {_format_clip_timestamp(clip['clip_utc'])}</p>
</div>"""


def _demo_clip_card(clip: Mapping[str, Any], asset_prefix: str = "") -> str:
    """v7 `.clip-card` for one `docs/demo/assets/manifest.json` `"state"`-kind
    clip (ordinary per-operating-mode audio, `render_clip_cards(..., "modes")`'s
    only source). Sourced from `<site>/assets/modes/` -- a site-LOCAL, tracked
    copy `publish_audio_review.py`'s `publish()` makes of the 3 real `"state"`-
    kind WAVs, NOT a `../demo/assets/` sibling-directory reference: the real
    local dev setup serves `docs/site/` itself as the web root (`npx serve -l
    5173 repos/rowii-monitor/docs/site`; there is no GitHub Pages workflow in
    this repo implying a `docs/`-rooted deployment instead), and a `..` above
    that root collapses away under standard relative-URL resolution (RFC 3986
    S5.2.4) rather than escaping it -- so `../demo/assets/*.wav` 404s for real,
    confirmed against a `docs/site/`-rooted server. `asset_prefix` mirrors
    `_strike_clip_card`'s own convention (prepended to the already-relative
    `assets/modes/...` path); empty by default, same reasoning as strikes."""
    src = f"{asset_prefix}assets/modes/{html.escape(clip['file'])}"
    return f"""<div class="clip-card">
  <div class="ct"><span class="n2">State {html.escape(str(clip['label']))}</span>
  <span class="badge">unsupervised cluster</span></div>
  <audio controls preload="none" src="{src}"></audio>
  <p class="meta">{_format_clip_timestamp(clip['start_utc'])}
  &middot; {html.escape(clip['source_run'])}</p>
</div>"""


def render_clip_cards(manifest: Mapping[str, Any], kind: str, asset_prefix: str = "") -> str:
    """Concatenated v7 `.clip-card` markup for one of `publish_audio_review.py`'s
    two non-candidate `audio_review.html` tabs -- `kind`:

    * `"strikes"` (Hammer strikes tab): *manifest* is `site_manifest.json`
      (`build_site_manifest`), mapped over its `"strikes"` list
      (`curate_strike_clips`) via `_strike_clip_card`.
    * `"modes"` (Per-mode audio tab): *manifest* is `docs/demo/assets/
      manifest.json` (`make_demo_assets`'s own curated library), filtered to
      `"kind": "state"` entries and mapped via `_demo_clip_card` -- ordinary
      audio from each detected operating mode, distinct from (and never
      touching) `manifest["candidates"]`/`candidates_meta.json`: that curated
      top-N-per-class preview role (the old `_candidate_clip_card`) is fully
      superseded by the FULL, interactive candidate register the "Flagged
      candidates" tab renders via `candidate_kit.render_candidates_fragment`.

    Supersedes the old `render_snippets`/`_candidate_clip_card`/
    `_demo_strike_clip_card`/`_CANDIDATE_CONTEXT_NOTES` (deleted: zero other
    callers, confirmed by repo-wide grep; the one candidate-specific research
    cross-reference they carried, id `290626-tu-11`, is preserved -- more
    precisely, since it is applied programmatically rather than hand-curated
    per rendered page -- by `candidate_kit.CONTEXT_NOTES`/`apply_context_notes`,
    which already flows into every candidate's own `context_note` field and is
    rendered by `render_candidates_fragment`'s fragment for every card, this
    one included). `_SNIPPETS_CSS` (their only stylesheet, with its own dangling
    `--radius-lg`/`--warn` custom-property references -- design.css defines
    neither) is deleted for the same reason: this function's markup uses only
    design.css's own already-v7 `.clip-card`/`.ct`/`.n2`/`.badge`/`.meta`
    classes, needing no page-specific CSS at all.
    """
    if kind == "strikes":
        return "".join(_strike_clip_card(c, asset_prefix) for c in manifest["strikes"])
    if kind == "modes":
        return "".join(
            _demo_clip_card(c, asset_prefix) for c in manifest["clips"] if c["kind"] == "state"
        )
    raise ValueError(f"render_clip_cards: unknown kind {kind!r} (expected 'strikes' or 'modes')")


def build_pages(
    assets_dir: Path = DEFAULT_ASSETS_DIR,
    manifest_path: Path = DEFAULT_MANIFEST_PATH,
    demo_manifest_path: Path = DEFAULT_DEMO_MANIFEST_PATH,
    site_dir: Path = DEFAULT_SITE_DIR,
) -> list[Path]:
    """Renders `index.html`/`sensors.html` (v7 markup, needs no manifest
    data) plus tiny same-repo redirect stubs at `snippets.html`/
    `review.html` pointing to `audio_review.html` -- the v7 redesign merges
    the old Listening Library (`snippets.html`) and Candidate Review
    (`review.html`) pages into that one page, so old bookmarks to either
    former page still land somewhere useful. `audio_review.html` itself is
    not one of this function's outputs -- `scripts/publish_audio_review.py`
    (which imports this module's `render_clip_cards` plus `candidate_kit.
    render_candidates_fragment`) is its ONE writer. `manifest_path`/
    `demo_manifest_path` are accepted (still `curate-clips`'s own output
    paths) but no longer read here now that nothing in THIS function's own
    output needs manifest data.

    `live.html` is deliberately NOT one of this function's outputs -- it is
    a full native control-room replay with its own real-data precompute
    step, owned by `scripts/build_live_replay.py` (this script's
    `curate-clips`/`build-pages` split, "touches no results/ data" contract,
    does not fit a per-window parquet/cache read). The `review.html` stub
    written here IS the real, final content at that path (unlike an earlier
    site-redesign stage, `scripts/publish_audio_review.py` no longer writes
    `docs/site/review.html`/`review_static.html` at all -- this function is
    now their only writer anywhere in `scripts/`); `review_static.html`
    itself has no writer left after that removal and is simply never
    produced again.
    """
    pages = {
        "index.html": render_index(),
        "sensors.html": render_sensors(),
        "snippets.html": render_redirect_stub("audio_review.html", "Listening library"),
        "review.html": render_redirect_stub("audio_review.html", "Candidate review"),
    }
    site_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for name, content in pages.items():
        out_path = site_dir / name
        out_path.write_text(content, encoding="utf-8")
        written.append(out_path)
    return written


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build the public-facing demo site under docs/site/: curate short audio "
            "clips from results/annotation-kit + results/candidate-kit (curate-clips), "
            "then render the four site pages (build-pages)."
        )
    )
    sub = parser.add_subparsers(dest="command", required=True)

    curate = sub.add_parser(
        "curate-clips",
        help="Trim curated strike/candidate WAV clips and write site_manifest.json.",
    )
    curate.add_argument("--annotation-kit-dir", type=Path, default=DEFAULT_ANNOTATION_KIT_DIR)
    curate.add_argument("--groundtruth-dir", type=Path, default=DEFAULT_GROUNDTRUTH_DIR)
    curate.add_argument("--candidate-kit-dir", type=Path, default=DEFAULT_CANDIDATE_KIT_DIR)
    curate.add_argument("--assets-dir", type=Path, default=DEFAULT_ASSETS_DIR)

    build = sub.add_parser("build-pages", help="Render the four docs/site/*.html pages.")
    build.add_argument("--assets-dir", type=Path, default=DEFAULT_ASSETS_DIR)
    build.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST_PATH)
    build.add_argument("--demo-manifest", type=Path, default=DEFAULT_DEMO_MANIFEST_PATH)
    build.add_argument("--site-dir", type=Path, default=DEFAULT_SITE_DIR)

    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "curate-clips":
        manifest = build_site_manifest(
            args.annotation_kit_dir, args.groundtruth_dir, args.candidate_kit_dir, args.assets_dir
        )
        n_clips = len(manifest["strikes"]) + 2 * len(manifest["candidates"])
        print(f"build_site: wrote {n_clips} clip(s) + site_manifest.json to {args.assets_dir}")
        return 0

    written = build_pages(args.assets_dir, args.manifest, args.demo_manifest, args.site_dir)
    print(f"build_site: wrote {len(written)} page(s) to {args.site_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
