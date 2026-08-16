"""Shared building blocks for the `docs/site/` v2 rebuild: design tokens (kept in
sync BY VALUE with `docs/site/assets/design.css` -- there is no build step that
derives one from the other, so a token changed in one place must be changed in the
other by hand) and the plan-view ring SVG generator used by BOTH `sensors.html`
(`scripts/build_sensor_map.py`) and `live.html`'s compact sensor panel
(`scripts/build_live_replay.py`) -- one function, so the two pages draw the
IDENTICAL ring geometry rather than two hand-maintained copies drifting apart.

Ring geometry (as-built, NOT the earlier planning draft's 5x72 deg layout -- see
`scripts/build_site.py`'s own `SensorPoint` docstring for the same as-built-vs-
planned caveat this module repeats): 4 microphones at 0/90/180/270 deg on each of
the generator and turbine rings, PLUS one additional turbine microphone
(`TurMicBottom`) off-ring, below the casing -- verified on site, not the delivered
spec. Each ring's 4 (+1) markers are wired to ONE `data-stream` group
(`RAWGeneratorMic__0` / `RAWTurbineMic__1`) rather than individually addressable
IDs: the exact physical-position <-> DAQ-channel-index correspondence for the
multi-channel ring recordings was never verified anywhere in this codebase
(`scripts/build_site.py`'s own `SensorPoint` docstring says so explicitly), and the
scored pipeline itself only ever consumes channel 0 of each stream -- so a live
per-position level would silently overclaim a mapping nobody checked. What IS
real and shown here: one live RMS level PER STREAM, applied uniformly to every
marker on that stream's own ring (`live.html`'s own task instruction says
"per-stream RMS levels", not per-position).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Design tokens -- BY VALUE identical to docs/site/assets/design.css's :root.
# ---------------------------------------------------------------------------

PAPER = "#eceef0"
PANEL = "#ffffff"
PANEL_2 = "#f7f8f9"
INK = "#1f2a37"
DIM = "#5b6b7c"
HAIR = "#c7cdd4"
HAIR_2 = "#dde1e6"

LIVE = "#0f766e"
ALARM = "#b93815"
WARN = "#a16207"

STATE_COLORS: dict[str, str] = {
    "turbine": "#2563a8",
    "pump": "#7c4dbc",
    "phase-shifter": "#1d8a70",
    "standstill": "#6b7684",
    "transition": "#c07f10",
    "unknown": "#aab2bc",
}
"""Fixed per-state colors -- the ONE shared vocabulary used everywhere a state is
drawn (ring legend, state chip, ribbon, timeline). Never repurposed for anything
that is not literally one of these five detector/SCADA states."""

STATE_DISPLAY_NAME: dict[str, str] = {
    "turbine": "Turbine",
    "pump": "Pump",
    "phase-shifter": "Phase-shifter",
    "standstill": "Standstill",
    "transition": "Transition",
    "unknown": "Unknown",
}

FONT_UI = '"Avenir Next","Helvetica Neue",Helvetica,Arial,sans-serif'
FONT_MONO = '"SF Mono",ui-monospace,"Cascadia Mono",Consolas,monospace'


def state_color(name: str) -> str:
    return STATE_COLORS.get(name, STATE_COLORS["unknown"])


# ---------------------------------------------------------------------------
# Shared page chrome (topbar nav + footer) -- one Python source, every generated
# page (sensors.html, live.html) calls this so nav markup never drifts between
# hand-written pages (index.html, snippets.html, review.html) and generated ones.
# ---------------------------------------------------------------------------

NAV_ITEMS: tuple[tuple[str, str], ...] = (
    ("index.html", "Overview"),
    ("sensors.html", "Sensors"),
    ("live.html", "Live Replay"),
    ("snippets.html", "Listening Library"),
    ("review.html", "Candidate Review"),
)


def topbar_html(active_href: str) -> str:
    links = "\n    ".join(
        f'<a href="{href}"{" class=\"active\"" if href == active_href else ""}>{label}</a>'
        for href, label in NAV_ITEMS
    )
    return (
        '<header class="topbar">\n'
        '  <div class="brand"><span class="brand-mark">ROWII</span>'
        '<span class="brand-sub">Monitor</span></div>\n'
        f'  <nav class="nav">\n    {links}\n  </nav>\n'
        "</header>"
    )


FOOTER_HTML = (
    '<footer class="site-footer">Krummenacher, 2026, University of St. Gallen &mdash; see '
    "README.md and CITATION.cff for the full citation. Sensor recordings are proprietary "
    "plant data and are not redistributed (DATA_ACCESS.md).</footer>"
)


# ---------------------------------------------------------------------------
# Plan-view ring SVG
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RingMarker:
    label: str
    angle_deg: float | None
    """Clock convention: 0=top, 90=right, 180=bottom, 270=left. `None` for the
    turbine's off-ring bottom microphone (drawn outside the ring, below it, not
    at a ring angle -- it is not on the ring, module docstring)."""
    kind: str
    """`"mic"` or `"vib"`."""
    stream: str
    desc: str


GENERATOR_MARKERS: tuple[RingMarker, ...] = (
    RingMarker("GenMic0", 0, "mic", "RAWGeneratorMic__0", "Generator microphone ring, 0 deg."),
    RingMarker("GenMic90", 90, "mic", "RAWGeneratorMic__0", "Generator microphone ring, 90 deg."),
    RingMarker("GenMic180", 180, "mic", "RAWGeneratorMic__0", "Generator microphone ring, 180 deg."),
    RingMarker("GenMic270", 270, "mic", "RAWGeneratorMic__0", "Generator microphone ring, 270 deg."),
    RingMarker("GenVib0", 0, "vib", "RAWGeneratorVib__2", "Generator tri-axial accelerometer, 0 deg."),
    RingMarker(
        "GenVib180", 180, "vib", "RAWGeneratorVib__2", "Generator tri-axial accelerometer, 180 deg."
    ),
)

TURBINE_MARKERS: tuple[RingMarker, ...] = (
    RingMarker("TurMic0", 0, "mic", "RAWTurbineMic__1", "Turbine microphone ring, 0 deg."),
    RingMarker("TurMic90", 90, "mic", "RAWTurbineMic__1", "Turbine microphone ring, 90 deg."),
    RingMarker("TurMic180", 180, "mic", "RAWTurbineMic__1", "Turbine microphone ring, 180 deg."),
    RingMarker("TurMic270", 270, "mic", "RAWTurbineMic__1", "Turbine microphone ring, 270 deg."),
    RingMarker(
        "TurMicBottom", None, "mic", "RAWTurbineMic__1",
        "Turbine microphone, bottom position (off-ring, below the casing).",
    ),
    RingMarker("TurVib0", 0, "vib", "RAWTurbineVib__3", "Turbine tri-axial accelerometer, 0 deg."),
    RingMarker(
        "TurVib180", 180, "vib", "RAWTurbineVib__3", "Turbine tri-axial accelerometer, 180 deg."
    ),
)


def _polar(cx: float, cy: float, r: float, angle_deg: float) -> tuple[float, float]:
    theta = math.radians(angle_deg)
    return cx + r * math.sin(theta), cy - r * math.cos(theta)


def render_ring_svg(
    title: str, markers: tuple[RingMarker, ...], *, size: int = 200, interactive: bool = True
) -> str:
    """One plan-view ring (top-down): a casing circle, mic dots at their ring
    angle, vib squares (tri-axial tick marks) at theirs, and (turbine only) the
    off-ring bottom mic drawn outside the circle. `interactive=True` adds
    hover/focus affordances (`sensors.html`); `interactive=False` renders the
    same markup without `tabindex`/`title` (the compact `live.html` panel, where
    hover detail is not the point -- level pulsing is, via `data-stream`).
    """
    cx = cy = size / 2
    r_ring = size * 0.30
    r_mic = r_ring
    r_vib = r_ring * 1.28
    r_bottom = r_ring * 1.5

    parts: list[str] = [
        f'<svg viewBox="0 0 {size} {size}" width="{size}" height="{size}" '
        f'role="img" aria-label="{title} plan view">',
        f'<circle cx="{cx}" cy="{cy}" r="{r_ring:.1f}" fill="{PANEL_2}" stroke="{HAIR}" '
        f'stroke-width="1.3"/>',
        f'<circle cx="{cx}" cy="{cy}" r="{r_ring * 0.42:.1f}" fill="none" stroke="{HAIR}" '
        f'stroke-width="1" stroke-dasharray="2 3"/>',
        f'<text x="{cx}" y="{cy + 3.5}" text-anchor="middle" font-size="9.5" '
        f'font-weight="700" fill="{DIM}" font-family="{FONT_UI}">{title}</text>',
    ]

    by_stream: dict[str, list[RingMarker]] = {}
    for m in markers:
        by_stream.setdefault(m.stream, []).append(m)

    for stream, group in by_stream.items():
        parts.append(f'<g class="ring-group" data-stream="{stream}">')
        for m in group:
            # Class names deliberately match `build_site.py`'s existing vertical-section
            # markers (`"sensor mic"` / `"sensor vib"`) -- ONE shared CSS ruleset + ONE
            # shared `_sensor_readout_script()` hover handler then drives BOTH the
            # elevation view and these plan-view rings, rather than a second copy.
            hover_attrs = (
                f' tabindex="0" data-label="{m.label}" data-desc="{m.desc}" '
                f'data-stream="{m.stream}"><title>{m.label} — {m.desc} ({m.stream})</title>'
                if interactive
                else ">"
            )
            if m.kind == "mic":
                if m.angle_deg is None:
                    x, y = cx, cy + r_bottom
                else:
                    x, y = _polar(cx, cy, r_mic, m.angle_deg)
                parts.append(
                    f'<g class="sensor mic ring-marker"{hover_attrs}'
                    f'<circle class="mic-dot" cx="{x:.1f}" cy="{y:.1f}" r="5.6" '
                    f'fill="{LIVE}" fill-opacity="0.34" stroke="{INK}" stroke-width="1"/>'
                    "</g>"
                )
            else:
                x, y = _polar(cx, cy, r_vib, m.angle_deg or 0)
                parts.append(
                    f'<g class="sensor vib ring-marker"{hover_attrs}'
                    f'<rect class="vib-dot" x="{x - 4.5:.1f}" y="{y - 4.5:.1f}" width="9" '
                    f'height="9" rx="2" fill="{PANEL}" fill-opacity="0.9" stroke="{WARN}" '
                    f'stroke-width="1.2"/>'
                    "</g>"
                )
        parts.append("</g>")

    parts.append("</svg>")
    return "".join(parts)


def render_plan_rings_block(*, size: int = 200, interactive: bool = True) -> str:
    """Both rings side by side, wrapped for the caller's own layout CSS
    (`.plan-rings` / `.plan-ring-cell`)."""
    gen_svg = render_ring_svg("GENERATOR", GENERATOR_MARKERS, size=size, interactive=interactive)
    tur_svg = render_ring_svg("TURBINE", TURBINE_MARKERS, size=size, interactive=interactive)
    return (
        '<div class="plan-rings">\n'
        f'  <div class="plan-ring-cell">{gen_svg}</div>\n'
        f'  <div class="plan-ring-cell">{tur_svg}</div>\n'
        "</div>"
    )
