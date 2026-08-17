"""Publish the merged Audio & review page (`docs/site/audio_review.html`) --
the v7 site redesign's single page for every audio artifact the site offers,
tabbed behind one shared app-bar/toolbar shell: the full model-flagged
candidate register (`candidate_kit.render_candidates_fragment`'s interactive
fragment -- SCADA context, click-to-seek, the expert assessment form), the
hammer-strike test library, and per-operating-mode listening clips (both via
`build_site.render_clip_cards`).

Renamed from `scripts/publish_review_site.py` (site redesign plan Task 11):
that script rendered `docs/site/review.html`/`review_static.html` from
`candidate_kit.py` alone. This one composes THREE data sources into one page
and stops producing `review.html`/`review_static.html` entirely --
`build_site.build_pages`'s own same-repo redirect stub is the ONLY thing that
still writes `docs/site/review.html` (old bookmarks land on `audio_review.
html`); `review_static.html` has no writer left at all after this change.

Steps:
    1. `candidate_kit.build_all` exactly as `candidate_kit.py build` itself
       would (same `out_dir`, unprefixed asset paths) -- reuses already-
       rendered WAV/PNG assets via that function's own existing reuse path,
       so a rerun after this task's edits is fast (nothing here re-extracts
       audio). This also (re)writes the standalone `results/candidate-kit/
       index.html`/`index_static.html`, exactly as `candidate_kit.py build`
       always has.
    2. Copy every candidate's per-session asset directory from `results/
       candidate-kit/<session>/` into `docs/site/assets/review/<session>/` (a
       TRACKED, committed location -- `results/` itself is gitignored, so the
       site's own copy is the only one that ever reaches git).
    3. Render the candidates fragment (`candidate_kit.render_candidates_
       fragment`, `asset_prefix=ASSET_PREFIX` so its embedded paths point at
       step 2's copy), the hammer-strikes clip cards (`build_site.
       render_clip_cards` over `docs/site/assets/site_manifest.json`'s own
       `"strikes"` list), and the per-mode clip cards (same function over
       `docs/demo/assets/manifest.json`'s `"state"`-kind clips).
    4. Compose (`compose_audio_review_html`) and write
       `docs/site/audio_review.html`.

Run from the repo root: `python scripts/publish_audio_review.py`.
"""
from __future__ import annotations

import json
import logging
import shutil
import sys
from collections.abc import Sequence
from pathlib import Path

_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

import build_site as bs  # noqa: E402
import candidate_kit as ck  # noqa: E402
import site_common as sc  # noqa: E402

from rowii.config import load_config  # noqa: E402

logger = logging.getLogger(__name__)

REPO_ROOT = _SCRIPTS_DIR.parent
SITE_DIR = REPO_ROOT / "docs" / "site"
SITE_ASSETS_DIR = SITE_DIR / "assets" / "review"
ASSET_PREFIX = "assets/review/"


_TABS_JS = r"""
(function () {
"use strict";

document.querySelectorAll(".seg-tabs button[data-tab]").forEach(function (btn) {
  btn.addEventListener("click", function () {
    var tab = btn.getAttribute("data-tab");
    document.querySelectorAll(".seg-tabs button[data-tab]").forEach(function (b) {
      b.classList.toggle("active", b === btn);
    });
    document.querySelectorAll("[data-tab-panel]").forEach(function (panel) {
      panel.hidden = panel.getAttribute("data-tab-panel") !== tab;
    });
  });
});

document.querySelectorAll(".pills .pill[data-session]").forEach(function (btn) {
  btn.addEventListener("click", function () {
    var session = btn.getAttribute("data-session");
    document.querySelectorAll(".pills .pill").forEach(function (b) {
      b.classList.toggle("active", b === btn);
    });
    var panel = document.querySelector('[data-tab-panel="candidates"]');
    if (panel) panel.setAttribute("data-session-filter", session);
    document.querySelectorAll(".candidate-card[data-session]").forEach(function (card) {
      var show = session === "all" || card.getAttribute("data-session") === session;
      card.hidden = !show;
    });
  });
});

// Progress counter: candidates with a non-empty stored assessment, over the
// total the kit fragment actually built (window.CandidateKit.cards, exposed
// by _CANDIDATE_JS -- enumeration reuses that existing registry rather than
// re-parsing the DOM; the assessment check itself reads localStorage
// directly via the kit's own storage-key prefix, one source of truth, no
// re-implementation of loadState's own parsing).
function updateProgress() {
  var note = document.getElementById("reviewProgress");
  var kit = window.CandidateKit;
  if (!note || !kit || !kit.cards) return;
  var ids = Object.keys(kit.cards);
  var assessed = 0;
  ids.forEach(function (id) {
    var raw = null;
    try {
      raw = localStorage.getItem("__STORAGE_PREFIX__" + id);
    } catch (e) {
      raw = null;
    }
    if (!raw) return;
    try {
      var parsed = JSON.parse(raw);
      if (parsed && typeof parsed.assessment === "string" && parsed.assessment !== "") {
        assessed++;
      }
    } catch (e) {
      /* malformed localStorage entry -- not counted as assessed */
    }
  });
  note.textContent = assessed + " / " + ids.length + " candidates assessed";
}
updateProgress();
document.addEventListener("change", function (e) {
  if (e.target && e.target.matches && e.target.matches('.assessment-options input[type="radio"]')) {
    updateProgress();
  }
});

// EXPORT delegates to the kit fragment's own existing export control -- no
// re-implementation of exportCsv/cardsToCsv.
var exportBtn = document.getElementById("exportBtn");
var kitExportBtn = document.getElementById("export-all-btn");
if (exportBtn && kitExportBtn) {
  exportBtn.addEventListener("click", function () {
    kitExportBtn.click();
  });
}
})();
"""


def compose_audio_review_html(
    *,
    candidates_fragment: tuple[str, str, str],
    strikes_html: str,
    modes_html: str,
    n_candidates: int,
    n_strikes: int,
    n_modes: int,
    sessions: Sequence[str],
) -> str:
    css, cand_body, cand_js = candidates_fragment
    session_pills = "".join(
        f'<button type="button" class="pill" data-session="{s}">{s}</button>' for s in sessions
    )
    tabs_js = _TABS_JS.replace("__STORAGE_PREFIX__", ck.STORAGE_PREFIX)
    return (
        "<!doctype html>\n<html lang=\"en\">\n<head>\n<meta charset=\"utf-8\">\n"
        '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
        "<title>Audio &amp; review — ROWII Monitor</title>\n"
        '<link rel="stylesheet" href="assets/design.css">\n'
        f"<style>{css}</style>\n</head>\n<body>\n"
        + sc.app_bar_html("audio_review.html")
        + '<main class="page">'
        + sc.group_label_html("Audio library &amp; expert review",
                              "assessments stay in your browser until exported")
        + '<div class="toolbar">'
        + '<div class="seg-tabs">'
        + (
            '<button type="button" class="active" data-tab="candidates">Flagged candidates'
            f'<span class="n">{n_candidates}</span></button>'
        )
        + (
            '<button type="button" data-tab="strikes">Hammer strikes'
            f'<span class="n">{n_strikes}</span></button>'
        )
        + (
            '<button type="button" data-tab="modes">Per-mode audio'
            f'<span class="n">{n_modes}</span></button>'
        )
        + "</div>"
        + (
            '<div class="pills"><button type="button" class="pill active" '
            f'data-session="all">all sessions</button>{session_pills}</div>'
        )
        + '<div class="right"><span class="progress-note" id="reviewProgress">—</span>'
        + '<button type="button" class="btn-export" id="exportBtn">EXPORT (.csv)</button></div>'
        + "</div>"
        + f'<section data-tab-panel="candidates">{cand_body}</section>'
        + (
            '<section data-tab-panel="strikes" hidden><div class="clip-grid">'
            f'{strikes_html}</div></section>'
        )
        + (
            '<section data-tab-panel="modes" hidden><div class="clip-grid">'
            f'{modes_html}</div></section>'
        )
        + "</main>"
        + sc.FOOTER_HTML
        + f"<script>{cand_js}</script>\n<script>{tabs_js}</script>\n</body>\n</html>\n"
    )


def publish(
    *,
    candidates_csv: Path = ck.DEFAULT_CANDIDATES_CSV,
    out_dir: Path = ck.DEFAULT_KIT_DIR,
    site_manifest_path: Path = bs.DEFAULT_MANIFEST_PATH,
    demo_manifest_path: Path = bs.DEFAULT_DEMO_MANIFEST_PATH,
) -> Path:
    cfg = load_config()
    results = ck.build_all(cfg, candidates_csv, out_dir)

    sessions = sorted({r.candidate.session for r in results})
    if SITE_ASSETS_DIR.exists():
        shutil.rmtree(SITE_ASSETS_DIR)
    SITE_ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    total_bytes = 0
    for session in sessions:
        src = out_dir / session
        dst = SITE_ASSETS_DIR / session
        shutil.copytree(src, dst)
        total_bytes += sum(f.stat().st_size for f in dst.rglob("*") if f.is_file())

    candidates_fragment = ck.render_candidates_fragment(results, asset_prefix=ASSET_PREFIX)

    site_manifest = json.loads(site_manifest_path.read_text(encoding="utf-8"))
    demo_manifest = json.loads(demo_manifest_path.read_text(encoding="utf-8"))
    strikes_html = bs.render_clip_cards(site_manifest, "strikes")
    modes_html = bs.render_clip_cards(demo_manifest, "modes")
    n_strikes = len(site_manifest["strikes"])
    n_modes = sum(1 for c in demo_manifest["clips"] if c["kind"] == "state")

    html = compose_audio_review_html(
        candidates_fragment=candidates_fragment,
        strikes_html=strikes_html, modes_html=modes_html,
        n_candidates=len(results), n_strikes=n_strikes, n_modes=n_modes,
        sessions=sessions,
    )
    out_path = SITE_DIR / "audio_review.html"
    out_path.write_text(html, encoding="utf-8")

    total_mb = total_bytes / (1024 * 1024)
    html_mb = out_path.stat().st_size / (1024 * 1024)
    print(
        f"publish_audio_review: {len(results)} candidate(s) across {len(sessions)} session(s), "
        f"{n_strikes} strike clip(s), {n_modes} mode clip(s) -> "
        f"{out_path} ({html_mb:.2f} MB), {SITE_ASSETS_DIR} ({total_mb:.1f} MB assets)"
    )
    return out_path


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    del argv
    publish()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
