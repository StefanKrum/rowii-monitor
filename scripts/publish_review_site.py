"""Publish the candidate-review kit onto the public site (`docs/site/review.html` +
`docs/site/review_static.html`) -- the v2 site redesign's "port scripts/
candidate_kit.py's build to emit the new visual system" step.

`results/candidate-kit/` (gitignored) stays the ONE canonical, reproducible build
location -- `candidate_kit.py build`'s own existing contract, untouched here. This
script:

    1. Calls `candidate_kit.build_all` exactly as `candidate_kit.py build` itself
       would (same `out_dir`, unprefixed asset paths) -- reuses already-rendered
       WAV/PNG assets via that function's own existing reuse path, so a rerun
       after this task's edits is fast (nothing here re-extracts audio).
    2. Copies every candidate's per-session asset directory from `results/
       candidate-kit/<session>/` into `docs/site/assets/review/<session>/` (a
       TRACKED, committed location -- `results/` itself is gitignored, so the
       site's own copy is the only one that ever reaches git).
    3. Re-renders `index.html`/`index_static.html` a SECOND time (cheap: pure
       string assembly over the already-in-memory `results`, no new data I/O)
       with `html_out_path`/`asset_prefix` pointed at the site copy, producing
       `docs/site/review.html` / `docs/site/review_static.html`.

Run from the repo root: `python scripts/publish_review_site.py`.
"""
from __future__ import annotations

import logging
import shutil
import sys
from pathlib import Path

_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

import candidate_kit as ck  # noqa: E402

from rowii.config import load_config  # noqa: E402

logger = logging.getLogger(__name__)

REPO_ROOT = _SCRIPTS_DIR.parent
SITE_DIR = REPO_ROOT / "docs" / "site"
SITE_ASSETS_DIR = SITE_DIR / "assets" / "review"
ASSET_PREFIX = "assets/review/"


def publish(
    *, candidates_csv: Path = ck.DEFAULT_CANDIDATES_CSV, out_dir: Path = ck.DEFAULT_KIT_DIR
) -> None:
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

    html_path = ck.render_index_html(
        results, out_dir, html_out_path=SITE_DIR / "review.html", asset_prefix=ASSET_PREFIX
    )
    static_path = ck.render_index_static_html(
        results, out_dir, html_out_path=SITE_DIR / "review_static.html", asset_prefix=ASSET_PREFIX
    )

    total_mb = total_bytes / (1024 * 1024)
    html_mb = html_path.stat().st_size / (1024 * 1024)
    static_mb = static_path.stat().st_size / (1024 * 1024)
    print(
        f"publish_review_site: {len(results)} candidate(s) across {len(sessions)} session(s) -> "
        f"{html_path} ({html_mb:.2f} MB), {static_path} ({static_mb:.2f} MB), "
        f"{SITE_ASSETS_DIR} ({total_mb:.1f} MB assets)"
    )


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    del argv
    publish()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
