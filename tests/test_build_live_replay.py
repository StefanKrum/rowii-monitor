import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import build_live_replay as blr


def test_live_sessions_registry() -> None:
    assert set(blr.LIVE_SESSIONS) == {
        "290626-tu", "080726-pu_strikes", "010726-tu1-morning", "270626-pu_ph_pu_ph_pu_ph-1",
    }
    assert blr.LIVE_SESSIONS["290626-tu"]["out"] == "live.html"
    outs = [s["out"] for s in blr.LIVE_SESSIONS.values()]
    assert len(outs) == len(set(outs))


def test_session_summary_shape() -> None:
    s = blr.session_summary("080726-pu_strikes", duration_s=3600.0, n_episodes=7)
    assert s["id"] == "080726-pu_strikes"
    assert s["display_name"] == "Hammer-strike day"
    assert s["events"] is True
    assert s["duration_s"] == 3600.0
    assert s["n_episodes"] == 7
    date_label = s["date_label"]
    assert isinstance(date_label, str)
    assert date_label.startswith("WED · 08 JUL 2026")


def test_humanized_names_helper() -> None:
    names = blr.humanize_names(["RAWGeneratorMic__0::ch0_log_rms", "bogus"])
    assert names[0] == "Generator mic 1 · loudness (log RMS)"
    assert names[1] == "bogus"
