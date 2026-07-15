"""Score-level fusion of the `fusion` variant's audio and vibration branches via
p-value combination (design spec `docs/superpowers/specs/2026-07-15-step2-package3-
baselines-design.md` D5, plan `docs/superpowers/plans/2026-07-15-step2-package3-
baselines.md` Task 5).

Feature-level fusion (package 1) concatenates audio and vibration feature columns onto
one grid and scores them with a SINGLE scorer. Score-level fusion instead scores each
branch SEPARATELY -- its own scorer, its own split-conformal calibration set, its own
conformal p-values (`rowii.anomaly.conformal.p_values`) -- and only combines the two
branches' evidence at the very end, as one scalar per window:

- `split_branch_columns` recovers which of the `fusion` variant's feature columns
  belong to the audio branch and which to the vibration branch, from `PreparedRun.
  feature_names`' own `"<stream>::<local_name>"` convention (`rowii.pipeline.
  _assemble_feature_names`) -- no new feature extraction.
- `fisher_statistic` and `tippett_statistic` each reduce a pair of per-window branch
  p-values `(p_a, p_v)` into one combined real-valued score, higher = more anomalous,
  so both slot into the SAME "higher = more anomalous" contract every scorer in
  `rowii.anomaly.scorers` already satisfies. Fisher's method sums evidence from BOTH
  branches (`-2 (ln p_a + ln p_v)`, the classical log-combination); Tippett's rule
  reports only the single MOST extreme branch (`1 - min(p_a, p_v)`) -- the two
  represent opposite fusion philosophies (require corroboration vs. trust the
  strongest signal alone), which is why the orchestration view
  (`scripts/run_step2.py`'s `_run_score_fusion_view`) reports both side by side rather
  than picking one.

**Statistical note -- why RE-CALIBRATION, not the classical Fisher/Tippett null,
restores the FAR guarantee (and what "same footing" requires).** Classically, Fisher's
method compares `-2(ln p_a + ln p_v)` against a `chi2(4)` reference distribution, and
Tippett's rule rejects when `min(p_a, p_v)` falls below a critical value derived from
`1 - (1 - min(p_a, p_v))^2` (the CDF of the minimum of two independent Uniform(0, 1)
variables) -- BOTH classical references assume `p_a` and `p_v` are INDEPENDENT
Uniform(0, 1) under the null, an assumption two branches of the SAME physical machine
(audio and vibration sensors on one running turbine) have no reason to satisfy.
Neither `fisher_statistic` nor `tippett_statistic` below ever compares against those
classical reference distributions: each is used only as a DETERMINISTIC score
transform, `(p_a, p_v) -> one real number`, and the actual decision threshold comes
from applying `rowii.anomaly.conformal.calibrate` to that combined statistic's own
values on a held-out conformal-side sample -- exactly the same split-conformal
machinery every other scorer in this package is thresholded with. Split conformal's
finite-sample FAR guarantee (`conformal.py`'s own module docstring) only ever requires
the CALIBRATION and SCORING draws of the combined statistic to be exchangeable; it
never required, and never assumed, anything about how `p_a` and `p_v` relate to EACH
OTHER.

That exchangeability requirement has one sharp precondition, though: the combined
statistic must be ONE fixed transform applied on the same footing to calibration-side
and scoring-side windows alike -- concretely, each window's branch p-value must be
evaluated against a reference that EXCLUDES that window. Scoring-side p-values satisfy
this automatically (`p_values(scoring, calibration)`, the reference never contains a
scoring window); the calibration side must use `rowii.anomaly.conformal.loo_p_values`
(leave-one-out: each calibration window against the other n-1). The superficially
natural `p_values(calibration, calibration)` puts each point in its OWN reference
(its self-match caps its p-value at no less than `2/(n+1)`, while a scoring window
can reach `1/(n+1)`) -- for a SINGLE branch that mismatch is a monotone transform of
the raw score and cancels exactly, but combined across two branches it breaks the
calibration/scoring exchangeability of the combined statistic and is
ANTI-CONSERVATIVE: measured mean realized FAR up to ~0.10 at alpha=0.05, n=39,
anti-correlated branches (review finding 2026-07-15, 6000-rep simulation, scratch
scripts not committed). With LOO calibration p-values the footing matches up to a
one-unit p-granularity residual (LOO reference has n-1 points vs the scoring side's
n, so the smallest LOO p is `1/n` vs `1/(n+1)`); that residual can only inflate the
calibration-side statistic, i.e. raise the threshold, i.e. suppress alarms
(conservative direction -- `loo_p_values`' docstring carries the pointwise
derivation). Validated empirically FOR THE FISHER RULE: mean realized FAR at or below
alpha within Monte-Carlo precision across independent, shared-latent-correlated
(rho ~ 0.78), anti-correlated, and identical branches at n in {39, 159}
(`tests/test_fusion.py`'s multi-regime validity test, one-sided `alpha + 3*SE` bound;
additionally at n=319 in the review-time simulation). Verified separately (scratch,
not committed) that the classical NON-recalibrated chi2(4) threshold's realized FAR
drifts well above alpha under branch correlation -- the re-calibration is
load-bearing, not decorative.

TIPPETT does not get the same clean statement (review round 2, 2026-07-15): a
min-rule combination cannot be exactly calibrated when the calibration set doubles as
the branches' p-value reference. The LOO rescaling is a strictly monotone map of the
calibration-side min-rank, so it never changes a single Tippett alarm decision
(verified bit-identical to the self-referential construction) -- the residual is
INTRINSIC, not a construction bug the LOO switch could fix. Under shared-latent
POSITIVE correlation the measured mean realized FAR carries a small excess: +0.007
absolute at alpha=0.05, n=39, decaying roughly like 1/n (0.0518 at n=159, 0.0512 at
n=319); independent, anti-correlated, and identical branches measure within
`alpha + 3*SE`. A dedicated p-reference split (branch p-values for calibration AND
scoring windows both computed against a third, held-out reference set) would restore
exactness -- deliberately NOT adopted: per-state calibration pools are the binding
resource in this project's data (the package-2 scarcity results, `rowii.anomaly.
scarcity`/spec D3, already put several states near the `1/(n+1)` achievability floor;
carving a third split out of every per-state pool would push them under it).
`far_table_scorefusion.csv`'s tippett rows are therefore reported as the max-rule
CONTRAST to Fisher, carrying this documented caveat -- NOT as guaranteed-FAR rows --
and `tests/test_fusion.py` pins the excess at `alpha + 0.010` (n=39) /
`alpha + 0.005` (n=159) under positive correlation so any regression beyond the
documented level fails loudly.

No clipping: `rowii.anomaly.conformal.p_values` guarantees its output lies in `(0, 1]`
(see that function's own docstring for the exact formula establishing the bound), so
`fisher_statistic`'s `log(p_a)`/`log(p_v)` are always finite for p-values produced by
that function -- `fisher_statistic` performs no clipping of its own, and none is
needed.

`tippett_statistic` reports `1.0 - min(p_a, p_v)` rather than the textbook `min(p_a,
p_v)` purely to match this module's own "higher = more anomalous" convention (the
textbook form is smaller = more anomalous, the opposite of every scorer in
`rowii.anomaly.scorers`); this flip is monotonic (`1 - x` is strictly decreasing), so
ranking windows by `tippett_statistic` descending reproduces the exact same order as
ranking by the textbook `min(p_a, p_v)` ascending -- order-equivalent to Tippett's
min-p rule (see `tests/test_fusion.py`'s ordering test).
"""
from __future__ import annotations

import numpy as np


def _check_paired_shapes(p_a: np.ndarray, p_v: np.ndarray) -> None:
    """Shared precondition for `fisher_statistic`/`tippett_statistic`: both branches'
    p-value arrays describe the SAME windows in the SAME order, so they must be the
    same shape.

    Raises:
        ValueError: if `p_a.shape != p_v.shape`.
    """
    if p_a.shape != p_v.shape:
        raise ValueError(
            f"p_a.shape ({p_a.shape}) must equal p_v.shape ({p_v.shape}) -- both "
            f"must be the same windows' worth of per-branch p-values"
        )


def split_branch_columns(feature_names: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Audio-branch and vibration-branch column indices within a `fusion`-variant
    `PreparedRun.feature_names`.

    Each name's stream prefix (the substring before its first `"::"`, `rowii.pipeline.
    _assemble_feature_names`'s own separator) is classified by substring: contains
    `"Mic"` -> audio, contains `"Vib"` -> vibration (matches `rowii.pipeline.
    _AUDIO_STREAMS`/`_VIB_STREAMS`'s own naming, e.g. `"RAWGeneratorMic__0"`/
    `"RAWGeneratorVib__2"`). Every name must fall into EXACTLY one of the two
    branches -- neither, or somehow both, is a caller/upstream bug, not a case this
    function silently tolerates.

    Args:
        feature_names: `PreparedRun.feature_names` for a `fusion`(-beats) variant --
            every entry is expected to contain a `"::"` stream separator.

    Returns:
        `(audio_idx, vib_idx)`, each an int64 array of column positions into
        *feature_names* (and, aligned by construction, into `PreparedRun.features`'
        columns), in ascending original order. The two arrays are disjoint and
        together cover every index in `range(len(feature_names))`.

    Raises:
        ValueError: naming every offending name (its index and either "no '::'
            separator" or which/neither of "Mic"/"Vib" its stream prefix matched), if
            any name fails to land in exactly one branch; separately, if the
            resulting audio or vibration branch would be empty (e.g. *feature_names*
            came from a single-modality variant, not `fusion`).
    """
    audio_idx: list[int] = []
    vib_idx: list[int] = []
    offenders: list[str] = []
    for i, name in enumerate(feature_names):
        if "::" not in name:
            offenders.append(f"[{i}] {name!r} (no '::' stream separator)")
            continue
        stream_prefix = name.split("::", 1)[0]
        is_audio = "Mic" in stream_prefix
        is_vib = "Vib" in stream_prefix
        if is_audio and not is_vib:
            audio_idx.append(i)
        elif is_vib and not is_audio:
            vib_idx.append(i)
        else:
            reason = (
                "stream prefix matches both 'Mic' and 'Vib'" if (is_audio and is_vib)
                else "stream prefix matches neither 'Mic' nor 'Vib'"
            )
            offenders.append(f"[{i}] {name!r} ({reason})")

    if offenders:
        raise ValueError(
            "split_branch_columns: every feature name must fall into exactly one "
            f"branch ('Mic' -> audio, 'Vib' -> vibration); {len(offenders)} "
            f"offending name(s): {'; '.join(offenders)}"
        )
    if not audio_idx:
        raise ValueError(
            "split_branch_columns: audio branch is empty (no stream prefix contains "
            "'Mic') -- feature_names must come from a 'fusion'(-beats) variant"
        )
    if not vib_idx:
        raise ValueError(
            "split_branch_columns: vibration branch is empty (no stream prefix "
            "contains 'Vib') -- feature_names must come from a 'fusion'(-beats) "
            "variant"
        )
    return np.array(audio_idx, dtype=np.int64), np.array(vib_idx, dtype=np.int64)


def fisher_statistic(p_a: np.ndarray, p_v: np.ndarray) -> np.ndarray:
    """Fisher's combined-evidence statistic: `-2 * (log(p_a) + log(p_v))`, float64,
    higher = more anomalous (module docstring: "sum evidence from both branches").
    Requires both branches to point at the same anomaly for the statistic to be
    large -- a single very small p-value from ONE branch is damped by the other
    branch's own (possibly large) p-value, unlike `tippett_statistic` below.

    Args:
        p_a: `(W,)` audio-branch conformal p-values (`rowii.anomaly.conformal.
            p_values`), one per window.
        p_v: `(W,)` vibration-branch conformal p-values, same windows, same order.

    Returns:
        `(W,)` float64 combined statistic. No clipping is applied or needed: conformal
        p-values are guaranteed in `(0, 1]` by construction (module docstring), so
        `log(p_a)`/`log(p_v)` are always finite.

    Raises:
        ValueError: if `p_a.shape != p_v.shape`.
    """
    _check_paired_shapes(p_a, p_v)
    result: np.ndarray = -2.0 * (np.log(p_a) + np.log(p_v))
    return result


def tippett_statistic(p_a: np.ndarray, p_v: np.ndarray) -> np.ndarray:
    """Tippett's max-rule statistic: `1.0 - np.minimum(p_a, p_v)`, float64, higher =
    more anomalous (module docstring: "trust the single strongest branch alone").
    `1 - min(p_a, p_v)` is a strictly monotonically decreasing transform of the
    textbook Tippett statistic `min(p_a, p_v)` (smaller = more anomalous there), so
    ranking windows by THIS function descending is order-equivalent to ranking by the
    textbook rule ascending (module docstring; `tests/test_fusion.py`'s ordering
    test).

    Calibration caveat: unlike the Fisher rule, a min-rule statistic cannot be
    exactly split-conformal-calibrated when the calibration set doubles as the
    branches' p-value reference -- under POSITIVELY correlated branches the realized
    FAR carries a small intrinsic excess (~+0.007 at n=39, decaying ~1/n; the module
    docstring's Statistical note carries the full story and the deliberate
    no-third-split trade-off), so downstream views report Tippett as the max-rule
    CONTRAST to Fisher, not as a guaranteed-FAR rule.

    Args:
        p_a: `(W,)` audio-branch conformal p-values, one per window.
        p_v: `(W,)` vibration-branch conformal p-values, same windows, same order.

    Returns:
        `(W,)` float64 combined statistic, in `[0, 1)` (conformal p-values are in
        `(0, 1]`, so `min(p_a, p_v)` is in `(0, 1]` too, and `1 - that` is in
        `[0, 1)`).

    Raises:
        ValueError: if `p_a.shape != p_v.shape`.
    """
    _check_paired_shapes(p_a, p_v)
    result: np.ndarray = 1.0 - np.minimum(p_a, p_v)
    return result
