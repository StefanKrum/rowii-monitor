"""Sticky HMM smoothing of per-window state labels.

Post-processes cluster/classification labels (e.g. from ``KMeansClusterer`` or
``GmmClusterer``) with a Gaussian HMM whose transition matrix is fixed to a high
self-transition probability. This removes isolated single-frame label flips while
preserving true state-boundary transitions, without letting EM re-estimate the
transition dynamics away from the intended "sticky" prior.
"""
from __future__ import annotations

import numpy as np
from hmmlearn.hmm import GaussianHMM

_MIN_COVAR = 1e-6


class StickyHmmSmoother:
    """Smooths init labels with a GaussianHMM whose transmat/startprob are fixed.

    The transition matrix is set once from `self_transition` and is never
    re-estimated (``params="mc"`` restricts EM to means/covariances only,
    ``init_params=""`` disables hmmlearn's own random initialization). Only
    Baum-Welch refinement of the per-state Gaussian emission parameters happens
    during `fit`; state assignment is decoded with Viterbi.
    """

    def __init__(self, self_transition: float = 0.98, random_seed: int = 7) -> None:
        """Initialize the smoother.

        Args:
            self_transition: Diagonal transition probability (P(stay in state)).
                Off-diagonal mass is spread uniformly over the remaining states.
            random_seed: Random seed passed to the underlying GaussianHMM.
        """
        self.self_transition = self_transition
        self.random_seed = random_seed
        self.last_model_: GaussianHMM | None = None

    def fit_decode(self, features: np.ndarray, init_labels: np.ndarray) -> np.ndarray:
        """Fit a sticky GaussianHMM and decode the Viterbi state sequence.

        Args:
            features: Per-window features, shape (W, F), float.
            init_labels: Per-window initial labels, shape (W,), integer ids.
                The number of components k is the number of unique ids in
                `init_labels`; ids need not be contiguous or zero-based.

        Returns:
            Smoothed labels, shape (W,), dtype int64, using the SAME id space
            as `init_labels` (i.e. values are drawn from `init_labels`'s
            unique ids, not remapped to 0..k-1).
        """
        unique_ids = np.unique(init_labels)
        k = len(unique_ids)

        if k <= 1:
            self.last_model_ = None
            return np.asarray(init_labels, dtype=np.int64)

        features = np.asarray(features, dtype=np.float64)
        n_features = features.shape[1]

        id_to_component = {label_id: idx for idx, label_id in enumerate(unique_ids)}
        component_labels = np.array(
            [id_to_component[label] for label in init_labels], dtype=np.int64
        )

        transmat = _sticky_transmat(k, self.self_transition)
        startprob = np.full(k, 1.0 / k)
        means, covars = _init_means_covars(features, component_labels, k, n_features)

        model = GaussianHMM(
            n_components=k,
            covariance_type="diag",
            params="mc",
            init_params="",
            random_state=self.random_seed,
        )
        model.startprob_ = startprob
        model.transmat_ = transmat
        model.means_ = means
        model.covars_ = covars

        model.fit(features)

        if not np.allclose(model.transmat_, transmat):
            raise RuntimeError("sticky transmat was re-estimated")

        self.last_model_ = model

        decoded_components = model.predict(features)
        component_to_id = {idx: label_id for label_id, idx in id_to_component.items()}
        decoded = np.array(
            [component_to_id[c] for c in decoded_components], dtype=np.int64
        )
        return decoded


def _sticky_transmat(k: int, self_transition: float) -> np.ndarray:
    """Build a (k, k) transition matrix with fixed diagonal and uniform off-diagonal."""
    off_diagonal = (1.0 - self_transition) / (k - 1)
    transmat = np.full((k, k), off_diagonal, dtype=np.float64)
    np.fill_diagonal(transmat, self_transition)
    return transmat


def _init_means_covars(
    features: np.ndarray, component_labels: np.ndarray, k: int, n_features: int
) -> tuple[np.ndarray, np.ndarray]:
    """Per-component means and floored diagonal covariances from init label groups.

    A component with a single member (or zero variance) falls back to the
    global per-feature variance, floored at `_MIN_COVAR`.
    """
    global_var = np.var(features, axis=0)
    global_var_floored = np.maximum(global_var, _MIN_COVAR)

    means = np.empty((k, n_features), dtype=np.float64)
    covars = np.empty((k, n_features), dtype=np.float64)
    for component in range(k):
        group = features[component_labels == component]
        means[component] = np.mean(group, axis=0)
        if len(group) < 2:
            covars[component] = global_var_floored
        else:
            covars[component] = np.maximum(np.var(group, axis=0), _MIN_COVAR)
    return means, covars
