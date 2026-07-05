import numpy as np

from rowii.state.smooth import StickyHmmSmoother


def _synthetic_blocks(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """3-state synthetic features in blocks A-B-C, 200 frames each, well separated."""
    block_a = rng.normal(loc=(0.0, 0.0), scale=0.3, size=(200, 2))
    block_b = rng.normal(loc=(10.0, 10.0), scale=0.3, size=(200, 2))
    block_c = rng.normal(loc=(-10.0, 10.0), scale=0.3, size=(200, 2))
    features = np.vstack([block_a, block_b, block_c])
    truth = np.array([0] * 200 + [1] * 200 + [2] * 200)
    return features, truth


def _inject_flips(
    truth: np.ndarray, rng: np.random.Generator, n_flips: int = 15
) -> tuple[np.ndarray, np.ndarray]:
    """Flip n_flips single frames (not touching the first/last 5 frames of the
    sequence, so a flip can never coincide with a true block boundary itself)."""
    init_labels = truth.copy()
    n = len(truth)
    candidate_positions = np.arange(5, n - 5)
    flip_positions = rng.choice(candidate_positions, size=n_flips, replace=False)
    all_labels = {0, 1, 2}
    for pos in flip_positions:
        other_labels = list(all_labels - {truth[pos]})
        init_labels[pos] = rng.choice(other_labels)
    return init_labels, flip_positions


def test_removes_at_least_90_percent_of_injected_flips_and_preserves_boundaries() -> None:
    rng = np.random.default_rng(0)
    features, truth = _synthetic_blocks(rng)
    init_labels, flip_positions = _inject_flips(truth, rng, n_flips=15)

    smoother = StickyHmmSmoother(self_transition=0.98, random_seed=7)
    decoded = smoother.fit_decode(features, init_labels)

    assert decoded.dtype == np.int64
    assert set(np.unique(decoded)) <= set(np.unique(init_labels))

    # >= 90% of injected flips corrected back to the true label.
    n_flips = len(flip_positions)
    n_still_wrong = int(np.sum(decoded[flip_positions] != truth[flip_positions]))
    n_corrected = n_flips - n_still_wrong
    assert n_corrected / n_flips >= 0.90, (
        f"only corrected {n_corrected}/{n_flips} flips"
    )

    # True block boundaries (200 and 400) preserved within +/-1 frame: the decoded
    # label must match the true label everywhere except at most 1 frame on either
    # side of each boundary.
    true_boundaries = [200, 400]
    mismatches = np.where(decoded != truth)[0]
    for idx in mismatches:
        assert any(abs(idx - b) <= 1 for b in true_boundaries), (
            f"unexpected mismatch at frame {idx}, far from any true boundary"
        )


def test_transmat_not_reestimated_after_fit() -> None:
    rng = np.random.default_rng(1)
    features, truth = _synthetic_blocks(rng)
    init_labels, _ = _inject_flips(truth, rng, n_flips=15)

    self_transition = 0.98
    smoother = StickyHmmSmoother(self_transition=self_transition, random_seed=7)
    smoother.fit_decode(features, init_labels)

    k = 3
    off_diag = (1.0 - self_transition) / (k - 1)
    expected_transmat = np.full((k, k), off_diag)
    np.fill_diagonal(expected_transmat, self_transition)

    assert smoother.last_model_ is not None
    np.testing.assert_allclose(smoother.last_model_.transmat_, expected_transmat)


def test_k_equals_1_returns_input_unchanged() -> None:
    rng = np.random.default_rng(2)
    features = rng.normal(size=(50, 2))
    init_labels = np.zeros(50, dtype=np.int64)

    smoother = StickyHmmSmoother(self_transition=0.98, random_seed=7)
    decoded = smoother.fit_decode(features, init_labels)

    np.testing.assert_array_equal(decoded, init_labels)
    assert decoded.dtype == np.int64


def test_noncontiguous_label_ids_are_preserved() -> None:
    rng = np.random.default_rng(3)
    features, truth = _synthetic_blocks(rng)
    # Map {0, 1, 2} -> {0, 2, 5} (non-contiguous ids).
    id_map = {0: 0, 1: 2, 2: 5}
    init_labels = np.array([id_map[label] for label in truth], dtype=np.int64)

    smoother = StickyHmmSmoother(self_transition=0.98, random_seed=7)
    decoded = smoother.fit_decode(features, init_labels)

    assert set(np.unique(decoded)) <= {0, 2, 5}
    assert decoded.dtype == np.int64
