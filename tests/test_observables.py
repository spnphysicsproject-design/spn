from __future__ import annotations

import numpy as np
import pytest

from spn.observables import (
    spectral_gap,
    ordered_eigensystem,
    central_gap_4level,
    state_norm,
    normalize_state,
    expectation_value,
    expectation_value_real,
    pauli_matrices,
    bloch_vector,
    bloch_radius,
    bloch_vectors_from_states,
    sector_weights,
    sector_purity,
    dominant_sector,
    sector_mixing_ratio,
    low_sector_retention,
    leakage_out_of_low_sector,
    sector_weights_from_states,
    low_sector_retention_from_states,
    leakage_from_states,
    block_frobenius_norm,
    inter_sector_coupling_norm,
    coupling_to_gap_ratio,
    gap_to_coupling_ratio,
    is_hermitian,
)


# ---------------------------------------------------------------------
# Basic spectral observables
# ---------------------------------------------------------------------


def test_spectral_gap_returns_absolute_gap() -> None:
    eigenvalues = np.array([3.0, 1.0, 5.0])

    assert spectral_gap(eigenvalues, i=0, j=1) == pytest.approx(2.0)
    assert spectral_gap(eigenvalues, i=1, j=2) == pytest.approx(4.0)


def test_spectral_gap_rejects_non_1d_input() -> None:
    with pytest.raises(ValueError):
        spectral_gap(np.eye(2), i=0, j=1)


def test_spectral_gap_rejects_bad_indices() -> None:
    eigenvalues = np.array([1.0, 2.0])

    with pytest.raises(IndexError):
        spectral_gap(eigenvalues, i=0, j=2)


def test_ordered_eigensystem_returns_sorted_hermitian_eigensystem() -> None:
    H = np.array(
        [
            [2.0, 0.0],
            [0.0, 1.0],
        ],
        dtype=complex,
    )

    vals, vecs = ordered_eigensystem(H)

    assert np.allclose(vals, np.array([1.0, 2.0]))
    assert vecs.shape == (2, 2)
    assert np.allclose(vecs.conj().T @ vecs, np.eye(2))


def test_ordered_eigensystem_rejects_non_square_matrix() -> None:
    with pytest.raises(ValueError):
        ordered_eigensystem(np.ones((2, 3), dtype=complex))


def test_ordered_eigensystem_rejects_non_hermitian_matrix() -> None:
    A = np.array(
        [
            [1.0, 2.0],
            [3.0, 4.0],
        ],
        dtype=complex,
    )

    with pytest.raises(ValueError):
        ordered_eigensystem(A)


def test_central_gap_4level_assumes_ordered_by_default() -> None:
    eigenvalues = np.array([0.0, 1.0, 3.5, 10.0])

    assert central_gap_4level(eigenvalues) == pytest.approx(2.5)


def test_central_gap_4level_can_sort_if_requested() -> None:
    eigenvalues = np.array([10.0, 3.5, 0.0, 1.0])

    assert central_gap_4level(eigenvalues, assume_ordered=False) == pytest.approx(2.5)


def test_central_gap_4level_rejects_too_few_values() -> None:
    with pytest.raises(ValueError):
        central_gap_4level(np.array([0.0, 1.0, 2.0]))


# ---------------------------------------------------------------------
# State normalization and expectation values
# ---------------------------------------------------------------------


def test_state_norm_returns_vector_norm() -> None:
    psi = np.array([3.0 + 0.0j, 4.0 + 0.0j])

    assert state_norm(psi) == pytest.approx(5.0)


def test_state_norm_rejects_non_vector() -> None:
    with pytest.raises(ValueError):
        state_norm(np.eye(2, dtype=complex))


def test_normalize_state_returns_unit_vector() -> None:
    psi = np.array([3.0 + 0.0j, 4.0 + 0.0j])

    normalized = normalize_state(psi)

    assert np.linalg.norm(normalized) == pytest.approx(1.0)
    assert np.allclose(normalized, psi / 5.0)


def test_normalize_state_rejects_zero_vector() -> None:
    with pytest.raises(ValueError):
        normalize_state(np.zeros(2, dtype=complex))


def test_expectation_value_normalizes_input_state() -> None:
    psi = np.array([2.0 + 0.0j, 0.0 + 0.0j])
    operator = np.array(
        [
            [3.0, 0.0],
            [0.0, 5.0],
        ],
        dtype=complex,
    )

    assert expectation_value(psi, operator) == pytest.approx(3.0 + 0.0j)


def test_expectation_value_accepts_complex_values() -> None:
    psi = np.array([1.0 + 0.0j, 1.0j])
    operator = np.array(
        [
            [0.0, -1.0j],
            [1.0j, 0.0],
        ],
        dtype=complex,
    )

    value = expectation_value(psi, operator)

    # psi normalizes to [1/sqrt(2), i/sqrt(2)].
    # For sigma_y, <psi|sigma_y|psi> = +1.
    assert value == pytest.approx(1.0 + 0.0j)


def test_expectation_value_rejects_dimension_mismatch() -> None:
    psi = np.array([1.0 + 0.0j, 0.0 + 0.0j])
    operator = np.eye(3, dtype=complex)

    with pytest.raises(ValueError):
        expectation_value(psi, operator)


def test_expectation_value_real_returns_real_for_hermitian_operator() -> None:
    psi = np.array([1.0 + 0.0j, 0.0 + 0.0j])
    operator = np.array(
        [
            [2.0, 0.0],
            [0.0, 3.0],
        ],
        dtype=complex,
    )

    assert expectation_value_real(psi, operator) == pytest.approx(2.0)


def test_expectation_value_real_rejects_non_hermitian_when_required() -> None:
    psi = np.array([1.0 + 0.0j, 0.0 + 0.0j])
    operator = np.array(
        [
            [1.0, 2.0],
            [3.0, 4.0],
        ],
        dtype=complex,
    )

    with pytest.raises(ValueError):
        expectation_value_real(psi, operator, require_hermitian=True)


def test_expectation_value_real_can_ignore_hermitian_requirement() -> None:
    psi = np.array([1.0 + 0.0j, 0.0 + 0.0j])
    operator = np.array(
        [
            [1.0, 2.0],
            [3.0, 4.0],
        ],
        dtype=complex,
    )

    assert expectation_value_real(
        psi,
        operator,
        require_hermitian=False,
    ) == pytest.approx(1.0)


# ---------------------------------------------------------------------
# Bloch / two-level observables
# ---------------------------------------------------------------------


def test_pauli_matrices_have_expected_forms() -> None:
    sx, sy, sz = pauli_matrices()

    assert np.allclose(sx, np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex))
    assert np.allclose(sy, np.array([[0.0, -1.0j], [1.0j, 0.0]], dtype=complex))
    assert np.allclose(sz, np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex))


def test_bloch_vector_for_basis_zero_points_up_z() -> None:
    psi = np.array([1.0 + 0.0j, 0.0 + 0.0j])

    assert np.allclose(bloch_vector(psi), np.array([0.0, 0.0, 1.0]))


def test_bloch_vector_for_basis_one_points_down_z() -> None:
    psi = np.array([0.0 + 0.0j, 1.0 + 0.0j])

    assert np.allclose(bloch_vector(psi), np.array([0.0, 0.0, -1.0]))


def test_bloch_vector_for_equal_superposition_points_x() -> None:
    psi = np.array([1.0 + 0.0j, 1.0 + 0.0j])

    assert np.allclose(bloch_vector(psi), np.array([1.0, 0.0, 0.0]))


def test_bloch_vector_rejects_non_2_component_state() -> None:
    with pytest.raises(ValueError):
        bloch_vector(np.ones(3, dtype=complex))


def test_bloch_radius_for_pure_state_is_one() -> None:
    psi = np.array([1.0 + 0.0j, 1.0j])

    assert bloch_radius(psi) == pytest.approx(1.0)


def test_bloch_vectors_from_states_returns_expected_shape() -> None:
    states = [
        np.array([1.0 + 0.0j, 0.0 + 0.0j]),
        np.array([0.0 + 0.0j, 1.0 + 0.0j]),
    ]

    vectors = bloch_vectors_from_states(states)

    assert vectors.shape == (2, 3)
    assert np.allclose(vectors[0], np.array([0.0, 0.0, 1.0]))
    assert np.allclose(vectors[1], np.array([0.0, 0.0, -1.0]))


# ---------------------------------------------------------------------
# Sector decomposition observables
# ---------------------------------------------------------------------


def test_sector_weights_for_two_blocks() -> None:
    psi = np.array([1.0, 1.0, 2.0, 0.0], dtype=complex)

    weights = sector_weights(psi, sector_dims=[2, 2])

    # normalized |psi|^2 total = 1 + 1 + 4 = 6
    assert np.allclose(weights, np.array([2.0 / 6.0, 4.0 / 6.0]))
    assert np.sum(weights) == pytest.approx(1.0)


def test_sector_weights_rejects_bad_sector_sum() -> None:
    psi = np.ones(4, dtype=complex)

    with pytest.raises(ValueError):
        sector_weights(psi, sector_dims=[2, 1])


def test_sector_weights_rejects_nonpositive_sector_dim() -> None:
    psi = np.ones(4, dtype=complex)

    with pytest.raises(ValueError):
        sector_weights(psi, sector_dims=[2, 0, 2])


def test_sector_purity_returns_largest_weight() -> None:
    psi = np.array([1.0, 1.0, 2.0, 0.0], dtype=complex)

    assert sector_purity(psi, sector_dims=[2, 2]) == pytest.approx(4.0 / 6.0)


def test_dominant_sector_returns_argmax_sector() -> None:
    psi = np.array([1.0, 1.0, 2.0, 0.0], dtype=complex)

    assert dominant_sector(psi, sector_dims=[2, 2]) == 1


def test_sector_mixing_ratio_is_one_minus_purity() -> None:
    psi = np.array([1.0, 1.0, 2.0, 0.0], dtype=complex)

    assert sector_mixing_ratio(psi, sector_dims=[2, 2]) == pytest.approx(
        1.0 - 4.0 / 6.0
    )


def test_low_sector_retention_returns_first_block_weight() -> None:
    psi = np.array([1.0, 1.0, 2.0, 0.0], dtype=complex)

    assert low_sector_retention(psi, low_dim=2) == pytest.approx(2.0 / 6.0)


def test_low_sector_retention_rejects_bad_low_dim() -> None:
    psi = np.ones(4, dtype=complex)

    with pytest.raises(ValueError):
        low_sector_retention(psi, low_dim=0)

    with pytest.raises(ValueError):
        low_sector_retention(psi, low_dim=5)


def test_leakage_out_of_low_sector_is_one_minus_retention() -> None:
    psi = np.array([1.0, 1.0, 2.0, 0.0], dtype=complex)

    retention = low_sector_retention(psi, low_dim=2)
    leakage = leakage_out_of_low_sector(psi, low_dim=2)

    assert leakage == pytest.approx(1.0 - retention)


def test_sector_weights_from_states_returns_expected_shape() -> None:
    states = [
        np.array([1.0, 0.0, 0.0, 0.0], dtype=complex),
        np.array([0.0, 0.0, 1.0, 0.0], dtype=complex),
    ]

    weights = sector_weights_from_states(states, sector_dims=[2, 2])

    assert weights.shape == (2, 2)
    assert np.allclose(weights[0], np.array([1.0, 0.0]))
    assert np.allclose(weights[1], np.array([0.0, 1.0]))


def test_low_sector_retention_from_states_returns_expected_values() -> None:
    states = [
        np.array([1.0, 0.0, 0.0, 0.0], dtype=complex),
        np.array([0.0, 0.0, 1.0, 0.0], dtype=complex),
    ]

    retentions = low_sector_retention_from_states(states, low_dim=2)

    assert np.allclose(retentions, np.array([1.0, 0.0]))


def test_leakage_from_states_returns_expected_values() -> None:
    states = [
        np.array([1.0, 0.0, 0.0, 0.0], dtype=complex),
        np.array([0.0, 0.0, 1.0, 0.0], dtype=complex),
    ]

    leakage = leakage_from_states(states, low_dim=2)

    assert np.allclose(leakage, np.array([0.0, 1.0]))


# ---------------------------------------------------------------------
# Matrix-level inter-sector diagnostics
# ---------------------------------------------------------------------


def test_block_frobenius_norm_returns_block_norm() -> None:
    H = np.array(
        [
            [1.0, 0.0, 3.0, 4.0],
            [0.0, 1.0, 0.0, 0.0],
            [3.0, 0.0, 2.0, 0.0],
            [4.0, 0.0, 0.0, 2.0],
        ],
        dtype=complex,
    )

    norm = block_frobenius_norm(H, slice(0, 2), slice(2, 4))

    assert norm == pytest.approx(5.0)


def test_inter_sector_coupling_norm_returns_expected_block_norm() -> None:
    H = np.array(
        [
            [1.0, 0.0, 3.0, 4.0],
            [0.0, 1.0, 0.0, 0.0],
            [3.0, 0.0, 2.0, 0.0],
            [4.0, 0.0, 0.0, 2.0],
        ],
        dtype=complex,
    )

    norm = inter_sector_coupling_norm(
        H,
        sector_dims=[2, 2],
        a=0,
        b=1,
    )

    assert norm == pytest.approx(5.0)


def test_inter_sector_coupling_norm_rejects_bad_sector_index() -> None:
    H = np.eye(4, dtype=complex)

    with pytest.raises(IndexError):
        inter_sector_coupling_norm(
            H,
            sector_dims=[2, 2],
            a=0,
            b=2,
        )


def test_inter_sector_coupling_norm_rejects_bad_sector_dims() -> None:
    H = np.eye(4, dtype=complex)

    with pytest.raises(ValueError):
        inter_sector_coupling_norm(
            H,
            sector_dims=[2, 1],
            a=0,
            b=1,
        )


def test_coupling_to_gap_ratio_returns_coupling_over_gap() -> None:
    eigenvalues = np.array([0.0, 2.0])
    coupling_norm = 0.5

    assert coupling_to_gap_ratio(
        eigenvalues,
        coupling_norm=coupling_norm,
        i=0,
        j=1,
    ) == pytest.approx(0.25)


def test_coupling_to_gap_ratio_returns_inf_for_zero_gap() -> None:
    eigenvalues = np.array([1.0, 1.0])

    assert coupling_to_gap_ratio(
        eigenvalues,
        coupling_norm=0.5,
        i=0,
        j=1,
    ) == np.inf


def test_coupling_to_gap_ratio_rejects_negative_coupling() -> None:
    eigenvalues = np.array([0.0, 2.0])

    with pytest.raises(ValueError):
        coupling_to_gap_ratio(
            eigenvalues,
            coupling_norm=-0.1,
            i=0,
            j=1,
        )


def test_gap_to_coupling_ratio_alias_matches_coupling_to_gap_ratio() -> None:
    eigenvalues = np.array([0.0, 2.0])
    coupling_norm = 0.5

    assert gap_to_coupling_ratio(
        eigenvalues,
        coupling_norm=coupling_norm,
        i=0,
        j=1,
    ) == pytest.approx(
        coupling_to_gap_ratio(
            eigenvalues,
            coupling_norm=coupling_norm,
            i=0,
            j=1,
        )
    )


# ---------------------------------------------------------------------
# Matrix property helper
# ---------------------------------------------------------------------


def test_is_hermitian_true_for_hermitian_matrix() -> None:
    H = np.array(
        [
            [1.0, 1.0 + 2.0j],
            [1.0 - 2.0j, 3.0],
        ],
        dtype=complex,
    )

    assert is_hermitian(H)


def test_is_hermitian_false_for_non_hermitian_matrix() -> None:
    A = np.array(
        [
            [1.0, 2.0],
            [3.0, 4.0],
        ],
        dtype=complex,
    )

    assert not is_hermitian(A)