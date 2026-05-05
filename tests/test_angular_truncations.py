from __future__ import annotations

import numpy as np
import pytest
from scipy.linalg import expm

from spn.angular_truncations import (
    angular_generator_matrix,
    angular_generator_spectrum,
    axisymmetric_enlarged_hamiltonian,
    axisymmetric_l_values,
    axisymmetric_transport_matrix,
    cos_theta_coupling_matrix,
    cos_theta_matrix_element,
    expected_reduced_low_block,
    is_hermitian,
    low_block_difference_norm,
    low_sector_indices,
    low_sector_mean_and_gap_from_spectrum,
    natural_low_block_from_axisymmetric,
    reduced_order_permutation_matrix,
    reordered_low_block_from_axisymmetric,
)
from spn.kernels import generator_laplace_beltrami


# ---------------------------------------------------------------------
# Robust numerical thresholds
# ---------------------------------------------------------------------

HERMITIAN_ATOL = 1e-12
UNITARY_ATOL = 1e-10
NORM_ATOL = 1e-10
BLOCK_ATOL = 1e-12

ZERO_LEAKAGE_ATOL = 1e-11

BASELINE_LEAKAGE_MAX = 7.5e-3
LMAX_CONVERGENCE_MAX = 7.5e-3
LMAX_PLATEAU_ABS_DIFF = 5e-4

HIGH_K_LEAKAGE_MAX = 7.5e-2
K03_LEAKAGE_MIN = 1e-3

INITIAL_STATE_L1_MAX = 1e-2
INITIAL_STATE_RATIO_MIN = 20.0


# ---------------------------------------------------------------------
# Local test helpers
# ---------------------------------------------------------------------

def state_norm(psi: np.ndarray) -> float:
    psi = np.asarray(psi, dtype=complex)
    return float(np.real_if_close(np.vdot(psi, psi)))


def unitary_from_hamiltonian(H: np.ndarray, tau: float) -> tuple[np.ndarray, float]:
    U = expm(-1j * tau * H)
    unitary_error = np.linalg.norm(U.conj().T @ U - np.eye(H.shape[0]))
    return U, float(unitary_error)


def subspace_weight(psi: np.ndarray, indices: np.ndarray) -> float:
    psi = np.asarray(psi, dtype=complex)
    indices = np.asarray(indices, dtype=int)
    return float(np.sum(np.abs(psi[indices]) ** 2))


def subspace_leakage(psi: np.ndarray, indices: np.ndarray) -> float:
    leakage = 1.0 - subspace_weight(psi, indices)
    return float(max(leakage, 0.0))


def embedded_low_state(l_max: int, coeffs_low: list[complex]) -> np.ndarray:
    psi = np.zeros(l_max + 1, dtype=complex)
    psi[0] = coeffs_low[0]
    psi[1] = coeffs_low[1]

    norm = np.sqrt(np.vdot(psi, psi).real)
    if norm == 0:
        raise ValueError("low-sector state has zero norm")

    return psi / norm


def peak_leakage_for_case(
    *,
    l_max: int,
    k: float,
    initial_state: np.ndarray,
    n_steps: int = 1_000,
    L: float = 1.0,
    tau: float = 1.0,
    alpha: float = 1.0,
    transport_coupling_multiplier: float = 1.0,
) -> dict[str, float]:
    H, *_ = axisymmetric_enlarged_hamiltonian(
        l_max=l_max,
        k=k,
        L=L,
        tau=tau,
        generator_fn=generator_laplace_beltrami,
        generator_params={"alpha": alpha},
        transport_coupling_multiplier=transport_coupling_multiplier,
    )

    U, unitary_error = unitary_from_hamiltonian(H, tau=tau)

    psi = np.asarray(initial_state, dtype=complex)
    psi = psi / np.sqrt(np.vdot(psi, psi).real)

    low_indices = low_sector_indices()

    leakages = []
    norm_errors = []

    for _ in range(n_steps + 1):
        leakages.append(subspace_leakage(psi, low_indices))
        norm_errors.append(abs(state_norm(psi) - 1.0))
        psi = U @ psi

    return {
        "peak_leakage": float(np.max(leakages)),
        "final_leakage": float(leakages[-1]),
        "max_norm_error": float(np.max(norm_errors)),
        "unitary_error": float(unitary_error),
    }


# ---------------------------------------------------------------------
# Basis and validation tests
# ---------------------------------------------------------------------

def test_axisymmetric_l_values_returns_expected_basis() -> None:
    result = axisymmetric_l_values(4)

    np.testing.assert_array_equal(result, np.array([0, 1, 2, 3, 4]))


@pytest.mark.parametrize("bad_lmax", [0, -1, -5])
def test_axisymmetric_l_values_rejects_lmax_below_one(bad_lmax: int) -> None:
    with pytest.raises(ValueError):
        axisymmetric_l_values(bad_lmax)


@pytest.mark.parametrize("bad_lmax", [1.0, "3", None])
def test_axisymmetric_l_values_rejects_non_integer_lmax(bad_lmax: object) -> None:
    with pytest.raises(TypeError):
        axisymmetric_l_values(bad_lmax)  # type: ignore[arg-type]


def test_low_sector_indices_returns_first_two_natural_indices() -> None:
    np.testing.assert_array_equal(low_sector_indices(), np.array([0, 1]))


# ---------------------------------------------------------------------
# cos(theta) matrix tests
# ---------------------------------------------------------------------

def test_cos_theta_matrix_element_matches_known_values() -> None:
    assert np.isclose(cos_theta_matrix_element(0), 1.0 / np.sqrt(3.0))
    assert np.isclose(cos_theta_matrix_element(1), 2.0 / np.sqrt(15.0))
    assert np.isclose(cos_theta_matrix_element(2), 3.0 / np.sqrt(35.0))


@pytest.mark.parametrize("bad_l", [-1, -3])
def test_cos_theta_matrix_element_rejects_negative_l(bad_l: int) -> None:
    with pytest.raises(ValueError):
        cos_theta_matrix_element(bad_l)


@pytest.mark.parametrize("bad_l", [1.0, "1", None])
def test_cos_theta_matrix_element_rejects_non_integer_l(bad_l: object) -> None:
    with pytest.raises(TypeError):
        cos_theta_matrix_element(bad_l)  # type: ignore[arg-type]


def test_cos_theta_coupling_matrix_has_expected_shape_and_symmetry() -> None:
    l_max = 5
    C = cos_theta_coupling_matrix(l_max)

    assert C.shape == (l_max + 1, l_max + 1)
    assert np.allclose(C, C.T, atol=HERMITIAN_ATOL)


def test_cos_theta_coupling_matrix_has_only_nearest_neighbour_entries() -> None:
    l_max = 6
    C = cos_theta_coupling_matrix(l_max)

    for i in range(l_max + 1):
        for j in range(l_max + 1):
            if abs(i - j) > 1 or i == j:
                assert np.isclose(C[i, j], 0.0)


def test_cos_theta_coupling_matrix_low_entries_match_conventions() -> None:
    C = cos_theta_coupling_matrix(4)

    assert np.isclose(C[0, 1], 1.0 / np.sqrt(3.0))
    assert np.isclose(C[1, 0], 1.0 / np.sqrt(3.0))

    assert np.isclose(C[1, 2], 2.0 / np.sqrt(15.0))
    assert np.isclose(C[2, 1], 2.0 / np.sqrt(15.0))


# ---------------------------------------------------------------------
# Angular generator tests
# ---------------------------------------------------------------------

def test_angular_generator_spectrum_laplace_beltrami() -> None:
    spectrum = angular_generator_spectrum(
        l_max=5,
        generator_fn=generator_laplace_beltrami,
        alpha=1.0,
    )

    expected = np.array([0.0, 2.0, 6.0, 12.0, 20.0, 30.0])
    np.testing.assert_allclose(spectrum, expected)


def test_angular_generator_spectrum_rejects_non_callable_generator() -> None:
    with pytest.raises(TypeError):
        angular_generator_spectrum(3, generator_fn=42)  # type: ignore[arg-type]


def test_angular_generator_spectrum_rejects_non_finite_values() -> None:
    def bad_generator(l: int) -> float:
        return np.inf if l == 1 else float(l)

    with pytest.raises(ValueError):
        angular_generator_spectrum(3, generator_fn=bad_generator)


def test_angular_generator_matrix_is_diagonal() -> None:
    H_omega, spectrum = angular_generator_matrix(
        l_max=4,
        generator_fn=generator_laplace_beltrami,
        alpha=1.0,
    )

    assert H_omega.shape == (5, 5)
    np.testing.assert_allclose(H_omega, np.diag(spectrum))
    assert is_hermitian(H_omega)


def test_low_sector_mean_and_gap_from_spectrum_uses_half_gap() -> None:
    spectrum = np.array([0.0, 2.0, 6.0])
    fbar, delta = low_sector_mean_and_gap_from_spectrum(spectrum)

    assert np.isclose(fbar, 1.0)
    assert np.isclose(delta, 1.0)


def test_low_sector_mean_and_gap_rejects_too_short_spectrum() -> None:
    with pytest.raises(ValueError):
        low_sector_mean_and_gap_from_spectrum(np.array([0.0]))


# ---------------------------------------------------------------------
# Transport and enlarged Hamiltonian tests
# ---------------------------------------------------------------------

def test_axisymmetric_transport_matrix_has_expected_low_coupling() -> None:
    H_transport, C = axisymmetric_transport_matrix(
        l_max=4,
        k=0.3,
        L=1.0,
        tau=1.0,
    )

    expected_low_coupling = 0.3 / np.sqrt(3.0)

    assert np.isclose(C[0, 1], 1.0 / np.sqrt(3.0))
    assert np.isclose(H_transport[0, 1], expected_low_coupling)
    assert np.isclose(H_transport[1, 0], expected_low_coupling)
    assert is_hermitian(H_transport)


def test_axisymmetric_transport_matrix_zero_multiplier_is_zero_matrix() -> None:
    H_transport, C = axisymmetric_transport_matrix(
        l_max=4,
        k=0.3,
        L=1.0,
        tau=1.0,
        transport_coupling_multiplier=0.0,
    )

    np.testing.assert_allclose(H_transport, np.zeros((5, 5), dtype=complex))
    assert np.any(np.abs(C) > 0.0)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"k": -0.1, "L": 1.0, "tau": 1.0},
        {"k": 0.1, "L": 0.0, "tau": 1.0},
        {"k": 0.1, "L": -1.0, "tau": 1.0},
        {"k": 0.1, "L": 1.0, "tau": 0.0},
        {"k": 0.1, "L": 1.0, "tau": -1.0},
    ],
)
def test_axisymmetric_transport_matrix_rejects_invalid_scale_inputs(
    kwargs: dict[str, float],
) -> None:
    with pytest.raises(ValueError):
        axisymmetric_transport_matrix(l_max=3, **kwargs)


def test_axisymmetric_transport_matrix_rejects_negative_multiplier() -> None:
    with pytest.raises(ValueError):
        axisymmetric_transport_matrix(
            l_max=3,
            k=0.3,
            L=1.0,
            tau=1.0,
            transport_coupling_multiplier=-1.0,
        )


def test_axisymmetric_enlarged_hamiltonian_is_hermitian() -> None:
    H, H_omega, H_transport, spectrum, C = axisymmetric_enlarged_hamiltonian(
        l_max=6,
        k=0.3,
        L=1.0,
        tau=1.0,
        generator_fn=generator_laplace_beltrami,
        generator_params={"alpha": 1.0},
    )

    assert H.shape == (7, 7)
    assert is_hermitian(H, atol=HERMITIAN_ATOL)
    assert is_hermitian(H_omega, atol=HERMITIAN_ATOL)
    assert is_hermitian(H_transport, atol=HERMITIAN_ATOL)
    assert C.shape == (7, 7)
    assert spectrum.shape == (7,)


# ---------------------------------------------------------------------
# Reduced low-block reproduction tests
# ---------------------------------------------------------------------

def test_natural_low_block_from_axisymmetric_extracts_l0_l1_block() -> None:
    H, *_ = axisymmetric_enlarged_hamiltonian(
        l_max=4,
        k=0.3,
        L=1.0,
        tau=1.0,
        generator_fn=generator_laplace_beltrami,
        generator_params={"alpha": 1.0},
    )

    block = natural_low_block_from_axisymmetric(H)

    expected = np.array(
        [
            [0.0, 0.3 / np.sqrt(3.0)],
            [0.3 / np.sqrt(3.0), 2.0],
        ],
        dtype=complex,
    )

    np.testing.assert_allclose(block, expected, atol=BLOCK_ATOL)


def test_reduced_order_permutation_matrix_swaps_low_basis_order() -> None:
    P = reduced_order_permutation_matrix()

    natural_vector_l0 = np.array([1.0, 0.0])
    natural_vector_l1 = np.array([0.0, 1.0])

    np.testing.assert_allclose(P @ natural_vector_l0, np.array([0.0, 1.0]))
    np.testing.assert_allclose(P @ natural_vector_l1, np.array([1.0, 0.0]))


def test_reordered_low_block_matches_expected_paper2_hamiltonian() -> None:
    k = 0.3
    L = 1.0
    tau = 1.0

    H, _, _, spectrum, _ = axisymmetric_enlarged_hamiltonian(
        l_max=6,
        k=k,
        L=L,
        tau=tau,
        generator_fn=generator_laplace_beltrami,
        generator_params={"alpha": 1.0},
    )

    fbar, delta = low_sector_mean_and_gap_from_spectrum(spectrum)

    reduced_block = reordered_low_block_from_axisymmetric(H)
    expected = expected_reduced_low_block(
        fbar=fbar,
        delta=delta,
        k=k,
        L=L,
        tau=tau,
    )

    np.testing.assert_allclose(reduced_block, expected, atol=BLOCK_ATOL)


def test_low_block_difference_norm_is_zero_for_baseline() -> None:
    k = 0.3
    L = 1.0
    tau = 1.0

    H, _, _, spectrum, _ = axisymmetric_enlarged_hamiltonian(
        l_max=10,
        k=k,
        L=L,
        tau=tau,
        generator_fn=generator_laplace_beltrami,
        generator_params={"alpha": 1.0},
    )

    diff = low_block_difference_norm(H, spectrum, k=k, L=L, tau=tau)

    assert diff < BLOCK_ATOL


def test_expected_reduced_low_block_rejects_invalid_inputs() -> None:
    with pytest.raises(ValueError):
        expected_reduced_low_block(fbar=1.0, delta=1.0, k=-0.1, L=1.0, tau=1.0)

    with pytest.raises(ValueError):
        expected_reduced_low_block(fbar=1.0, delta=1.0, k=0.1, L=0.0, tau=1.0)

    with pytest.raises(ValueError):
        expected_reduced_low_block(fbar=1.0, delta=1.0, k=0.1, L=1.0, tau=0.0)


# ---------------------------------------------------------------------
# Basic enlarged evolution and leakage tests
# ---------------------------------------------------------------------

def test_enlarged_unitary_preserves_norm() -> None:
    l_max = 6
    H, *_ = axisymmetric_enlarged_hamiltonian(
        l_max=l_max,
        k=0.3,
        L=1.0,
        tau=1.0,
        generator_fn=generator_laplace_beltrami,
        generator_params={"alpha": 1.0},
    )

    U, unitary_error = unitary_from_hamiltonian(H, tau=1.0)
    assert unitary_error < UNITARY_ATOL

    psi = embedded_low_state(l_max, [1.0, 1.0])

    max_norm_error = 0.0
    for _ in range(1_000):
        max_norm_error = max(max_norm_error, abs(state_norm(psi) - 1.0))
        psi = U @ psi

    max_norm_error = max(max_norm_error, abs(state_norm(psi) - 1.0))

    assert max_norm_error < NORM_ATOL


def test_lmax_one_has_zero_low_sector_leakage() -> None:
    l_max = 1
    psi0 = embedded_low_state(l_max, [1.0, 1.0])

    result = peak_leakage_for_case(
        l_max=l_max,
        k=0.3,
        initial_state=psi0,
    )

    assert result["peak_leakage"] < ZERO_LEAKAGE_ATOL
    assert result["max_norm_error"] < NORM_ATOL
    assert result["unitary_error"] < UNITARY_ATOL


def test_baseline_axisymmetric_leakage_is_small() -> None:
    l_max = 10
    psi0 = embedded_low_state(l_max, [1.0, 1.0])

    result = peak_leakage_for_case(
        l_max=l_max,
        k=0.3,
        initial_state=psi0,
    )

    assert result["peak_leakage"] < BASELINE_LEAKAGE_MAX
    assert result["peak_leakage"] > K03_LEAKAGE_MIN
    assert result["max_norm_error"] < NORM_ATOL
    assert result["unitary_error"] < UNITARY_ATOL


def test_axisymmetric_leakage_converges_with_lmax() -> None:
    leakages = {}

    for l_max in [1, 2, 6, 10]:
        psi0 = embedded_low_state(l_max, [1.0, 1.0])
        result = peak_leakage_for_case(
            l_max=l_max,
            k=0.3,
            initial_state=psi0,
        )
        leakages[l_max] = result["peak_leakage"]

    assert leakages[1] < ZERO_LEAKAGE_ATOL
    assert max(leakages[2], leakages[6], leakages[10]) < LMAX_CONVERGENCE_MAX
    assert abs(leakages[10] - leakages[6]) < LMAX_PLATEAU_ABS_DIFF


def test_axisymmetric_leakage_vanishes_at_zero_k() -> None:
    l_max = 10
    psi0 = embedded_low_state(l_max, [1.0, 1.0])

    result = peak_leakage_for_case(
        l_max=l_max,
        k=0.0,
        initial_state=psi0,
    )

    assert result["peak_leakage"] < ZERO_LEAKAGE_ATOL
    assert result["max_norm_error"] < NORM_ATOL
    assert result["unitary_error"] < UNITARY_ATOL


def test_axisymmetric_leakage_increases_with_k() -> None:
    l_max = 10
    psi0 = embedded_low_state(l_max, [1.0, 1.0])

    leak_0 = peak_leakage_for_case(
        l_max=l_max,
        k=0.0,
        initial_state=psi0,
    )["peak_leakage"]

    leak_03 = peak_leakage_for_case(
        l_max=l_max,
        k=0.3,
        initial_state=psi0,
    )["peak_leakage"]

    leak_10 = peak_leakage_for_case(
        l_max=l_max,
        k=1.0,
        initial_state=psi0,
    )["peak_leakage"]

    assert leak_0 < ZERO_LEAKAGE_ATOL
    assert leak_03 > K03_LEAKAGE_MIN
    assert leak_10 > leak_03
    assert leak_10 < HIGH_K_LEAKAGE_MAX


def test_l1_initial_state_leaks_more_than_l0() -> None:
    l_max = 10

    psi_l0 = embedded_low_state(l_max, [1.0, 0.0])
    psi_l1 = embedded_low_state(l_max, [0.0, 1.0])
    psi_equal = embedded_low_state(l_max, [1.0, 1.0])

    leak_l0 = peak_leakage_for_case(
        l_max=l_max,
        k=0.3,
        initial_state=psi_l0,
    )["peak_leakage"]

    leak_l1 = peak_leakage_for_case(
        l_max=l_max,
        k=0.3,
        initial_state=psi_l1,
    )["peak_leakage"]

    leak_equal = peak_leakage_for_case(
        l_max=l_max,
        k=0.3,
        initial_state=psi_equal,
    )["peak_leakage"]

    assert leak_l1 > leak_equal
    assert leak_equal > leak_l0
    assert leak_l1 > INITIAL_STATE_RATIO_MIN * leak_l0
    assert leak_l1 < INITIAL_STATE_L1_MAX


def test_transport_coupling_multiplier_controls_leakage() -> None:
    l_max = 10
    psi0 = embedded_low_state(l_max, [1.0, 1.0])

    leak_0 = peak_leakage_for_case(
        l_max=l_max,
        k=0.3,
        initial_state=psi0,
        transport_coupling_multiplier=0.0,
    )["peak_leakage"]

    leak_half = peak_leakage_for_case(
        l_max=l_max,
        k=0.3,
        initial_state=psi0,
        transport_coupling_multiplier=0.5,
    )["peak_leakage"]

    leak_full = peak_leakage_for_case(
        l_max=l_max,
        k=0.3,
        initial_state=psi0,
        transport_coupling_multiplier=1.0,
    )["peak_leakage"]

    assert leak_0 < ZERO_LEAKAGE_ATOL
    assert leak_half > leak_0
    assert leak_full > leak_half
    assert leak_full < BASELINE_LEAKAGE_MAX