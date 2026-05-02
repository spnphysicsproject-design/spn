from __future__ import annotations

import numpy as np

from spn.evolution import (
    reduced_axisymmetric_hamiltonian,
    reduced_axisymmetric_unitary,
    unitary_from_hermitian,
)
from spn.kernels import generator_laplace_beltrami


# Natural SPN reduced-model units:
# step_length = 1 means one microscopic length unit L.
# time_step = 1 means one fundamental model tick tau.
# One application of the unitary represents one discrete SPN tick,
# not one ordinary second.
#
# Important:
# time_step enters H_low through v = L / (tau sqrt(3)), so the enlarged
# Hamiltonian and the reference two-level reduced unitary must use the same
# time_step.
DEFAULT_STEP_LENGTH = 1.0
DEFAULT_TIME_STEP = 1.0
DEFAULT_ALPHA = 0.4
DEFAULT_K_VECTOR = np.array([0.25, -0.15, 0.35], dtype=float)


def normalized_low_sector_state() -> np.ndarray:
    """
    Return a nontrivial normalized two-component low-sector state.
    """
    psi = np.array([1.0 + 0.2j, -0.35 + 0.7j], dtype=complex)
    return psi / np.linalg.norm(psi)


def representative_low_sector_states() -> list[np.ndarray]:
    """
    Return representative normalized low-sector states.

    These guard against leakage conclusions depending on one favourable
    initial spinor.
    """
    states = [
        np.array([1.0, 0.0], dtype=complex),
        np.array([0.0, 1.0], dtype=complex),
        np.array([1.0, 1.0j], dtype=complex),
        np.array([1.0 + 0.2j, -0.35 + 0.7j], dtype=complex),
    ]

    return [state / np.linalg.norm(state) for state in states]


def embed_low_state(psi_low: np.ndarray) -> np.ndarray:
    """
    Embed a 2-component low-sector state into a 4-component enlarged state.

    Basis convention for this toy enlarged model:

        components 0:2 -> low Paper-2 sector
        components 2:4 -> toy complement sector
    """
    psi_low = np.asarray(psi_low, dtype=complex)

    if psi_low.shape != (2,):
        raise ValueError("psi_low must have shape (2,)")

    psi = np.zeros(4, dtype=complex)
    psi[:2] = psi_low
    return psi


def low_sector_retention(psi: np.ndarray) -> float:
    """
    Return probability retained in the first two components.

    This test intentionally assumes contiguous low-sector basis blocks:
    components 0:2 are the reduced Paper-2 low sector.
    """
    psi = np.asarray(psi, dtype=complex)
    norm_sq = float(np.vdot(psi, psi).real)

    if norm_sq == 0.0:
        raise ValueError("psi must be nonzero")

    low_weight = float(np.vdot(psi[:2], psi[:2]).real)
    return low_weight / norm_sq


def leakage_out_of_low_sector(psi: np.ndarray) -> float:
    """
    Return probability outside the first two low-sector components.

    Tiny negative values can occur from floating-point roundoff when retention
    is numerically just above 1, so tests should allow a small negative
    tolerance.
    """
    return 1.0 - low_sector_retention(psi)


def evolve_state(U: np.ndarray, psi0: np.ndarray, n_ticks: int) -> np.ndarray:
    """
    Evolve a state by repeated application of U.

    This is a discrete model-tick evolution. The number of ticks is a
    numerical iteration count, not a duration in ordinary seconds.
    """
    if n_ticks < 0:
        raise ValueError("n_ticks must be non-negative")

    psi = np.asarray(psi0, dtype=complex)

    for _ in range(n_ticks):
        psi = U @ psi

    return psi


def leakage_history(U: np.ndarray, psi0: np.ndarray, n_ticks: int) -> np.ndarray:
    """
    Return leakage after each model tick, including the initial state.

    This is preferable to checking only final-time leakage because
    finite-dimensional unitary systems can show oscillatory leakage and
    recurrence.
    """
    if n_ticks < 0:
        raise ValueError("n_ticks must be non-negative")

    psi = np.asarray(psi0, dtype=complex)
    leakages = [leakage_out_of_low_sector(psi)]

    for _ in range(n_ticks):
        psi = U @ psi
        leakages.append(leakage_out_of_low_sector(psi))

    return np.array(leakages, dtype=float)


def enlarged_sector_hamiltonian(
    *,
    epsilon: float,
    step_length: float = DEFAULT_STEP_LENGTH,
    time_step: float = DEFAULT_TIME_STEP,
    alpha: float = DEFAULT_ALPHA,
    k_vector: np.ndarray = DEFAULT_K_VECTOR,
    complement_sector_energies: tuple[float, float] = (8.0, 11.0),
) -> np.ndarray:
    """
    Build a controlled 4x4 toy enlarged-sector Hamiltonian.

    The upper-left 2x2 block is the current Paper-2 reduced Hamiltonian:

        H_low = fbar I + Delta sigma_z + v k sigma_x

    The lower-right 2x2 block is a separated toy complement sector.

    The off-diagonal blocks are controlled by epsilon. This is a toy
    pressure test of reduction stability, not a derived full angular
    truncation.

    Important:
        time_step enters H_low through v = L / (tau sqrt(3)), so tests that
        compare the enlarged low block against the two-level reduced model
        must use the same time_step in both places.
    """
    H_low = reduced_axisymmetric_hamiltonian(
        k_vector=k_vector,
        step_length=step_length,
        time_step=time_step,
        generator_fn=generator_laplace_beltrami,
        alpha=alpha,
    )

    H_complement = np.diag(np.array(complement_sector_energies, dtype=float)).astype(
        complex
    )

    # A fixed, nontrivial low-complement coupling pattern. The scale epsilon
    # controls leakage strength. The matrix is intentionally not diagonal so
    # that both low states can weakly communicate with both complement states.
    coupling_pattern = np.array(
        [
            [1.0, 0.25 - 0.1j],
            [-0.4 + 0.2j, 0.7],
        ],
        dtype=complex,
    )
    V = epsilon * coupling_pattern

    H = np.zeros((4, 4), dtype=complex)
    H[:2, :2] = H_low
    H[2:, 2:] = H_complement
    H[:2, 2:] = V
    H[2:, :2] = V.conj().T

    return H


def test_enlarged_sector_hamiltonian_is_hermitian() -> None:
    """
    The toy enlarged-sector Hamiltonian must be Hermitian before it is used
    for leakage tests.
    """
    H = enlarged_sector_hamiltonian(epsilon=0.01)

    np.testing.assert_allclose(H, H.conj().T, rtol=1e-14, atol=1e-14)


def test_enlarged_sector_low_block_matches_reduced_hamiltonian() -> None:
    """
    The low-sector block of the toy enlarged Hamiltonian should exactly match
    the current tested two-level Paper-2 reduced Hamiltonian.
    """
    H_enlarged = enlarged_sector_hamiltonian(
        epsilon=0.01,
        step_length=DEFAULT_STEP_LENGTH,
        time_step=DEFAULT_TIME_STEP,
        alpha=DEFAULT_ALPHA,
        k_vector=DEFAULT_K_VECTOR,
    )

    H_low = reduced_axisymmetric_hamiltonian(
        k_vector=DEFAULT_K_VECTOR,
        step_length=DEFAULT_STEP_LENGTH,
        time_step=DEFAULT_TIME_STEP,
        generator_fn=generator_laplace_beltrami,
        alpha=DEFAULT_ALPHA,
    )

    np.testing.assert_allclose(
        H_enlarged[:2, :2],
        H_low,
        rtol=1e-14,
        atol=1e-14,
    )


def test_zero_coupling_has_no_low_complement_blocks() -> None:
    """
    With epsilon = 0, the toy enlarged Hamiltonian should be exactly block
    diagonal between the low sector and complement sector.
    """
    H = enlarged_sector_hamiltonian(
        epsilon=0.0,
        step_length=DEFAULT_STEP_LENGTH,
        time_step=DEFAULT_TIME_STEP,
        alpha=DEFAULT_ALPHA,
        k_vector=DEFAULT_K_VECTOR,
    )

    np.testing.assert_allclose(
        H[:2, 2:],
        np.zeros((2, 2), dtype=complex),
        rtol=0.0,
        atol=0.0,
    )
    np.testing.assert_allclose(
        H[2:, :2],
        np.zeros((2, 2), dtype=complex),
        rtol=0.0,
        atol=0.0,
    )


def test_zero_coupling_keeps_low_sector_closed_to_numerical_precision() -> None:
    """
    With zero low-complement coupling, initially low-sector states should
    remain in the low sector up to numerical precision.

    This checks that leakage is caused only by the controlled off-diagonal
    coupling and not by the embedding itself.
    """
    n_ticks = 500

    H = enlarged_sector_hamiltonian(
        epsilon=0.0,
        step_length=DEFAULT_STEP_LENGTH,
        time_step=DEFAULT_TIME_STEP,
        alpha=DEFAULT_ALPHA,
        k_vector=DEFAULT_K_VECTOR,
    )
    U = unitary_from_hermitian(H, tau=DEFAULT_TIME_STEP)

    for psi_low0 in representative_low_sector_states():
        psi0 = embed_low_state(psi_low0)
        psi_final = evolve_state(U, psi0, n_ticks=n_ticks)

        np.testing.assert_allclose(
            np.linalg.norm(psi_final),
            1.0,
            rtol=1e-12,
            atol=1e-12,
        )

        complement_weight = float(np.vdot(psi_final[2:], psi_final[2:]).real)
        assert complement_weight < 1.0e-24

        np.testing.assert_allclose(
            low_sector_retention(psi_final),
            1.0,
            rtol=1e-12,
            atol=1e-12,
        )

        assert abs(leakage_out_of_low_sector(psi_final)) < 1.0e-12


def test_zero_coupling_matches_embedded_two_level_reduced_evolution() -> None:
    """
    With zero low-complement coupling, the enlarged 4D model should reproduce
    the tested two-level reduced evolution in the low block.

    This confirms that the toy enlarged-sector construction preserves the
    Paper-2 reduced dynamics when the complement is decoupled.
    """
    n_ticks = 200

    H_enlarged = enlarged_sector_hamiltonian(
        epsilon=0.0,
        step_length=DEFAULT_STEP_LENGTH,
        time_step=DEFAULT_TIME_STEP,
        alpha=DEFAULT_ALPHA,
        k_vector=DEFAULT_K_VECTOR,
    )
    U_enlarged = unitary_from_hermitian(H_enlarged, tau=DEFAULT_TIME_STEP)

    U_low = reduced_axisymmetric_unitary(
        k_vector=DEFAULT_K_VECTOR,
        step_length=DEFAULT_STEP_LENGTH,
        time_step=DEFAULT_TIME_STEP,
        generator_fn=generator_laplace_beltrami,
        alpha=DEFAULT_ALPHA,
    )

    for psi_low0 in representative_low_sector_states():
        psi0 = embed_low_state(psi_low0)

        psi_enlarged_final = evolve_state(U_enlarged, psi0, n_ticks=n_ticks)
        psi_low_final = evolve_state(U_low, psi_low0, n_ticks=n_ticks)

        np.testing.assert_allclose(
            psi_enlarged_final[:2],
            psi_low_final,
            rtol=1e-12,
            atol=1e-12,
        )

        complement_weight = float(
            np.vdot(psi_enlarged_final[2:], psi_enlarged_final[2:]).real
        )
        assert complement_weight < 1.0e-24


def test_weak_off_resonant_coupling_leaks_less_than_stronger_coupling() -> None:
    """
    With the same separated complement sector, weaker low-complement coupling
    should produce less peak leakage than stronger low-complement coupling
    for representative low-sector initial states.

    This avoids relying on an arbitrary absolute leakage threshold. It checks
    the controlled perturbative direction: increasing epsilon should increase
    peak leakage in this toy enlarged-sector model.
    """
    n_ticks = 1_000

    H_weak = enlarged_sector_hamiltonian(
        epsilon=1.0e-3,
        step_length=DEFAULT_STEP_LENGTH,
        time_step=DEFAULT_TIME_STEP,
        alpha=DEFAULT_ALPHA,
        k_vector=DEFAULT_K_VECTOR,
        complement_sector_energies=(8.0, 11.0),
    )
    H_stronger = enlarged_sector_hamiltonian(
        epsilon=1.0e-2,
        step_length=DEFAULT_STEP_LENGTH,
        time_step=DEFAULT_TIME_STEP,
        alpha=DEFAULT_ALPHA,
        k_vector=DEFAULT_K_VECTOR,
        complement_sector_energies=(8.0, 11.0),
    )

    U_weak = unitary_from_hermitian(H_weak, tau=DEFAULT_TIME_STEP)
    U_stronger = unitary_from_hermitian(H_stronger, tau=DEFAULT_TIME_STEP)

    for psi_low0 in representative_low_sector_states():
        psi0 = embed_low_state(psi_low0)

        weak_peak_leakage = float(np.max(leakage_history(U_weak, psi0, n_ticks)))
        stronger_peak_leakage = float(
            np.max(leakage_history(U_stronger, psi0, n_ticks))
        )

        assert weak_peak_leakage >= -1.0e-12
        assert stronger_peak_leakage > weak_peak_leakage


def test_far_complement_sector_leaks_less_than_near_complement_sector() -> None:
    """
    For the same low-complement coupling, peak leakage should be smaller when
    the complement sector is farther away in energy, for representative
    low-sector initial states.

    This checks the expected gap-suppression intuition behind reduction
    stability in the controlled toy model.
    """
    n_ticks = 1_000
    epsilon = 1.0e-3

    H_far = enlarged_sector_hamiltonian(
        epsilon=epsilon,
        step_length=DEFAULT_STEP_LENGTH,
        time_step=DEFAULT_TIME_STEP,
        alpha=DEFAULT_ALPHA,
        k_vector=DEFAULT_K_VECTOR,
        complement_sector_energies=(8.0, 11.0),
    )
    H_near = enlarged_sector_hamiltonian(
        epsilon=epsilon,
        step_length=DEFAULT_STEP_LENGTH,
        time_step=DEFAULT_TIME_STEP,
        alpha=DEFAULT_ALPHA,
        k_vector=DEFAULT_K_VECTOR,
        complement_sector_energies=(1.2, 1.6),
    )

    U_far = unitary_from_hermitian(H_far, tau=DEFAULT_TIME_STEP)
    U_near = unitary_from_hermitian(H_near, tau=DEFAULT_TIME_STEP)

    for psi_low0 in representative_low_sector_states():
        psi0 = embed_low_state(psi_low0)

        far_peak_leakage = float(np.max(leakage_history(U_far, psi0, n_ticks)))
        near_peak_leakage = float(np.max(leakage_history(U_near, psi0, n_ticks)))

        assert far_peak_leakage >= -1.0e-12
        assert near_peak_leakage > far_peak_leakage


def test_low_sector_retention_and_leakage_sum_to_one() -> None:
    """
    Retention and leakage diagnostics should be complementary probabilities.
    """
    n_ticks = 500

    H = enlarged_sector_hamiltonian(
        epsilon=0.01,
        step_length=DEFAULT_STEP_LENGTH,
        time_step=DEFAULT_TIME_STEP,
        alpha=DEFAULT_ALPHA,
        k_vector=DEFAULT_K_VECTOR,
    )
    U = unitary_from_hermitian(H, tau=DEFAULT_TIME_STEP)

    for psi_low0 in representative_low_sector_states():
        psi0 = embed_low_state(psi_low0)
        psi_final = evolve_state(U, psi0, n_ticks=n_ticks)

        retention = low_sector_retention(psi_final)
        leakage = leakage_out_of_low_sector(psi_final)

        np.testing.assert_allclose(retention + leakage, 1.0, rtol=1e-14, atol=1e-14)
        assert 0.0 <= retention <= 1.0 + 1.0e-12
        assert -1.0e-12 <= leakage <= 1.0 + 1.0e-12


def test_enlarged_sector_unitary_preserves_total_norm_with_coupling() -> None:
    """
    The enlarged toy model should preserve total state norm even when
    low-complement coupling is nonzero.
    """
    n_ticks = 1_000

    H = enlarged_sector_hamiltonian(
        epsilon=0.01,
        step_length=DEFAULT_STEP_LENGTH,
        time_step=DEFAULT_TIME_STEP,
        alpha=DEFAULT_ALPHA,
        k_vector=DEFAULT_K_VECTOR,
    )
    U = unitary_from_hermitian(H, tau=DEFAULT_TIME_STEP)

    for psi_low0 in representative_low_sector_states():
        psi0 = embed_low_state(psi_low0)
        psi_final = evolve_state(U, psi0, n_ticks=n_ticks)

        np.testing.assert_allclose(
            np.linalg.norm(psi_final),
            np.linalg.norm(psi0),
            rtol=1e-12,
            atol=1e-12,
        )