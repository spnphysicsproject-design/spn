from __future__ import annotations

import numpy as np
import pytest

from spn.kernels import generator_laplace_beltrami, generator_linear
from spn.evolution import (
    ReducedState,
    identity,
    is_hermitian,
    low_sector_delta,
    max_norm_drift,
    norms_from_history,
    positions_from_history,
    reduced_axisymmetric_hamiltonian,
    reduced_axisymmetric_unitary,
    reduced_coupling_speed,
    reduced_energy_eigenvalues,
    reduced_group_velocity_signed,
    reduced_group_velocity_vector,
    spinors_from_history,
    unitary_from_hermitian,
    wavevector_direction,
    wavevector_magnitude,
    evolve_one_tick_axisymmetric,
    evolve_n_ticks_axisymmetric,
    final_displacement,
    final_distance_from_origin,
)


# ---------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------


def k_vector_x(k: float = 0.2) -> np.ndarray:
    return np.array([k, 0.0, 0.0], dtype=float)


def default_state() -> ReducedState:
    return ReducedState(
        position=np.zeros(3, dtype=float),
        spinor=np.array([1.0 + 0.0j, 0.0 + 0.0j], dtype=complex),
        tick=0,
    )


# ---------------------------------------------------------------------
# ReducedState validation
# ---------------------------------------------------------------------


def test_reduced_state_accepts_valid_inputs() -> None:
    state = ReducedState(
        position=np.array([1.0, 2.0, 3.0]),
        spinor=np.array([1.0 + 0.0j, 0.0 + 1.0j]),
        tick=4,
    )

    assert np.allclose(state.position, np.array([1.0, 2.0, 3.0]))
    assert np.allclose(state.spinor, np.array([1.0 + 0.0j, 0.0 + 1.0j]))
    assert state.tick == 4


def test_reduced_state_converts_inputs_to_arrays() -> None:
    state = ReducedState(
        position=[1.0, 2.0, 3.0],  # type: ignore[arg-type]
        spinor=[1.0 + 0.0j, 0.0 + 0.0j],  # type: ignore[arg-type]
    )

    assert isinstance(state.position, np.ndarray)
    assert isinstance(state.spinor, np.ndarray)
    assert state.position.dtype == float
    assert state.spinor.dtype == complex


def test_reduced_state_rejects_non_3_vector_position() -> None:
    with pytest.raises(ValueError):
        ReducedState(
            position=np.array([1.0, 2.0]),
            spinor=np.array([1.0 + 0.0j, 0.0 + 0.0j]),
        )


def test_reduced_state_rejects_nonfinite_position() -> None:
    with pytest.raises(ValueError):
        ReducedState(
            position=np.array([1.0, np.inf, 3.0]),
            spinor=np.array([1.0 + 0.0j, 0.0 + 0.0j]),
        )


def test_reduced_state_rejects_non_1d_spinor() -> None:
    with pytest.raises(ValueError):
        ReducedState(
            position=np.zeros(3),
            spinor=np.eye(2, dtype=complex),
        )


def test_reduced_state_rejects_empty_spinor() -> None:
    with pytest.raises(ValueError):
        ReducedState(
            position=np.zeros(3),
            spinor=np.array([], dtype=complex),
        )


def test_reduced_state_rejects_nonfinite_spinor() -> None:
    with pytest.raises(ValueError):
        ReducedState(
            position=np.zeros(3),
            spinor=np.array([1.0 + 0.0j, np.nan + 0.0j]),
        )


def test_reduced_state_rejects_negative_tick() -> None:
    with pytest.raises(ValueError):
        ReducedState(
            position=np.zeros(3),
            spinor=np.array([1.0 + 0.0j, 0.0 + 0.0j]),
            tick=-1,
        )


# ---------------------------------------------------------------------
# Basic matrix helpers
# ---------------------------------------------------------------------


def test_identity_returns_complex_identity() -> None:
    I = identity(3)

    assert I.shape == (3, 3)
    assert I.dtype == complex
    assert np.allclose(I, np.eye(3, dtype=complex))


def test_identity_rejects_nonpositive_size() -> None:
    with pytest.raises(ValueError):
        identity(0)


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


def test_is_hermitian_false_for_non_square_matrix() -> None:
    A = np.ones((2, 3), dtype=complex)

    assert not is_hermitian(A)


def test_unitary_from_hermitian_returns_unitary_matrix() -> None:
    H = np.array(
        [
            [1.0, 0.2],
            [0.2, 2.0],
        ],
        dtype=complex,
    )
    tau = 0.3

    U = unitary_from_hermitian(H, tau=tau)
    I = np.eye(2, dtype=complex)

    assert U.shape == (2, 2)
    assert np.allclose(U.conj().T @ U, I)
    assert np.allclose(U @ U.conj().T, I)


def test_unitary_from_hermitian_matches_diagonal_case() -> None:
    H = np.diag([1.0, 3.0]).astype(complex)
    tau = 0.25

    U = unitary_from_hermitian(H, tau=tau)

    expected = np.diag(np.exp(-1j * tau * np.array([1.0, 3.0])))

    assert np.allclose(U, expected)


def test_unitary_from_hermitian_rejects_non_square_matrix() -> None:
    with pytest.raises(ValueError):
        unitary_from_hermitian(np.ones((2, 3), dtype=complex), tau=1.0)


def test_unitary_from_hermitian_rejects_negative_tau() -> None:
    with pytest.raises(ValueError):
        unitary_from_hermitian(np.eye(2, dtype=complex), tau=-1.0)


def test_unitary_from_hermitian_rejects_non_hermitian_matrix() -> None:
    A = np.array(
        [
            [1.0, 2.0],
            [3.0, 4.0],
        ],
        dtype=complex,
    )

    with pytest.raises(ValueError):
        unitary_from_hermitian(A, tau=1.0)


# ---------------------------------------------------------------------
# Wave-vector helpers
# ---------------------------------------------------------------------


def test_wavevector_magnitude_returns_norm() -> None:
    k = np.array([3.0, 4.0, 0.0])

    assert wavevector_magnitude(k) == pytest.approx(5.0)


def test_wavevector_magnitude_rejects_non_3_vector() -> None:
    with pytest.raises(ValueError):
        wavevector_magnitude(np.array([1.0, 2.0]))


def test_wavevector_magnitude_rejects_nonfinite_vector() -> None:
    with pytest.raises(ValueError):
        wavevector_magnitude(np.array([1.0, np.inf, 0.0]))


def test_wavevector_direction_returns_unit_direction() -> None:
    k = np.array([3.0, 4.0, 0.0])

    direction = wavevector_direction(k)

    assert np.allclose(direction, np.array([0.6, 0.8, 0.0]))
    assert np.linalg.norm(direction) == pytest.approx(1.0)


def test_wavevector_direction_returns_zero_for_zero_vector() -> None:
    direction = wavevector_direction(np.zeros(3))

    assert np.allclose(direction, np.zeros(3))


# ---------------------------------------------------------------------
# Reduced Hamiltonian helpers
# ---------------------------------------------------------------------


def test_low_sector_delta_matches_kernel_half_gap_convention() -> None:
    alpha = 0.5

    # f(0) = 0
    # f(1) = 2 alpha = 1
    # Delta = (f(1)-f(0))/2 = 0.5
    assert low_sector_delta(
        generator_laplace_beltrami,
        alpha=alpha,
    ) == pytest.approx(0.5)


def test_reduced_coupling_speed_returns_l_over_tau_sqrt_3() -> None:
    assert reduced_coupling_speed(
        step_length=2.0,
        time_step=4.0,
    ) == pytest.approx(2.0 / (4.0 * np.sqrt(3.0)))


def test_reduced_coupling_speed_rejects_negative_step_length() -> None:
    with pytest.raises(ValueError):
        reduced_coupling_speed(step_length=-1.0, time_step=1.0)


def test_reduced_coupling_speed_rejects_nonpositive_time_step() -> None:
    with pytest.raises(ValueError):
        reduced_coupling_speed(step_length=1.0, time_step=0.0)


def test_reduced_axisymmetric_hamiltonian_has_expected_shape_and_is_hermitian() -> None:
    H = reduced_axisymmetric_hamiltonian(
        k_vector=k_vector_x(0.2),
        step_length=1.0,
        time_step=1.0,
        generator_fn=generator_laplace_beltrami,
        alpha=0.5,
    )

    assert H.shape == (2, 2)
    assert np.allclose(H, H.conj().T)


def test_reduced_axisymmetric_hamiltonian_matches_expected_matrix() -> None:
    H = reduced_axisymmetric_hamiltonian(
        k_vector=k_vector_x(0.2),
        step_length=1.0,
        time_step=1.0,
        generator_fn=generator_laplace_beltrami,
        alpha=0.5,
    )

    fbar = 0.5
    delta = 0.5
    v = 1.0 / np.sqrt(3.0)
    k = 0.2

    expected = np.array(
        [
            [fbar + delta, v * k],
            [v * k, fbar - delta],
        ],
        dtype=complex,
    )

    assert np.allclose(H, expected)


def test_reduced_axisymmetric_unitary_is_unitary() -> None:
    U = reduced_axisymmetric_unitary(
        k_vector=k_vector_x(0.2),
        step_length=1.0,
        time_step=1.0,
        generator_fn=generator_laplace_beltrami,
        alpha=0.5,
    )

    I = np.eye(2, dtype=complex)

    assert U.shape == (2, 2)
    assert np.allclose(U.conj().T @ U, I)
    assert np.allclose(U @ U.conj().T, I)


# ---------------------------------------------------------------------
# Reduced dispersion and group velocity
# ---------------------------------------------------------------------


def test_reduced_energy_eigenvalues_match_expected_formula() -> None:
    E_minus, E_plus = reduced_energy_eigenvalues(
        k_vector=k_vector_x(0.2),
        step_length=1.0,
        time_step=1.0,
        generator_fn=generator_laplace_beltrami,
        alpha=0.5,
    )

    fbar = 0.5
    delta = 0.5
    v = 1.0 / np.sqrt(3.0)
    k = 0.2
    spread = np.sqrt(delta**2 + (v * k) ** 2)

    assert E_minus == pytest.approx(fbar - spread)
    assert E_plus == pytest.approx(fbar + spread)


def test_reduced_group_velocity_signed_matches_expected_formula() -> None:
    speed = reduced_group_velocity_signed(
        k_vector=k_vector_x(0.2),
        step_length=1.0,
        time_step=1.0,
        generator_fn=generator_laplace_beltrami,
        branch=1,
        alpha=0.5,
    )

    delta = 0.5
    v = 1.0 / np.sqrt(3.0)
    k = 0.2

    expected = (v**2 * k) / np.sqrt(delta**2 + (v * k) ** 2)

    assert speed == pytest.approx(expected)


def test_reduced_group_velocity_signed_changes_sign_for_lower_branch() -> None:
    upper = reduced_group_velocity_signed(
        k_vector=k_vector_x(0.2),
        step_length=1.0,
        time_step=1.0,
        generator_fn=generator_laplace_beltrami,
        branch=1,
        alpha=0.5,
    )

    lower = reduced_group_velocity_signed(
        k_vector=k_vector_x(0.2),
        step_length=1.0,
        time_step=1.0,
        generator_fn=generator_laplace_beltrami,
        branch=-1,
        alpha=0.5,
    )

    assert lower == pytest.approx(-upper)


def test_reduced_group_velocity_signed_rejects_invalid_branch() -> None:
    with pytest.raises(ValueError):
        reduced_group_velocity_signed(
            k_vector=k_vector_x(0.2),
            step_length=1.0,
            time_step=1.0,
            generator_fn=generator_laplace_beltrami,
            branch=0,  # type: ignore[arg-type]
            alpha=0.5,
        )


def test_reduced_group_velocity_is_zero_for_zero_k() -> None:
    velocity = reduced_group_velocity_vector(
        k_vector=np.zeros(3),
        step_length=1.0,
        time_step=1.0,
        generator_fn=generator_laplace_beltrami,
        branch=1,
        alpha=0.5,
    )

    assert np.allclose(velocity, np.zeros(3))


def test_reduced_group_velocity_vector_points_along_k_vector() -> None:
    velocity = reduced_group_velocity_vector(
        k_vector=np.array([0.0, 0.3, 0.0]),
        step_length=1.0,
        time_step=1.0,
        generator_fn=generator_laplace_beltrami,
        branch=1,
        alpha=0.5,
    )

    assert velocity[0] == pytest.approx(0.0)
    assert velocity[1] > 0.0
    assert velocity[2] == pytest.approx(0.0)


def test_reduced_group_velocity_vector_changes_sign_for_lower_branch() -> None:
    upper = reduced_group_velocity_vector(
        k_vector=k_vector_x(0.2),
        step_length=1.0,
        time_step=1.0,
        generator_fn=generator_laplace_beltrami,
        branch=1,
        alpha=0.5,
    )

    lower = reduced_group_velocity_vector(
        k_vector=k_vector_x(0.2),
        step_length=1.0,
        time_step=1.0,
        generator_fn=generator_laplace_beltrami,
        branch=-1,
        alpha=0.5,
    )

    assert np.allclose(lower, -upper)


# ---------------------------------------------------------------------
# One-tick and n-tick evolution
# ---------------------------------------------------------------------


def test_evolve_one_tick_axisymmetric_advances_tick() -> None:
    state = default_state()

    next_state = evolve_one_tick_axisymmetric(
        state,
        k_vector=k_vector_x(0.2),
        step_length=1.0,
        time_step=1.0,
        generator_fn=generator_laplace_beltrami,
        alpha=0.5,
    )

    assert next_state.tick == 1


def test_evolve_one_tick_axisymmetric_preserves_spinor_norm() -> None:
    state = default_state()

    next_state = evolve_one_tick_axisymmetric(
        state,
        k_vector=k_vector_x(0.2),
        step_length=1.0,
        time_step=1.0,
        generator_fn=generator_laplace_beltrami,
        alpha=0.5,
    )

    assert np.linalg.norm(next_state.spinor) == pytest.approx(
        np.linalg.norm(state.spinor)
    )


def test_evolve_one_tick_axisymmetric_allows_renormalize_flag() -> None:
    state = default_state()

    next_state = evolve_one_tick_axisymmetric(
        state,
        k_vector=k_vector_x(0.2),
        step_length=1.0,
        time_step=1.0,
        generator_fn=generator_laplace_beltrami,
        renormalize=True,
        alpha=0.5,
    )

    assert np.linalg.norm(next_state.spinor) == pytest.approx(1.0)


def test_evolve_one_tick_axisymmetric_moves_packet_center_by_group_velocity() -> None:
    state = default_state()
    k_vec = k_vector_x(0.2)
    time_step = 1.0

    next_state = evolve_one_tick_axisymmetric(
        state,
        k_vector=k_vec,
        step_length=1.0,
        time_step=time_step,
        generator_fn=generator_laplace_beltrami,
        alpha=0.5,
    )

    expected_velocity = reduced_group_velocity_vector(
        k_vector=k_vec,
        step_length=1.0,
        time_step=time_step,
        generator_fn=generator_laplace_beltrami,
        branch=1,
        alpha=0.5,
    )

    assert np.allclose(
        next_state.position,
        state.position + time_step * expected_velocity,
    )


def test_evolve_one_tick_axisymmetric_rejects_non_2_component_spinor() -> None:
    state = ReducedState(
        position=np.zeros(3),
        spinor=np.ones(3, dtype=complex),
    )

    with pytest.raises(ValueError):
        evolve_one_tick_axisymmetric(
            state,
            k_vector=k_vector_x(0.2),
            step_length=1.0,
            time_step=1.0,
            generator_fn=generator_laplace_beltrami,
            alpha=0.5,
        )


def test_evolve_n_ticks_axisymmetric_returns_history_by_default() -> None:
    history = evolve_n_ticks_axisymmetric(
        default_state(),
        n_ticks=5,
        k_vector=k_vector_x(0.2),
        step_length=1.0,
        time_step=1.0,
        generator_fn=generator_laplace_beltrami,
        alpha=0.5,
    )

    assert isinstance(history, list)
    assert len(history) == 6
    assert history[0].tick == 0
    assert history[-1].tick == 5


def test_evolve_n_ticks_axisymmetric_can_return_final_state_only() -> None:
    state = evolve_n_ticks_axisymmetric(
        default_state(),
        n_ticks=5,
        k_vector=k_vector_x(0.2),
        step_length=1.0,
        time_step=1.0,
        generator_fn=generator_laplace_beltrami,
        keep_history=False,
        alpha=0.5,
    )

    assert isinstance(state, ReducedState)
    assert state.tick == 5


def test_evolve_n_ticks_axisymmetric_rejects_negative_n_ticks() -> None:
    with pytest.raises(ValueError):
        evolve_n_ticks_axisymmetric(
            default_state(),
            n_ticks=-1,
            k_vector=k_vector_x(0.2),
            step_length=1.0,
            time_step=1.0,
            generator_fn=generator_laplace_beltrami,
            alpha=0.5,
        )


def test_evolve_n_ticks_axisymmetric_preserves_norms() -> None:
    history = evolve_n_ticks_axisymmetric(
        default_state(),
        n_ticks=20,
        k_vector=k_vector_x(0.2),
        step_length=1.0,
        time_step=1.0,
        generator_fn=generator_laplace_beltrami,
        alpha=0.5,
    )

    norms = norms_from_history(history)

    assert np.allclose(norms, np.ones_like(norms))


def test_evolve_n_ticks_axisymmetric_final_displacement_matches_group_velocity() -> None:
    n_ticks = 10
    k_vec = k_vector_x(0.2)
    time_step = 1.0

    history = evolve_n_ticks_axisymmetric(
        default_state(),
        n_ticks=n_ticks,
        k_vector=k_vec,
        step_length=1.0,
        time_step=time_step,
        generator_fn=generator_laplace_beltrami,
        alpha=0.5,
    )

    expected_velocity = reduced_group_velocity_vector(
        k_vector=k_vec,
        step_length=1.0,
        time_step=time_step,
        generator_fn=generator_laplace_beltrami,
        branch=1,
        alpha=0.5,
    )

    expected_displacement = n_ticks * time_step * expected_velocity

    assert np.allclose(final_displacement(history), expected_displacement)


def test_evolve_n_ticks_axisymmetric_zero_k_leaves_packet_center_fixed() -> None:
    history = evolve_n_ticks_axisymmetric(
        default_state(),
        n_ticks=5,
        k_vector=np.zeros(3),
        step_length=1.0,
        time_step=1.0,
        generator_fn=generator_laplace_beltrami,
        alpha=0.5,
    )

    assert np.allclose(history[0].position, history[-1].position)


# ---------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------


def test_positions_from_history_returns_position_array() -> None:
    history = evolve_n_ticks_axisymmetric(
        default_state(),
        n_ticks=3,
        k_vector=k_vector_x(0.2),
        step_length=1.0,
        time_step=1.0,
        generator_fn=generator_laplace_beltrami,
        alpha=0.5,
    )

    positions = positions_from_history(history)

    assert positions.shape == (4, 3)


def test_spinors_from_history_returns_spinor_array() -> None:
    history = evolve_n_ticks_axisymmetric(
        default_state(),
        n_ticks=3,
        k_vector=k_vector_x(0.2),
        step_length=1.0,
        time_step=1.0,
        generator_fn=generator_laplace_beltrami,
        alpha=0.5,
    )

    spinors = spinors_from_history(history)

    assert spinors.shape == (4, 2)


def test_norms_from_history_returns_norm_array() -> None:
    history = evolve_n_ticks_axisymmetric(
        default_state(),
        n_ticks=3,
        k_vector=k_vector_x(0.2),
        step_length=1.0,
        time_step=1.0,
        generator_fn=generator_laplace_beltrami,
        alpha=0.5,
    )

    norms = norms_from_history(history)

    assert norms.shape == (4,)
    assert np.allclose(norms, np.ones(4))


def test_max_norm_drift_is_small_for_unitary_evolution() -> None:
    history = evolve_n_ticks_axisymmetric(
        default_state(),
        n_ticks=20,
        k_vector=k_vector_x(0.2),
        step_length=1.0,
        time_step=1.0,
        generator_fn=generator_laplace_beltrami,
        alpha=0.5,
    )

    assert max_norm_drift(history) < 1e-12


def test_max_norm_drift_rejects_empty_history() -> None:
    with pytest.raises(ValueError):
        max_norm_drift([])


def test_final_displacement_rejects_empty_history() -> None:
    with pytest.raises(ValueError):
        final_displacement([])


def test_final_distance_from_origin_matches_norm_of_displacement() -> None:
    history = evolve_n_ticks_axisymmetric(
        default_state(),
        n_ticks=5,
        k_vector=k_vector_x(0.2),
        step_length=1.0,
        time_step=1.0,
        generator_fn=generator_laplace_beltrami,
        alpha=0.5,
    )

    assert final_distance_from_origin(history) == pytest.approx(
        np.linalg.norm(final_displacement(history))
    )


# ---------------------------------------------------------------------
# Alternate generator smoke test
# ---------------------------------------------------------------------


def test_evolution_accepts_linear_generator() -> None:
    history = evolve_n_ticks_axisymmetric(
        default_state(),
        n_ticks=3,
        k_vector=k_vector_x(0.2),
        step_length=1.0,
        time_step=1.0,
        generator_fn=generator_linear,
        kappa=0.5,
    )

    assert len(history) == 4
    assert max_norm_drift(history) < 1e-12