from __future__ import annotations

import numpy as np

from spn.evolution import (
    ReducedState,
    evolve_n_ticks_axisymmetric,
    final_displacement,
    max_norm_drift,
    positions_from_history,
    reduced_coupling_speed,
    reduced_group_velocity_vector,
)
from spn.kernels import generator_laplace_beltrami


def normalized_spinor() -> np.ndarray:
    """
    Return a nontrivial normalized two-component reduced spinor.
    """
    spinor = np.array([1.0 + 0.25j, -0.35 + 0.8j], dtype=complex)
    return spinor / np.linalg.norm(spinor)


def test_long_time_reduced_spinor_norm_is_preserved() -> None:
    """
    Evolve the reduced Paper-2 state for many ticks and check norm stability.

    Important convention:
        time_step is tau, the model tick duration. In physical interpretation
        this corresponds to one microscopic tick, potentially Planck-time-like,
        not ordinary wall-clock seconds.

    This is therefore a numerical many-iteration stability test, not a claim
    about long macroscopic physical duration.
    """
    n_ticks = 10_000

    # Natural units for the reduced model:
    # step_length = 1 means one microscopic length unit L.
    # time_step = 1 means one model tick tau, not one second.
    step_length = 1.0
    time_step = 1.0

    alpha = 0.4

    initial_state = ReducedState(
        position=np.zeros(3, dtype=float),
        spinor=normalized_spinor(),
        tick=0,
    )

    history = evolve_n_ticks_axisymmetric(
        initial_state,
        n_ticks=n_ticks,
        k_vector=np.array([0.2, -0.3, 0.4], dtype=float),
        step_length=step_length,
        time_step=time_step,
        generator_fn=generator_laplace_beltrami,
        branch=1,
        keep_history=True,
        renormalize=False,
        alpha=alpha,
    )

    assert isinstance(history, list)
    assert len(history) == n_ticks + 1
    assert history[-1].tick == n_ticks

    drift = max_norm_drift(history)
    assert drift < 1.0e-10


def test_long_time_packet_centre_matches_analytic_group_velocity() -> None:
    """
    The reduced packet centre should move linearly with the reduced group
    velocity:

        x_N = x_0 + N * tau * v_group

    This is reduced packet-centre propagation, not literal microscopic
    fixed-step transport over N spatial Planck-length jumps.
    """
    n_ticks = 10_000

    # Natural units for the reduced model:
    # step_length = 1 means one microscopic length unit L.
    # time_step = 1 means one model tick tau, not one second.
    step_length = 1.0
    time_step = 1.0

    alpha = 0.4
    k_vector = np.array([0.2, -0.3, 0.4], dtype=float)

    initial_position = np.array([1.0, -2.0, 0.5], dtype=float)
    initial_state = ReducedState(
        position=initial_position,
        spinor=normalized_spinor(),
        tick=0,
    )

    history = evolve_n_ticks_axisymmetric(
        initial_state,
        n_ticks=n_ticks,
        k_vector=k_vector,
        step_length=step_length,
        time_step=time_step,
        generator_fn=generator_laplace_beltrami,
        branch=1,
        keep_history=True,
        renormalize=False,
        alpha=alpha,
    )

    assert isinstance(history, list)

    v_group = reduced_group_velocity_vector(
        k_vector=k_vector,
        step_length=step_length,
        time_step=time_step,
        generator_fn=generator_laplace_beltrami,
        branch=1,
        alpha=alpha,
    )

    expected_final_position = initial_position + n_ticks * time_step * v_group
    actual_final_position = history[-1].position

    np.testing.assert_allclose(
        actual_final_position,
        expected_final_position,
        rtol=1e-12,
        atol=1e-10,
    )

    expected_displacement = n_ticks * time_step * v_group
    actual_displacement = final_displacement(history)

    np.testing.assert_allclose(
        actual_displacement,
        expected_displacement,
        rtol=1e-12,
        atol=1e-10,
    )


def test_long_time_packet_centre_has_constant_tick_to_tick_displacement() -> None:
    """
    For fixed k_vector and fixed generator parameters, the reduced group
    velocity is constant, so each tick should add the same packet-centre
    displacement.

    The per-tick displacement is tau * v_group. It is not assumed to have
    microscopic length L.
    """
    n_ticks = 2_000

    # Natural units for the reduced model:
    # step_length = 1 means one microscopic length unit L.
    # time_step = 1 means one model tick tau, not one second.
    step_length = 1.0
    time_step = 1.0

    alpha = 0.4
    k_vector = np.array([0.15, 0.25, -0.35], dtype=float)

    initial_state = ReducedState(
        position=np.array([0.5, -0.25, 1.0], dtype=float),
        spinor=normalized_spinor(),
        tick=0,
    )

    history = evolve_n_ticks_axisymmetric(
        initial_state,
        n_ticks=n_ticks,
        k_vector=k_vector,
        step_length=step_length,
        time_step=time_step,
        generator_fn=generator_laplace_beltrami,
        branch=1,
        keep_history=True,
        renormalize=False,
        alpha=alpha,
    )

    assert isinstance(history, list)

    positions = positions_from_history(history)
    tick_displacements = np.diff(positions, axis=0)

    v_group = reduced_group_velocity_vector(
        k_vector=k_vector,
        step_length=step_length,
        time_step=time_step,
        generator_fn=generator_laplace_beltrami,
        branch=1,
        alpha=alpha,
    )

    expected_tick_displacement = time_step * v_group

    np.testing.assert_allclose(
        tick_displacements,
        np.broadcast_to(expected_tick_displacement, tick_displacements.shape),
        rtol=1e-12,
        atol=1e-12,
    )


def test_long_time_reduced_packet_speed_stays_below_reduced_speed_limit() -> None:
    """
    The reduced packet-centre speed should remain bounded by:

        v = L / (tau sqrt(3))

    This is deliberately below the microscopic transport speed L / tau.

    Since time_step is tau per tick, tau=1 here means one model tick in
    natural units, not one physical second.
    """
    n_ticks = 10_000

    # Natural units for the reduced model:
    # step_length = 1 means one microscopic length unit L.
    # time_step = 1 means one model tick tau, not one second.
    step_length = 1.0
    time_step = 1.0

    alpha = 0.4
    k_vector = np.array([0.4, -0.2, 0.1], dtype=float)

    initial_state = ReducedState(
        position=np.zeros(3, dtype=float),
        spinor=normalized_spinor(),
        tick=0,
    )

    history = evolve_n_ticks_axisymmetric(
        initial_state,
        n_ticks=n_ticks,
        k_vector=k_vector,
        step_length=step_length,
        time_step=time_step,
        generator_fn=generator_laplace_beltrami,
        branch=1,
        keep_history=True,
        renormalize=False,
        alpha=alpha,
    )

    assert isinstance(history, list)

    total_displacement = final_displacement(history)
    measured_speed = np.linalg.norm(total_displacement) / (n_ticks * time_step)

    reduced_speed_limit = reduced_coupling_speed(
        step_length=step_length,
        time_step=time_step,
    )
    microscopic_speed = step_length / time_step

    assert measured_speed <= reduced_speed_limit + 1.0e-12
    assert reduced_speed_limit < microscopic_speed


def test_long_time_evolution_without_history_returns_final_state() -> None:
    """
    The no-history path should return only the final ReducedState while still
    preserving the long-time tick count, norm, and packet-centre displacement.
    """
    n_ticks = 10_000

    # Natural units for the reduced model:
    # step_length = 1 means one microscopic length unit L.
    # time_step = 1 means one model tick tau, not one second.
    step_length = 1.0
    time_step = 1.0

    alpha = 0.4
    k_vector = np.array([0.2, -0.3, 0.4], dtype=float)

    initial_position = np.array([-1.0, 0.25, 2.0], dtype=float)
    initial_spinor = normalized_spinor()

    initial_state = ReducedState(
        position=initial_position,
        spinor=initial_spinor,
        tick=0,
    )

    final_state = evolve_n_ticks_axisymmetric(
        initial_state,
        n_ticks=n_ticks,
        k_vector=k_vector,
        step_length=step_length,
        time_step=time_step,
        generator_fn=generator_laplace_beltrami,
        branch=1,
        keep_history=False,
        renormalize=False,
        alpha=alpha,
    )

    assert isinstance(final_state, ReducedState)
    assert final_state.tick == n_ticks

    np.testing.assert_allclose(
        np.linalg.norm(final_state.spinor),
        np.linalg.norm(initial_spinor),
        rtol=1e-12,
        atol=1e-10,
    )

    v_group = reduced_group_velocity_vector(
        k_vector=k_vector,
        step_length=step_length,
        time_step=time_step,
        generator_fn=generator_laplace_beltrami,
        branch=1,
        alpha=alpha,
    )

    expected_position = initial_position + n_ticks * time_step * v_group

    np.testing.assert_allclose(
        final_state.position,
        expected_position,
        rtol=1e-12,
        atol=1e-10,
    )