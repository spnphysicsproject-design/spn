from __future__ import annotations

import numpy as np
import pytest

from spn.kernels import generator_laplace_beltrami
from spn.reduced_models import (
    ReducedModelConfig,
    ReducedAxisymmetricModel,
    make_reduced_axisymmetric_model,
)


# ---------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------


def make_basic_model() -> ReducedAxisymmetricModel:
    return make_reduced_axisymmetric_model(
        step_length=1.0,
        time_step=1.0,
        k_vector=np.array([0.2, 0.0, 0.0]),
        generator_fn=generator_laplace_beltrami,
        alpha=0.5,
    )


# ---------------------------------------------------------------------
# ReducedModelConfig validation
# ---------------------------------------------------------------------


def test_reduced_model_config_accepts_valid_inputs() -> None:
    config = ReducedModelConfig(
        step_length=1.0,
        time_step=1.0,
        k_vector=np.array([0.1, 0.0, 0.0]),
        generator_fn=generator_laplace_beltrami,
        generator_params={"alpha": 0.5},
        branch=1,
    )

    assert config.step_length == pytest.approx(1.0)
    assert config.time_step == pytest.approx(1.0)
    assert np.allclose(config.k_vector, np.array([0.1, 0.0, 0.0]))
    assert config.generator_fn is generator_laplace_beltrami
    assert config.generator_params == {"alpha": 0.5}
    assert config.branch == 1


def test_reduced_model_config_rejects_negative_step_length() -> None:
    with pytest.raises(ValueError):
        ReducedModelConfig(
            step_length=-1.0,
            time_step=1.0,
            k_vector=np.array([0.1, 0.0, 0.0]),
            generator_fn=generator_laplace_beltrami,
            generator_params={"alpha": 0.5},
        )


def test_reduced_model_config_rejects_nonpositive_time_step() -> None:
    with pytest.raises(ValueError):
        ReducedModelConfig(
            step_length=1.0,
            time_step=0.0,
            k_vector=np.array([0.1, 0.0, 0.0]),
            generator_fn=generator_laplace_beltrami,
            generator_params={"alpha": 0.5},
        )


def test_reduced_model_config_rejects_non_3_vector_k() -> None:
    with pytest.raises(ValueError):
        ReducedModelConfig(
            step_length=1.0,
            time_step=1.0,
            k_vector=np.array([0.1, 0.0]),
            generator_fn=generator_laplace_beltrami,
            generator_params={"alpha": 0.5},
        )


def test_reduced_model_config_rejects_invalid_branch() -> None:
    with pytest.raises(ValueError):
        ReducedModelConfig(
            step_length=1.0,
            time_step=1.0,
            k_vector=np.array([0.1, 0.0, 0.0]),
            generator_fn=generator_laplace_beltrami,
            generator_params={"alpha": 0.5},
            branch=0,
        )


def test_reduced_model_config_rejects_non_callable_generator() -> None:
    with pytest.raises(TypeError):
        ReducedModelConfig(
            step_length=1.0,
            time_step=1.0,
            k_vector=np.array([0.1, 0.0, 0.0]),
            generator_fn="not callable",  # type: ignore[arg-type]
            generator_params={"alpha": 0.5},
        )


def test_reduced_model_config_defensively_copies_k_vector() -> None:
    k_vector = np.array([0.1, 0.0, 0.0])

    config = ReducedModelConfig(
        step_length=1.0,
        time_step=1.0,
        k_vector=k_vector,
        generator_fn=generator_laplace_beltrami,
        generator_params={"alpha": 0.5},
    )

    k_vector[0] = 999.0

    assert np.allclose(config.k_vector, np.array([0.1, 0.0, 0.0]))


def test_reduced_model_config_defensively_copies_generator_params() -> None:
    params = {"alpha": 0.5}

    config = ReducedModelConfig(
        step_length=1.0,
        time_step=1.0,
        k_vector=np.array([0.1, 0.0, 0.0]),
        generator_fn=generator_laplace_beltrami,
        generator_params=params,
    )

    params["alpha"] = 999.0

    assert config.generator_params == {"alpha": 0.5}


# ---------------------------------------------------------------------
# Convenience constructor and basic properties
# ---------------------------------------------------------------------


def test_make_reduced_axisymmetric_model_returns_model() -> None:
    model = make_basic_model()

    assert isinstance(model, ReducedAxisymmetricModel)


def test_model_basic_properties() -> None:
    model = make_basic_model()

    assert model.step_length == pytest.approx(1.0)
    assert model.time_step == pytest.approx(1.0)
    assert model.k == pytest.approx(0.2)
    assert np.allclose(model.k_direction, np.array([1.0, 0.0, 0.0]))
    assert model.branch == 1


def test_model_k_vector_property_returns_copy() -> None:
    model = make_basic_model()

    k_vector = model.k_vector
    k_vector[0] = 999.0

    assert np.allclose(model.k_vector, np.array([0.2, 0.0, 0.0]))
    assert model.k == pytest.approx(0.2)


def test_model_generator_params_property_returns_copy() -> None:
    model = make_basic_model()

    params = model.generator_params
    params["alpha"] = 999.0

    assert model.generator_params == {"alpha": 0.5}
    assert model.delta == pytest.approx(0.5)


def test_model_coupling_speed_and_delta() -> None:
    model = make_basic_model()

    assert model.coupling_speed == pytest.approx(1.0 / np.sqrt(3.0))

    # alpha = 0.5
    # f(0) = 0
    # f(1) = alpha * 2 = 1
    # Delta = (1 - 0) / 2 = 0.5
    assert model.delta == pytest.approx(0.5)


# ---------------------------------------------------------------------
# Hamiltonian, unitary, energies, and group velocity
# ---------------------------------------------------------------------


def test_model_hamiltonian_is_2_by_2_hermitian() -> None:
    model = make_basic_model()

    H = model.hamiltonian()

    assert H.shape == (2, 2)
    assert np.allclose(H, H.conj().T)


def test_model_hamiltonian_matches_expected_matrix() -> None:
    model = make_basic_model()

    H = model.hamiltonian()

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


def test_model_unitary_is_2_by_2_unitary() -> None:
    model = make_basic_model()

    U = model.unitary()
    identity = np.eye(2, dtype=complex)

    assert U.shape == (2, 2)
    assert np.allclose(U.conj().T @ U, identity)
    assert np.allclose(U @ U.conj().T, identity)


def test_model_energy_eigenvalues_match_reduced_dispersion() -> None:
    model = make_basic_model()

    E_minus, E_plus = model.energy_eigenvalues()

    fbar = 0.5
    delta = 0.5
    v = 1.0 / np.sqrt(3.0)
    k = 0.2
    spread = np.sqrt(delta**2 + (v * k) ** 2)

    assert E_minus == pytest.approx(fbar - spread)
    assert E_plus == pytest.approx(fbar + spread)


def test_model_group_velocity_matches_reduced_dispersion() -> None:
    model = make_basic_model()

    v_group = model.group_velocity()

    delta = 0.5
    v = 1.0 / np.sqrt(3.0)
    k = 0.2

    expected_speed = (v**2 * k) / np.sqrt(delta**2 + (v * k) ** 2)
    expected = np.array([expected_speed, 0.0, 0.0])

    assert v_group.shape == (3,)
    assert np.allclose(v_group, expected)


def test_model_group_velocity_changes_sign_for_lower_branch() -> None:
    upper = make_reduced_axisymmetric_model(
        step_length=1.0,
        time_step=1.0,
        k_vector=np.array([0.2, 0.0, 0.0]),
        generator_fn=generator_laplace_beltrami,
        branch=1,
        alpha=0.5,
    )

    lower = make_reduced_axisymmetric_model(
        step_length=1.0,
        time_step=1.0,
        k_vector=np.array([0.2, 0.0, 0.0]),
        generator_fn=generator_laplace_beltrami,
        branch=-1,
        alpha=0.5,
    )

    assert np.allclose(lower.group_velocity(), -upper.group_velocity())


def test_model_zero_k_has_zero_group_velocity() -> None:
    model = make_reduced_axisymmetric_model(
        step_length=1.0,
        time_step=1.0,
        k_vector=np.zeros(3),
        generator_fn=generator_laplace_beltrami,
        alpha=0.5,
    )

    assert model.k == pytest.approx(0.0)
    assert np.allclose(model.k_direction, np.zeros(3))
    assert np.allclose(model.group_velocity(), np.zeros(3))


# ---------------------------------------------------------------------
# State construction
# ---------------------------------------------------------------------


def test_make_state_defaults() -> None:
    model = make_basic_model()

    state = model.make_state()

    assert np.allclose(state.position, np.zeros(3))
    assert np.allclose(state.spinor, np.array([1.0 + 0.0j, 0.0 + 0.0j]))
    assert state.tick == 0


def test_make_state_normalizes_spinor_by_default() -> None:
    model = make_basic_model()

    state = model.make_state(
        spinor=np.array([2.0 + 0.0j, 0.0 + 0.0j]),
    )

    assert np.linalg.norm(state.spinor) == pytest.approx(1.0)
    assert np.allclose(state.spinor, np.array([1.0 + 0.0j, 0.0 + 0.0j]))


def test_make_state_can_preserve_unnormalized_spinor() -> None:
    model = make_basic_model()

    state = model.make_state(
        spinor=np.array([2.0 + 0.0j, 0.0 + 0.0j]),
        normalize=False,
    )

    assert np.linalg.norm(state.spinor) == pytest.approx(2.0)


def test_make_state_rejects_zero_spinor_when_normalizing() -> None:
    model = make_basic_model()

    with pytest.raises(ValueError):
        model.make_state(spinor=np.zeros(2, dtype=complex), normalize=True)


def test_make_state_rejects_non_2_component_spinor() -> None:
    model = make_basic_model()

    with pytest.raises(ValueError):
        model.make_state(spinor=np.ones(3, dtype=complex))


def test_make_state_preserves_tick() -> None:
    model = make_basic_model()

    state = model.make_state(tick=7)

    assert state.tick == 7


# ---------------------------------------------------------------------
# Evolution
# ---------------------------------------------------------------------


def test_evolve_default_returns_history_with_n_ticks_plus_one_states() -> None:
    model = make_basic_model()

    history = model.evolve_default(n_ticks=5)

    assert isinstance(history, list)
    assert len(history) == 6
    assert history[0].tick == 0
    assert history[-1].tick == 5


def test_evolve_default_can_return_final_state_only() -> None:
    model = make_basic_model()

    final_state = model.evolve_default(n_ticks=5, keep_history=False)

    assert not isinstance(final_state, list)
    assert final_state.tick == 5


def test_evolve_preserves_spinor_norm() -> None:
    model = make_basic_model()

    history = model.evolve_default(n_ticks=20)

    norms = np.array([np.linalg.norm(state.spinor) for state in history])

    assert np.allclose(norms, np.ones_like(norms))


def test_evolve_advances_packet_center_by_group_velocity() -> None:
    model = make_basic_model()

    n_ticks = 5
    history = model.evolve_default(n_ticks=n_ticks)

    expected_displacement = n_ticks * model.time_step * model.group_velocity()

    assert np.allclose(
        history[-1].position - history[0].position,
        expected_displacement,
    )


def test_evolve_default_respects_initial_position() -> None:
    model = make_basic_model()

    position = np.array([10.0, 20.0, 30.0])
    history = model.evolve_default(n_ticks=3, position=position)

    assert np.allclose(history[0].position, position)


def test_evolve_default_respects_initial_tick() -> None:
    model = make_basic_model()

    history = model.evolve_default(n_ticks=3, tick=10)

    assert history[0].tick == 10
    assert history[-1].tick == 13


# ---------------------------------------------------------------------
# Trajectory arrays and diagnostics
# ---------------------------------------------------------------------


def test_trajectory_arrays_returns_expected_keys_and_shapes() -> None:
    model = make_basic_model()

    history = model.evolve_default(n_ticks=4)
    arrays = model.trajectory_arrays(history)

    assert set(arrays.keys()) == {
        "positions",
        "spinors",
        "norms",
        "bloch_vectors",
    }

    assert arrays["positions"].shape == (5, 3)
    assert arrays["spinors"].shape == (5, 2)
    assert arrays["norms"].shape == (5,)
    assert arrays["bloch_vectors"].shape == (5, 3)


def test_diagnostics_returns_expected_core_values() -> None:
    model = make_basic_model()

    history = model.evolve_default(n_ticks=4)
    diagnostics = model.diagnostics(history)

    expected_keys = {
        "n_states",
        "n_ticks",
        "k",
        "k_direction",
        "step_length",
        "time_step",
        "coupling_speed",
        "delta",
        "branch",
        "energy_eigenvalues",
        "energy_gap",
        "group_velocity",
        "norm_initial",
        "norm_final",
        "max_norm_drift",
        "final_position",
        "final_displacement",
        "final_distance",
        "final_bloch_vector",
        "final_bloch_radius",
    }

    assert set(diagnostics.keys()) == expected_keys

    assert diagnostics["n_states"] == 5
    assert diagnostics["n_ticks"] == 4
    assert diagnostics["k"] == pytest.approx(0.2)
    assert diagnostics["step_length"] == pytest.approx(1.0)
    assert diagnostics["time_step"] == pytest.approx(1.0)
    assert diagnostics["delta"] == pytest.approx(0.5)
    assert diagnostics["branch"] == 1
    assert diagnostics["norm_initial"] == pytest.approx(1.0)
    assert diagnostics["norm_final"] == pytest.approx(1.0)
    assert diagnostics["max_norm_drift"] < 1e-12
    assert diagnostics["final_bloch_radius"] == pytest.approx(1.0)


def test_diagnostics_rejects_empty_history() -> None:
    model = make_basic_model()

    with pytest.raises(ValueError):
        model.diagnostics([])


def test_trajectory_arrays_rejects_empty_history() -> None:
    model = make_basic_model()

    with pytest.raises(ValueError):
        model.trajectory_arrays([])