from __future__ import annotations

import numpy as np
import pytest

from spn.transport import (
    unit_direction,
    transport_displacement,
    transported_position,
    transport_phase,
    microscopic_speed,
    net_displacement,
    average_displacement_per_tick,
    effective_velocity,
    effective_speed,
)


# ---------------------------------------------------------------------
# Direction vectors
# ---------------------------------------------------------------------


@pytest.mark.parametrize(
    "theta, phi, expected",
    [
        (0.0, 0.0, np.array([0.0, 0.0, 1.0])),
        (np.pi / 2.0, 0.0, np.array([1.0, 0.0, 0.0])),
        (np.pi / 2.0, np.pi / 2.0, np.array([0.0, 1.0, 0.0])),
        (np.pi, 0.0, np.array([0.0, 0.0, -1.0])),
    ],
)
def test_unit_direction_special_cases(
    theta: float,
    phi: float,
    expected: np.ndarray,
) -> None:
    direction = unit_direction(theta, phi)

    assert direction.shape == (3,)
    assert np.allclose(direction, expected, atol=1e-12)


@pytest.mark.parametrize(
    "theta, phi",
    [
        (0.0, 0.0),
        (np.pi / 4.0, np.pi / 3.0),
        (np.pi / 2.0, np.pi),
        (3.0 * np.pi / 4.0, 5.0 * np.pi / 3.0),
        (np.pi, 0.0),
    ],
)
def test_unit_direction_has_unit_norm(theta: float, phi: float) -> None:
    direction = unit_direction(theta, phi)

    assert direction.shape == (3,)
    assert np.linalg.norm(direction) == pytest.approx(1.0)


# ---------------------------------------------------------------------
# Physical displacement
# ---------------------------------------------------------------------


def test_transport_displacement_has_length_step_length() -> None:
    theta = np.pi / 3.0
    phi = np.pi / 5.0
    step_length = 2.5

    displacement = transport_displacement(theta, phi, step_length)

    assert displacement.shape == (3,)
    assert np.linalg.norm(displacement) == pytest.approx(step_length)


def test_transport_displacement_equals_step_length_times_direction() -> None:
    theta = np.pi / 2.0
    phi = 0.0
    step_length = 3.0

    displacement = transport_displacement(theta, phi, step_length)

    assert np.allclose(displacement, np.array([3.0, 0.0, 0.0]))


def test_transport_displacement_rejects_negative_step_length() -> None:
    with pytest.raises(ValueError):
        transport_displacement(
            theta=np.pi / 2.0,
            phi=0.0,
            step_length=-1.0,
        )


def test_transported_position_uses_backward_shift() -> None:
    position = np.array([1.0, 2.0, 3.0])
    theta = np.pi / 2.0
    phi = 0.0
    step_length = 4.0

    new_position = transported_position(
        position,
        theta,
        phi,
        step_length,
    )

    # transport.py defines x -> x - L n_hat(Omega)
    expected = np.array([-3.0, 2.0, 3.0])

    assert new_position.shape == (3,)
    assert np.allclose(new_position, expected)


def test_transported_position_does_not_mutate_input_position() -> None:
    position = np.array([1.0, 2.0, 3.0])
    original = position.copy()

    _ = transported_position(
        position,
        theta=np.pi / 2.0,
        phi=0.0,
        step_length=4.0,
    )

    assert np.allclose(position, original)


def test_transported_position_rejects_non_3_vector_position() -> None:
    with pytest.raises(ValueError):
        transported_position(
            np.array([1.0, 2.0]),
            theta=np.pi / 2.0,
            phi=0.0,
            step_length=1.0,
        )


# ---------------------------------------------------------------------
# Momentum-space phase
# ---------------------------------------------------------------------


def test_transport_phase_matches_expected_exponential() -> None:
    k = np.array([2.0, 0.0, 0.0])
    theta = np.pi / 2.0
    phi = 0.0
    step_length = 0.25

    # displacement = [0.25, 0, 0]
    # k dot displacement = 2.0 * 0.25 = 0.5
    expected = np.exp(-1j * 0.5)

    phase = transport_phase(
        k=k,
        theta=theta,
        phi=phi,
        step_length=step_length,
    )

    assert phase == pytest.approx(expected)


def test_transport_phase_has_unit_modulus() -> None:
    k = np.array([0.3, -0.4, 0.5])
    theta = np.pi / 3.0
    phi = np.pi / 7.0
    step_length = 1.2

    phase = transport_phase(
        k=k,
        theta=theta,
        phi=phi,
        step_length=step_length,
    )

    assert abs(phase) == pytest.approx(1.0)


def test_transport_phase_zero_k_is_one() -> None:
    phase = transport_phase(
        k=np.zeros(3),
        theta=np.pi / 3.0,
        phi=np.pi / 7.0,
        step_length=1.2,
    )

    assert phase == pytest.approx(1.0 + 0.0j)


def test_transport_phase_rejects_non_3_vector_k() -> None:
    with pytest.raises(ValueError):
        transport_phase(
            k=np.array([1.0, 2.0]),
            theta=np.pi / 2.0,
            phi=0.0,
            step_length=1.0,
        )


# ---------------------------------------------------------------------
# Microscopic speed
# ---------------------------------------------------------------------


def test_microscopic_speed_returns_step_over_time() -> None:
    assert microscopic_speed(step_length=3.0, time_step=2.0) == pytest.approx(1.5)


def test_microscopic_speed_rejects_negative_step_length() -> None:
    with pytest.raises(ValueError):
        microscopic_speed(step_length=-1.0, time_step=1.0)


def test_microscopic_speed_rejects_nonpositive_time_step() -> None:
    with pytest.raises(ValueError):
        microscopic_speed(step_length=1.0, time_step=0.0)


# ---------------------------------------------------------------------
# Coarse-grained path / packet diagnostics
# ---------------------------------------------------------------------


def test_net_displacement_sums_step_vectors() -> None:
    step_vectors = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 2.0, 0.0],
            [-1.0, 0.0, 3.0],
        ],
        dtype=float,
    )

    expected = np.array([0.0, 2.0, 3.0])

    assert np.allclose(net_displacement(step_vectors), expected)


def test_net_displacement_rejects_wrong_shape() -> None:
    with pytest.raises(ValueError):
        net_displacement(np.array([1.0, 2.0, 3.0]))


def test_average_displacement_per_tick_returns_mean_step() -> None:
    step_vectors = np.array(
        [
            [1.0, 0.0, 0.0],
            [3.0, 2.0, 0.0],
        ],
        dtype=float,
    )

    expected = np.array([2.0, 1.0, 0.0])

    assert np.allclose(average_displacement_per_tick(step_vectors), expected)


def test_average_displacement_per_tick_rejects_empty_array() -> None:
    with pytest.raises(ValueError):
        average_displacement_per_tick(np.empty((0, 3)))


def test_effective_velocity_divides_average_displacement_by_time_step() -> None:
    step_vectors = np.array(
        [
            [2.0, 0.0, 0.0],
            [4.0, 2.0, 0.0],
        ],
        dtype=float,
    )
    time_step = 2.0

    # average displacement = [3, 1, 0]
    # effective velocity = [3, 1, 0] / 2
    expected = np.array([1.5, 0.5, 0.0])

    assert np.allclose(
        effective_velocity(step_vectors, time_step=time_step),
        expected,
    )


def test_effective_velocity_rejects_nonpositive_time_step() -> None:
    with pytest.raises(ValueError):
        effective_velocity(
            np.array([[1.0, 0.0, 0.0]]),
            time_step=0.0,
        )


def test_effective_speed_is_norm_of_effective_velocity() -> None:
    step_vectors = np.array(
        [
            [3.0, 4.0, 0.0],
            [3.0, 4.0, 0.0],
        ],
        dtype=float,
    )

    # average displacement = [3, 4, 0]
    # time_step = 1
    # speed = 5
    assert effective_speed(step_vectors, time_step=1.0) == pytest.approx(5.0)