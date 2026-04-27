from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------
# Geometric transport on the sphere of directions
# ---------------------------------------------------------------------

def unit_direction(theta: float, phi: float) -> np.ndarray:
    """
    Return the 3D unit direction vector n_hat(Omega) for spherical angles
    theta in [0, pi], phi in [0, 2pi).

    Convention:
        x = sin(theta) cos(phi)
        y = sin(theta) sin(phi)
        z = cos(theta)

    This matches the SPN directional geometry used in papers 1 and 2.
    """
    return np.array(
        [
            np.sin(theta) * np.cos(phi),
            np.sin(theta) * np.sin(phi),
            np.cos(theta),
        ],
        dtype=float,
    )


def transport_displacement(theta: float, phi: float, step_length: float) -> np.ndarray:
    """
    Return the one-step transport displacement vector:
        Δx = L * n_hat(Omega)

    Parameters
    ----------
    theta, phi : float
        Spherical direction angles.
    step_length : float
        Microscopic one-step transport length L.
    """
    if step_length < 0:
        raise ValueError("step_length must be non-negative")
    return step_length * unit_direction(theta, phi)


def transported_position(
    x: np.ndarray,
    theta: float,
    phi: float,
    step_length: float,
) -> np.ndarray:
    """
    Return the backward-shifted position used in the configuration-space
    SPN update:
        x -> x - L * n_hat(Omega)

    This corresponds to sampling the previous state one step back along
    direction Omega.

    Parameters
    ----------
    x : np.ndarray
        3D position vector of shape (3,).
    theta, phi : float
        Spherical direction angles.
    step_length : float
        Microscopic one-step transport length L.
    """
    x = np.asarray(x, dtype=float)
    if x.shape != (3,):
        raise ValueError("x must be a 3-vector with shape (3,)")
    return x - transport_displacement(theta, phi, step_length)


# ---------------------------------------------------------------------
# Momentum-space transport
# ---------------------------------------------------------------------

def transport_phase(
    k: np.ndarray,
    theta: float,
    phi: float,
    step_length: float,
) -> complex:
    """
    Return the momentum-space transport phase:
        exp(-i k · (L n_hat(Omega)))

    This is the momentum-space representation of one-step transport used
    in the paper-2 low-sector derivations.

    Parameters
    ----------
    k : np.ndarray
        3D wavevector of shape (3,).
    theta, phi : float
        Spherical direction angles.
    step_length : float
        Microscopic one-step transport length L.
    """
    k = np.asarray(k, dtype=float)
    if k.shape != (3,):
        raise ValueError("k must be a 3-vector with shape (3,)")
    displacement = transport_displacement(theta, phi, step_length)
    return np.exp(-1j * np.dot(k, displacement))


# ---------------------------------------------------------------------
# Microscopic transport speed
# ---------------------------------------------------------------------

def microscopic_speed(step_length: float, time_step: float) -> float:
    """
    Return the microscopic transport speed:
        c = L / tau

    In the SPN picture, the underlying transport is lightlike at this speed.

    Parameters
    ----------
    step_length : float
        Microscopic one-step transport length L.
    time_step : float
        Microscopic time step tau.
    """
    if step_length < 0:
        raise ValueError("step_length must be non-negative")
    if time_step <= 0:
        raise ValueError("time_step must be positive")
    return step_length / time_step


# ---------------------------------------------------------------------
# Coarse-grained path/packet diagnostics
# ---------------------------------------------------------------------

def net_displacement(step_vectors: np.ndarray) -> np.ndarray:
    """
    Sum a sequence of step displacement vectors to get the net displacement:
        Δx_net = sum_i Δx_i

    Parameters
    ----------
    step_vectors : np.ndarray
        Array of shape (N, 3), where each row is a one-step displacement.
    """
    step_vectors = np.asarray(step_vectors, dtype=float)
    if step_vectors.ndim != 2 or step_vectors.shape[1] != 3:
        raise ValueError("step_vectors must have shape (N, 3)")
    return np.sum(step_vectors, axis=0)


def average_displacement_per_tick(step_vectors: np.ndarray) -> np.ndarray:
    """
    Compute the average displacement per tick:
        <Δx> = (1/N) sum_i Δx_i

    This is a geometric coarse-grained quantity. It becomes a physical
    velocity only after division by the microscopic time step.
    """
    step_vectors = np.asarray(step_vectors, dtype=float)
    if step_vectors.ndim != 2 or step_vectors.shape[1] != 3:
        raise ValueError("step_vectors must have shape (N, 3)")
    if len(step_vectors) == 0:
        raise ValueError("step_vectors must be non-empty")
    return np.mean(step_vectors, axis=0)


def effective_velocity(step_vectors: np.ndarray, time_step: float = 1.0) -> np.ndarray:
    """
    Compute the coarse-grained effective velocity:
        v_eff = (1 / (N * tau)) sum_i Δx_i

    If time_step=1, this reduces to the average displacement per tick.

    Parameters
    ----------
    step_vectors : np.ndarray
        Array of shape (N, 3), where each row is a one-step displacement.
    time_step : float, default=1.0
        Microscopic time step tau.
    """
    if time_step <= 0:
        raise ValueError("time_step must be positive")
    return average_displacement_per_tick(step_vectors) / time_step


def effective_speed(step_vectors: np.ndarray, time_step: float = 1.0) -> float:
    """
    Magnitude of the coarse-grained effective velocity:
        |v_eff|

    Parameters
    ----------
    step_vectors : np.ndarray
        Array of shape (N, 3), where each row is a one-step displacement.
    time_step : float, default=1.0
        Microscopic time step tau.
    """
    return float(np.linalg.norm(effective_velocity(step_vectors, time_step=time_step)))