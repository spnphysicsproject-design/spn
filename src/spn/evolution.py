from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

import numpy as np

from .kernels import (
    low_sector_gap,
    low_sector_mean,
)

# ---------------------------------------------------------------------
# Reduced low-sector state
# ---------------------------------------------------------------------


@dataclass(frozen=True)
class ReducedState:
    """
    Reduced SPN state at one discrete tick.

    This represents a low-sector internal state rather than a full
    directional wavefunction on L^2(S^2). It is aligned with the reduced
    Paper-2 evolution picture, not the full Paper-1 field formulation.

    Parameters
    ----------
    position:
        Coarse packet-centre position vector of shape (3,).

    spinor:
        Complex reduced-state vector. In the simplest Paper-2 setting this
        is a 2-component vector for the axisymmetric low sector
        span{Y_00, Y_10}.

    tick:
        Integer discrete time-step label.
    """

    position: np.ndarray
    spinor: np.ndarray
    tick: int = 0

    def __post_init__(self) -> None:
        position = np.asarray(self.position, dtype=float)
        spinor = np.asarray(self.spinor, dtype=complex)

        if position.shape != (3,):
            raise ValueError("position must be a 3-vector with shape (3,)")

        if not np.isfinite(position).all():
            raise ValueError("position must contain only finite values")

        if spinor.ndim != 1:
            raise ValueError("spinor must be a 1D complex vector")

        if spinor.size == 0:
            raise ValueError("spinor must be non-empty")

        if not np.isfinite(spinor.real).all() or not np.isfinite(spinor.imag).all():
            raise ValueError("spinor must contain only finite values")

        if self.tick < 0:
            raise ValueError("tick must be non-negative")

        object.__setattr__(self, "position", position)
        object.__setattr__(self, "spinor", spinor)


# ---------------------------------------------------------------------
# Basic matrix helpers
# ---------------------------------------------------------------------


def pauli_x() -> np.ndarray:
    """
    Return the Pauli sigma_x matrix.
    """
    return np.array(
        [
            [0.0, 1.0],
            [1.0, 0.0],
        ],
        dtype=complex,
    )


def pauli_z() -> np.ndarray:
    """
    Return the Pauli sigma_z matrix.
    """
    return np.array(
        [
            [1.0, 0.0],
            [0.0, -1.0],
        ],
        dtype=complex,
    )


def identity(n: int) -> np.ndarray:
    """
    Return the n x n complex identity matrix.
    """
    if n <= 0:
        raise ValueError("n must be positive")

    return np.eye(n, dtype=complex)


def is_hermitian(H: np.ndarray, *, atol: float = 1e-12) -> bool:
    """
    Return True if H is Hermitian to numerical tolerance.
    """
    H = np.asarray(H, dtype=complex)

    if H.ndim != 2 or H.shape[0] != H.shape[1]:
        return False

    return bool(np.allclose(H, H.conj().T, atol=atol))


def unitary_from_hermitian(H: np.ndarray, tau: float) -> np.ndarray:
    """
    Construct U = exp(-i tau H) from a Hermitian matrix H.

    Uses spectral decomposition. Suitable for small reduced sectors.
    """
    H = np.asarray(H, dtype=complex)

    if H.ndim != 2 or H.shape[0] != H.shape[1]:
        raise ValueError("H must be a square matrix")

    if tau < 0:
        raise ValueError("tau must be non-negative")

    if not is_hermitian(H):
        raise ValueError("H must be Hermitian")

    evals, evecs = np.linalg.eigh(H)
    phases = np.exp(-1j * tau * evals)

    return evecs @ np.diag(phases) @ evecs.conj().T


# ---------------------------------------------------------------------
# Wave-vector helpers
# ---------------------------------------------------------------------


def wavevector_magnitude(k_vector: np.ndarray) -> float:
    """
    Return ||k_vector|| after validating that it is a finite 3-vector.
    """
    k_vector = np.asarray(k_vector, dtype=float)

    if k_vector.shape != (3,):
        raise ValueError("k_vector must be a 3-vector with shape (3,)")

    if not np.isfinite(k_vector).all():
        raise ValueError("k_vector must contain only finite values")

    return float(np.linalg.norm(k_vector))


def wavevector_direction(k_vector: np.ndarray) -> np.ndarray:
    """
    Return the unit direction of k_vector.

    If ||k_vector|| = 0, return the zero vector. This represents a packet
    with no preferred coarse propagation direction.
    """
    k_vector = np.asarray(k_vector, dtype=float)
    k = wavevector_magnitude(k_vector)

    if k == 0.0:
        return np.zeros(3, dtype=float)

    return k_vector / k


# ---------------------------------------------------------------------
# Reduced Paper-2 Hamiltonian helpers
# ---------------------------------------------------------------------


def low_sector_delta(generator_fn, **generator_params) -> float:
    """
    Return Delta = (f(1) - f(0)) / 2.

    Important:
        kernels.low_sector_gap already returns the Paper-2 half-gap Delta,
        not the full difference f(1) - f(0). Therefore this function
        delegates directly to low_sector_gap and does not multiply by 0.5.
    """
    return low_sector_gap(generator_fn, **generator_params)


def reduced_coupling_speed(step_length: float, time_step: float) -> float:
    """
    Return v = L / (tau * sqrt(3)) used in the reduced Hamiltonian.

    This is not the full microscopic speed L / tau. It is the reduced-sector
    coupling speed appearing in the axisymmetric 2x2 Hamiltonian.
    """
    if step_length < 0:
        raise ValueError("step_length must be non-negative")

    if time_step <= 0:
        raise ValueError("time_step must be positive")

    return float(step_length / (time_step * np.sqrt(3.0)))


def reduced_axisymmetric_hamiltonian(
    *,
    k_vector: np.ndarray,
    step_length: float,
    time_step: float,
    generator_fn,
    **generator_params,
) -> np.ndarray:
    """
    Build the Paper-2 reduced 2x2 Hamiltonian in the axisymmetric low sector:

        H_red = fbar I + Delta sigma_z + v k sigma_x

    where:

        fbar  = (f(1) + f(0)) / 2
        Delta = (f(1) - f(0)) / 2
        v     = L / (tau sqrt(3))
        k     = ||k_vector||

    This corresponds to the span{Y_00, Y_10} reduction.
    """
    k = wavevector_magnitude(k_vector)

    fbar = low_sector_mean(generator_fn, **generator_params)
    delta = low_sector_delta(generator_fn, **generator_params)
    v = reduced_coupling_speed(step_length, time_step)

    H = fbar * identity(2) + delta * pauli_z() + (v * k) * pauli_x()

    if not is_hermitian(H):
        raise ValueError("reduced Hamiltonian must be Hermitian")

    return H


def reduced_axisymmetric_unitary(
    *,
    k_vector: np.ndarray,
    step_length: float,
    time_step: float,
    generator_fn,
    **generator_params,
) -> np.ndarray:
    """
    Construct the one-step reduced unitary:

        U = exp(-i tau H_red)
    """
    H = reduced_axisymmetric_hamiltonian(
        k_vector=k_vector,
        step_length=step_length,
        time_step=time_step,
        generator_fn=generator_fn,
        **generator_params,
    )

    return unitary_from_hermitian(H, tau=time_step)


# ---------------------------------------------------------------------
# Reduced dispersion and group velocity
# ---------------------------------------------------------------------


Branch = Literal[1, -1]


def reduced_energy_eigenvalues(
    *,
    k_vector: np.ndarray,
    step_length: float,
    time_step: float,
    generator_fn,
    **generator_params,
) -> tuple[float, float]:
    """
    Return the reduced eigenvalues E_minus, E_plus for:

        H_red = fbar I + Delta sigma_z + v k sigma_x

    with:

        E_± = fbar ± sqrt(Delta^2 + v^2 k^2)
    """
    k = wavevector_magnitude(k_vector)

    fbar = low_sector_mean(generator_fn, **generator_params)
    delta = low_sector_delta(generator_fn, **generator_params)
    v = reduced_coupling_speed(step_length, time_step)

    spread = float(np.sqrt(delta**2 + (v * k) ** 2))

    return fbar - spread, fbar + spread


def reduced_group_velocity_signed(
    *,
    k_vector: np.ndarray,
    step_length: float,
    time_step: float,
    generator_fn,
    branch: Branch = 1,
    **generator_params,
) -> float:
    """
    Return the reduced group velocity dE/dk for one branch.

    For:

        E_±(k) = fbar ± sqrt(Delta^2 + v^2 k^2)

    the group velocity is:

        dE_±/dk = ± v^2 k / sqrt(Delta^2 + v^2 k^2)

    This is a reduced packet-centre velocity, not the microscopic SPN
    pulse speed L / tau.
    """
    if branch not in (-1, 1):
        raise ValueError("branch must be either +1 or -1")

    k = wavevector_magnitude(k_vector)

    if k == 0.0:
        return 0.0

    delta = low_sector_delta(generator_fn, **generator_params)
    v = reduced_coupling_speed(step_length, time_step)

    denom = float(np.sqrt(delta**2 + (v * k) ** 2))

    if denom == 0.0:
        return 0.0

    return float(branch * (v**2 * k) / denom)


def reduced_group_velocity_vector(
    *,
    k_vector: np.ndarray,
    step_length: float,
    time_step: float,
    generator_fn,
    branch: Branch = 1,
    **generator_params,
) -> np.ndarray:
    """
    Return the reduced packet-centre group velocity vector.

    Direction is taken from k_vector. Magnitude is determined by the
    reduced Hamiltonian dispersion relation.
    """
    direction = wavevector_direction(k_vector)

    speed = reduced_group_velocity_signed(
        k_vector=k_vector,
        step_length=step_length,
        time_step=time_step,
        generator_fn=generator_fn,
        branch=branch,
        **generator_params,
    )

    return speed * direction


# ---------------------------------------------------------------------
# Reduced transport of packet centre
# ---------------------------------------------------------------------


def propagate_packet_center(
    position: np.ndarray,
    *,
    k_vector: np.ndarray,
    step_length: float,
    time_step: float,
    generator_fn,
    branch: Branch = 1,
    **generator_params,
) -> np.ndarray:
    """
    Propagate the reduced packet centre forward by one tick.

    This uses group-velocity transport:

        x_{t+1} = x_t + tau * v_group

    where v_group is derived from the reduced Hamiltonian dispersion.

    This is not the exact configuration-space field update:

        psi(x, Omega) -> psi(x - L n_hat(Omega), Omega)

    which belongs to the full SPN field formulation.
    """
    position = np.asarray(position, dtype=float)

    if position.shape != (3,):
        raise ValueError("position must be a 3-vector with shape (3,)")

    if not np.isfinite(position).all():
        raise ValueError("position must contain only finite values")

    v_group = reduced_group_velocity_vector(
        k_vector=k_vector,
        step_length=step_length,
        time_step=time_step,
        generator_fn=generator_fn,
        branch=branch,
        **generator_params,
    )

    return position + time_step * v_group


# ---------------------------------------------------------------------
# One-tick reduced evolution
# ---------------------------------------------------------------------


def evolve_one_tick_axisymmetric(
    state: ReducedState,
    *,
    k_vector: np.ndarray,
    step_length: float,
    time_step: float,
    generator_fn,
    branch: Branch = 1,
    renormalize: bool = False,
    **generator_params,
) -> ReducedState:
    """
    Evolve a reduced 2D axisymmetric low-sector state by one tick.

    Update order:
        1. Evolve the reduced internal spinor with U = exp(-i tau H_red).
        2. Propagate the packet centre using reduced group velocity.
        3. Increment tick.

    This is aligned with the reduced Paper-2 picture, not the full
    configuration-space SPN field evolution.

    Notes:
        - k is derived internally as ||k_vector|| to avoid inconsistency.
        - The packet-centre speed is not assumed to be L / tau.
        - The microscopic fixed-step assumption belongs to the underlying
          SPN field picture; this reduced model tracks only the effective
          packet centre.
    """
    spinor = np.asarray(state.spinor, dtype=complex)

    if spinor.shape != (2,):
        raise ValueError(
            "axisymmetric reduced evolution expects a 2-component spinor"
        )

    U = reduced_axisymmetric_unitary(
        k_vector=k_vector,
        step_length=step_length,
        time_step=time_step,
        generator_fn=generator_fn,
        **generator_params,
    )

    spinor_next = U @ spinor

    if renormalize:
        norm = np.linalg.norm(spinor_next)

        if norm == 0.0:
            raise ValueError("cannot renormalize a zero spinor")

        spinor_next = spinor_next / norm

    position_next = propagate_packet_center(
        state.position,
        k_vector=k_vector,
        step_length=step_length,
        time_step=time_step,
        generator_fn=generator_fn,
        branch=branch,
        **generator_params,
    )

    return ReducedState(
        position=position_next,
        spinor=spinor_next,
        tick=state.tick + 1,
    )


def evolve_n_ticks_axisymmetric(
    initial_state: ReducedState,
    *,
    n_ticks: int,
    k_vector: np.ndarray,
    step_length: float,
    time_step: float,
    generator_fn,
    branch: Branch = 1,
    keep_history: bool = True,
    renormalize: bool = False,
    **generator_params,
) -> list[ReducedState] | ReducedState:
    """
    Evolve a reduced 2D axisymmetric low-sector state for n_ticks.
    """
    if n_ticks < 0:
        raise ValueError("n_ticks must be non-negative")

    state = initial_state

    if keep_history:
        history: list[ReducedState] = [state]

        for _ in range(n_ticks):
            state = evolve_one_tick_axisymmetric(
                state,
                k_vector=k_vector,
                step_length=step_length,
                time_step=time_step,
                generator_fn=generator_fn,
                branch=branch,
                renormalize=renormalize,
                **generator_params,
            )
            history.append(state)

        return history

    for _ in range(n_ticks):
        state = evolve_one_tick_axisymmetric(
            state,
            k_vector=k_vector,
            step_length=step_length,
            time_step=time_step,
            generator_fn=generator_fn,
            branch=branch,
            renormalize=renormalize,
            **generator_params,
        )

    return state


# ---------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------


def positions_from_history(history: Sequence[ReducedState]) -> np.ndarray:
    """
    Extract packet-centre positions from a reduced trajectory.

    Returns
    -------
    np.ndarray
        Array of shape (n_states, 3).
    """
    return np.array([state.position for state in history], dtype=float)


def spinors_from_history(history: Sequence[ReducedState]) -> np.ndarray:
    """
    Extract reduced spinors from a reduced trajectory.

    Returns
    -------
    np.ndarray
        Array of shape (n_states, n_components).
    """
    return np.array([state.spinor for state in history], dtype=complex)


def norms_from_history(history: Sequence[ReducedState]) -> np.ndarray:
    """
    Extract reduced-state norms ||spinor|| from a trajectory.
    """
    return np.array(
        [np.linalg.norm(state.spinor) for state in history],
        dtype=float,
    )


def max_norm_drift(history: Sequence[ReducedState]) -> float:
    """
    Return the maximum absolute drift in spinor norm relative to the
    initial state's norm.
    """
    norms = norms_from_history(history)

    if norms.size == 0:
        raise ValueError("history must be non-empty")

    return float(np.max(np.abs(norms - norms[0])))


def final_displacement(history: Sequence[ReducedState]) -> np.ndarray:
    """
    Return final packet-centre displacement relative to the initial state.
    """
    positions = positions_from_history(history)

    if positions.shape[0] == 0:
        raise ValueError("history must be non-empty")

    return positions[-1] - positions[0]


def final_distance_from_origin(history: Sequence[ReducedState]) -> float:
    """
    Return ||x_final - x_initial|| for the packet centre.
    """
    return float(np.linalg.norm(final_displacement(history)))