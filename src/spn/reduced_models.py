from __future__ import annotations

"""
Convenience wrapper for Paper-2-style reduced SPN models.

This module packages the reduced low-sector ingredients into a small model
object that is easy to use in notebooks and tests.

It does not implement the full SPN field evolution on L^2(S^2). Instead, it
wraps the finite-dimensional reduced axisymmetric model:

    H_red = fbar I + Delta sigma_z + v k sigma_x

where:

    fbar  = (f(1) + f(0)) / 2
    Delta = (f(1) - f(0)) / 2
    v     = L / (tau sqrt(3))
    k     = ||k_vector||

The reduced packet centre is propagated using the group velocity derived from
the reduced dispersion relation.
"""

from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np

from .evolution import (
    Branch,
    ReducedState,
    evolve_n_ticks_axisymmetric,
    low_sector_delta,
    positions_from_history,
    reduced_axisymmetric_hamiltonian,
    reduced_axisymmetric_unitary,
    reduced_coupling_speed,
    reduced_energy_eigenvalues,
    reduced_group_velocity_vector,
    spinors_from_history,
    norms_from_history,
    max_norm_drift,
    final_displacement,
    final_distance_from_origin,
    wavevector_magnitude,
    wavevector_direction,
)

from .observables import (
    bloch_vector,
    bloch_radius,
    bloch_vectors_from_states,
)


GeneratorFn = Callable[..., float]


# ---------------------------------------------------------------------
# Reduced model configuration
# ---------------------------------------------------------------------


@dataclass(frozen=True)
class ReducedModelConfig:
    """
    Configuration for a Paper-2-style reduced SPN model.

    Parameters
    ----------
    step_length:
        Microscopic SPN step length L.

    time_step:
        Discrete time step tau.

    k_vector:
        3D wave-vector used to set both k = ||k_vector|| and the coarse
        packet propagation direction.

    generator_fn:
        Spectral generator f(l), usually defined in kernels.py.

    generator_params:
        Parameters passed to generator_fn.

    branch:
        Reduced dispersion branch. Use +1 for the upper branch and -1 for
        the lower branch.
    """

    step_length: float
    time_step: float
    k_vector: np.ndarray
    generator_fn: GeneratorFn
    generator_params: dict = field(default_factory=dict)
    branch: Branch = 1

    def __post_init__(self) -> None:
        k_vector = np.asarray(self.k_vector, dtype=float)

        if self.step_length < 0:
            raise ValueError("step_length must be non-negative")

        if self.time_step <= 0:
            raise ValueError("time_step must be positive")

        if k_vector.shape != (3,):
            raise ValueError("k_vector must be a 3-vector with shape (3,)")

        if not np.isfinite(k_vector).all():
            raise ValueError("k_vector must contain only finite values")

        if self.branch not in (-1, 1):
            raise ValueError("branch must be either +1 or -1")

        if not callable(self.generator_fn):
            raise TypeError("generator_fn must be callable")

        object.__setattr__(self, "k_vector", np.array(k_vector, dtype=float, copy=True))
        object.__setattr__(self, "generator_params", dict(self.generator_params))


# ---------------------------------------------------------------------
# Reduced model object
# ---------------------------------------------------------------------


@dataclass(frozen=True)
class ReducedAxisymmetricModel:
    """
    Convenience object for the Paper-2 axisymmetric reduced model.

    This model wraps the 2-component low-sector state:

        span{Y_00, Y_10}

    and exposes Hamiltonian, unitary, dispersion, group velocity, evolution,
    and basic diagnostics.
    """

    config: ReducedModelConfig

    # -----------------------------------------------------------------
    # Basic derived quantities
    # -----------------------------------------------------------------

    @property
    def step_length(self) -> float:
        return self.config.step_length

    @property
    def time_step(self) -> float:
        return self.config.time_step

    @property
    def k_vector(self) -> np.ndarray:
        return self.config.k_vector.copy()

    @property
    def generator_fn(self) -> GeneratorFn:
        return self.config.generator_fn

    @property
    def generator_params(self) -> dict:
        return dict(self.config.generator_params)

    @property
    def branch(self) -> Branch:
        return self.config.branch

    @property
    def k(self) -> float:
        """
        Return k = ||k_vector||.
        """
        return wavevector_magnitude(self.k_vector)

    @property
    def k_direction(self) -> np.ndarray:
        """
        Return k_vector / ||k_vector||, or zero vector if k = 0.
        """
        return wavevector_direction(self.k_vector)

    @property
    def coupling_speed(self) -> float:
        """
        Return v = L / (tau sqrt(3)).
        """
        return reduced_coupling_speed(self.step_length, self.time_step)

    @property
    def delta(self) -> float:
        """
        Return Delta = (f(1) - f(0)) / 2.

        This delegates to kernels.low_sector_gap through evolution.low_sector_delta.
        """
        return low_sector_delta(self.generator_fn, **self.generator_params)

    # -----------------------------------------------------------------
    # Reduced operators and dispersion
    # -----------------------------------------------------------------

    def hamiltonian(self) -> np.ndarray:
        """
        Return the 2x2 reduced axisymmetric Hamiltonian.
        """
        return reduced_axisymmetric_hamiltonian(
            k_vector=self.k_vector,
            step_length=self.step_length,
            time_step=self.time_step,
            generator_fn=self.generator_fn,
            **self.generator_params,
        )

    def unitary(self) -> np.ndarray:
        """
        Return the one-step reduced unitary U = exp(-i tau H_red).
        """
        return reduced_axisymmetric_unitary(
            k_vector=self.k_vector,
            step_length=self.step_length,
            time_step=self.time_step,
            generator_fn=self.generator_fn,
            **self.generator_params,
        )

    def energy_eigenvalues(self) -> tuple[float, float]:
        """
        Return E_minus, E_plus for the reduced Hamiltonian.
        """
        return reduced_energy_eigenvalues(
            k_vector=self.k_vector,
            step_length=self.step_length,
            time_step=self.time_step,
            generator_fn=self.generator_fn,
            **self.generator_params,
        )

    def group_velocity(self) -> np.ndarray:
        """
        Return the reduced packet-centre group velocity vector for this branch.
        """
        return reduced_group_velocity_vector(
            k_vector=self.k_vector,
            step_length=self.step_length,
            time_step=self.time_step,
            generator_fn=self.generator_fn,
            branch=self.branch,
            **self.generator_params,
        )

    # -----------------------------------------------------------------
    # State construction
    # -----------------------------------------------------------------

    def make_state(
        self,
        *,
        position: np.ndarray | None = None,
        spinor: np.ndarray | None = None,
        tick: int = 0,
        normalize: bool = True,
    ) -> ReducedState:
        """
        Construct a ReducedState for this model.

        Defaults
        --------
        position:
            [0, 0, 0]

        spinor:
            [1, 0], representing initial concentration in the first component
            of the chosen reduced basis ordering.
        """
        if position is None:
            position = np.zeros(3, dtype=float)

        if spinor is None:
            spinor = np.array([1.0 + 0.0j, 0.0 + 0.0j], dtype=complex)

        position = np.asarray(position, dtype=float)
        spinor = np.asarray(spinor, dtype=complex)

        if spinor.shape != (2,):
            raise ValueError("spinor must be a 2-component vector")

        if normalize:
            norm = np.linalg.norm(spinor)

            if norm == 0.0:
                raise ValueError("cannot normalize a zero spinor")

            spinor = spinor / norm

        return ReducedState(position=position, spinor=spinor, tick=tick)

    # -----------------------------------------------------------------
    # Evolution
    # -----------------------------------------------------------------

    def evolve(
        self,
        initial_state: ReducedState,
        *,
        n_ticks: int,
        keep_history: bool = True,
        renormalize: bool = False,
    ) -> list[ReducedState] | ReducedState:
        """
        Evolve an initial reduced state for n_ticks using this model.
        """
        return evolve_n_ticks_axisymmetric(
            initial_state,
            n_ticks=n_ticks,
            k_vector=self.k_vector,
            step_length=self.step_length,
            time_step=self.time_step,
            generator_fn=self.generator_fn,
            branch=self.branch,
            keep_history=keep_history,
            renormalize=renormalize,
            **self.generator_params,
        )

    def evolve_default(
        self,
        *,
        n_ticks: int,
        position: np.ndarray | None = None,
        spinor: np.ndarray | None = None,
        tick: int = 0,
        keep_history: bool = True,
        renormalize: bool = False,
    ) -> list[ReducedState] | ReducedState:
        """
        Construct a default initial state and evolve it.
        """
        state0 = self.make_state(position=position, spinor=spinor, tick=tick)

        return self.evolve(
            state0,
            n_ticks=n_ticks,
            keep_history=keep_history,
            renormalize=renormalize,
        )

    # -----------------------------------------------------------------
    # Diagnostics
    # -----------------------------------------------------------------

    def trajectory_arrays(
        self,
        history: Sequence[ReducedState],
    ) -> dict[str, np.ndarray]:
        """
        Return common trajectory arrays from a reduced-state history.
        """
        if len(history) == 0:
            raise ValueError("history must be non-empty")

        spinors = spinors_from_history(history)

        if spinors.ndim != 2 or spinors.shape[1] != 2:
            raise ValueError("history must contain 2-component reduced spinors")

        return {
            "positions": positions_from_history(history),
            "spinors": spinors,
            "norms": norms_from_history(history),
            "bloch_vectors": bloch_vectors_from_states(spinors),
        }

    def diagnostics(
        self,
        history: Sequence[ReducedState],
    ) -> dict[str, object]:
        """
        Return a compact diagnostic summary for a reduced trajectory.
        """
        if len(history) == 0:
            raise ValueError("history must be non-empty")

        spinors = spinors_from_history(history)

        if spinors.ndim != 2 or spinors.shape[1] != 2:
            raise ValueError("history must contain 2-component reduced spinors")

        final_spinor = spinors[-1]

        E_minus, E_plus = self.energy_eigenvalues()

        return {
            "n_states": len(history),
            "n_ticks": len(history) - 1,
            "k": self.k,
            "k_direction": self.k_direction,
            "step_length": self.step_length,
            "time_step": self.time_step,
            "coupling_speed": self.coupling_speed,
            "delta": self.delta,
            "branch": self.branch,
            "energy_eigenvalues": np.array([E_minus, E_plus], dtype=float),
            "energy_gap": float(E_plus - E_minus),
            "group_velocity": self.group_velocity(),
            "norm_initial": float(np.linalg.norm(spinors[0])),
            "norm_final": float(np.linalg.norm(final_spinor)),
            "max_norm_drift": max_norm_drift(history),
            "final_position": np.asarray(history[-1].position, dtype=float),
            "final_displacement": final_displacement(history),
            "final_distance": final_distance_from_origin(history),
            "final_bloch_vector": bloch_vector(final_spinor),
            "final_bloch_radius": bloch_radius(final_spinor),
        }


# ---------------------------------------------------------------------
# Convenience constructors
# ---------------------------------------------------------------------


def make_reduced_axisymmetric_model(
    *,
    step_length: float,
    time_step: float,
    k_vector: np.ndarray,
    generator_fn: GeneratorFn,
    branch: Branch = 1,
    **generator_params,
) -> ReducedAxisymmetricModel:
    """
    Construct a ReducedAxisymmetricModel from simple arguments.

    Example
    -------
    model = make_reduced_axisymmetric_model(
        step_length=1.0,
        time_step=1.0,
        k_vector=np.array([0.1, 0.0, 0.0]),
        generator_fn=generator_laplace_beltrami,
        alpha=0.5,
    )
    """
    config = ReducedModelConfig(
        step_length=step_length,
        time_step=time_step,
        k_vector=k_vector,
        generator_fn=generator_fn,
        generator_params=generator_params,
        branch=branch,
    )

    return ReducedAxisymmetricModel(config=config)