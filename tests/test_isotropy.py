from __future__ import annotations

import numpy as np

from spn.evolution import (
    reduced_axisymmetric_hamiltonian,
    reduced_coupling_speed,
    reduced_energy_eigenvalues,
    reduced_group_velocity_signed,
    reduced_group_velocity_vector,
)
from spn.kernels import generator_laplace_beltrami


def same_magnitude_k_vectors(k: float) -> list[np.ndarray]:
    """
    Return several 3D k-vectors with the same Euclidean magnitude.

    These are used to check that the reduced Paper-2 model depends on
    ||k_vector||, not on the particular direction of k_vector.
    """
    return [
        np.array([k, 0.0, 0.0], dtype=float),
        np.array([0.0, k, 0.0], dtype=float),
        np.array([0.0, 0.0, k], dtype=float),
        np.array([k / np.sqrt(2.0), k / np.sqrt(2.0), 0.0], dtype=float),
        np.array([k / np.sqrt(3.0), k / np.sqrt(3.0), k / np.sqrt(3.0)], dtype=float),
    ]


def test_same_magnitude_k_vectors_really_have_same_norm() -> None:
    """
    Guard the test fixture itself.

    If these vectors are edited incorrectly, the isotropy tests below become
    ambiguous.
    """
    k = 0.4
    norms = [np.linalg.norm(vec) for vec in same_magnitude_k_vectors(k)]

    for norm in norms:
        np.testing.assert_allclose(norm, k, rtol=1e-14, atol=1e-14)


def test_reduced_coupling_speed_matches_paper2_convention() -> None:
    """
    The reduced coupling speed should be v = L / (tau * sqrt(3)),
    not the microscopic speed L / tau.
    """
    step_length = 1.2
    time_step = 0.8

    expected = step_length / (time_step * np.sqrt(3.0))
    actual = reduced_coupling_speed(
        step_length=step_length,
        time_step=time_step,
    )

    np.testing.assert_allclose(actual, expected, rtol=1e-14, atol=1e-14)


def test_reduced_hamiltonian_matrix_depends_only_on_k_magnitude() -> None:
    """
    In the current reduced model:

        H_red = fbar I + Delta sigma_z + v k sigma_x

    Since k is defined as ||k_vector||, the full 2x2 Hamiltonian matrix should
    be identical for different k-vector directions with the same magnitude.
    """
    step_length = 1.2
    time_step = 0.8
    alpha = 0.35
    k = 0.6

    hamiltonians = []
    for k_vector in same_magnitude_k_vectors(k):
        H = reduced_axisymmetric_hamiltonian(
            k_vector=k_vector,
            step_length=step_length,
            time_step=time_step,
            generator_fn=generator_laplace_beltrami,
            alpha=alpha,
        )
        hamiltonians.append(H)

    reference = hamiltonians[0]
    for H in hamiltonians[1:]:
        np.testing.assert_allclose(H, reference, rtol=1e-14, atol=1e-14)


def test_reduced_energy_eigenvalues_depend_only_on_k_magnitude() -> None:
    """
    Holding ||k_vector|| fixed while rotating the vector should leave the
    reduced energy eigenvalues unchanged.
    """
    step_length = 1.0
    time_step = 1.0
    alpha = 0.25
    k = 0.4

    spectra = []
    for k_vector in same_magnitude_k_vectors(k):
        eigvals = reduced_energy_eigenvalues(
            k_vector=k_vector,
            step_length=step_length,
            time_step=time_step,
            generator_fn=generator_laplace_beltrami,
            alpha=alpha,
        )
        spectra.append(np.array(eigvals, dtype=float))

    reference = spectra[0]
    for eigvals in spectra[1:]:
        np.testing.assert_allclose(eigvals, reference, rtol=1e-14, atol=1e-14)


def test_reduced_group_speed_depends_only_on_k_magnitude() -> None:
    """
    The signed reduced group-speed for a given branch should depend only on
    ||k_vector||, not on the direction of k_vector.
    """
    step_length = 1.0
    time_step = 1.0
    alpha = 0.25
    k = 0.5

    speeds = []
    for k_vector in same_magnitude_k_vectors(k):
        speed = reduced_group_velocity_signed(
            k_vector=k_vector,
            step_length=step_length,
            time_step=time_step,
            generator_fn=generator_laplace_beltrami,
            branch=1,
            alpha=alpha,
        )
        speeds.append(speed)

    reference = speeds[0]
    for speed in speeds[1:]:
        np.testing.assert_allclose(speed, reference, rtol=1e-14, atol=1e-14)


def test_reduced_group_velocity_vector_magnitude_is_isotropic() -> None:
    """
    The reduced group-velocity vector may point along k_vector, but its
    magnitude should be invariant under rotations of k_vector at fixed ||k||.
    """
    step_length = 1.0
    time_step = 1.0
    alpha = 0.25
    k = 0.5

    magnitudes = []
    for k_vector in same_magnitude_k_vectors(k):
        vg = reduced_group_velocity_vector(
            k_vector=k_vector,
            step_length=step_length,
            time_step=time_step,
            generator_fn=generator_laplace_beltrami,
            branch=1,
            alpha=alpha,
        )
        magnitudes.append(np.linalg.norm(vg))

    reference = magnitudes[0]
    for magnitude in magnitudes[1:]:
        np.testing.assert_allclose(magnitude, reference, rtol=1e-14, atol=1e-14)


def test_reduced_group_velocity_points_along_k_vector_for_positive_branch() -> None:
    """
    The positive-branch reduced group velocity should be parallel to k_vector.

    This is reduced packet-centre motion, not microscopic fixed-step transport.
    """
    step_length = 1.0
    time_step = 1.0
    alpha = 0.3
    k_vector = np.array([0.25, -0.5, 0.75], dtype=float)

    vg = reduced_group_velocity_vector(
        k_vector=k_vector,
        step_length=step_length,
        time_step=time_step,
        generator_fn=generator_laplace_beltrami,
        branch=1,
        alpha=alpha,
    )

    cross = np.cross(vg, k_vector)
    np.testing.assert_allclose(cross, np.zeros(3), rtol=1e-14, atol=1e-14)

    dot = float(np.dot(vg, k_vector))
    assert dot > 0.0


def test_negative_branch_group_velocity_points_opposite_k_vector() -> None:
    """
    The negative branch should have group velocity opposite to k_vector.

    This guards the branch sign convention in dE_-/dk.
    """
    step_length = 1.0
    time_step = 1.0
    alpha = 0.3
    k_vector = np.array([0.25, -0.5, 0.75], dtype=float)

    vg = reduced_group_velocity_vector(
        k_vector=k_vector,
        step_length=step_length,
        time_step=time_step,
        generator_fn=generator_laplace_beltrami,
        branch=-1,
        alpha=alpha,
    )

    cross = np.cross(vg, k_vector)
    np.testing.assert_allclose(cross, np.zeros(3), rtol=1e-14, atol=1e-14)

    dot = float(np.dot(vg, k_vector))
    assert dot < 0.0