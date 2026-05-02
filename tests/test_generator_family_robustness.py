from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest

from spn.evolution import (
    reduced_axisymmetric_hamiltonian,
    reduced_axisymmetric_unitary,
    reduced_coupling_speed,
    reduced_energy_eigenvalues,
    reduced_group_velocity_signed,
    reduced_group_velocity_vector,
)
from spn.kernels import (
    generator_laplace_beltrami,
    generator_linear,
    generator_poly2,
    low_sector_gap,
    low_sector_mean,
)


GeneratorFn = Callable[..., float]


GENERATOR_CASES: list[tuple[GeneratorFn, dict[str, float]]] = [
    (
        generator_laplace_beltrami,
        {"alpha": 0.4},
    ),
    (
        generator_linear,
        {"kappa": 0.4},
    ),
    (
        generator_poly2,
        {"a": 0.25, "b": 0.05},
    ),
    (
        generator_poly2,
        {"a": 0.4, "b": -0.02},
    ),
]


GENERATOR_CASE_IDS = [
    "laplace_beltrami",
    "linear_l2",
    "poly2_positive",
    "poly2_mixed",
]


def k_vector_x(k: float) -> np.ndarray:
    """
    Return a simple 3D wave-vector with magnitude k.
    """
    return np.array([k, 0.0, 0.0], dtype=float)


def same_magnitude_k_vectors(k: float) -> list[np.ndarray]:
    """
    Return several 3D k-vectors with the same Euclidean magnitude.
    """
    return [
        np.array([k, 0.0, 0.0], dtype=float),
        np.array([0.0, k, 0.0], dtype=float),
        np.array([0.0, 0.0, k], dtype=float),
        np.array([k / np.sqrt(2.0), k / np.sqrt(2.0), 0.0], dtype=float),
        np.array([k / np.sqrt(3.0), k / np.sqrt(3.0), k / np.sqrt(3.0)], dtype=float),
    ]


@pytest.mark.parametrize(
    ("generator_fn", "generator_params"),
    GENERATOR_CASES,
    ids=GENERATOR_CASE_IDS,
)
def test_generator_family_low_sector_half_gap_is_finite_and_nonzero(
    generator_fn: GeneratorFn,
    generator_params: dict[str, float],
) -> None:
    """
    Each robustness-test generator should produce a finite nonzero low-sector
    half-gap Delta.

    This keeps the massive low-k expansion meaningful for these cases and
    protects the convention that low_sector_gap returns:

        Delta = (f(1) - f(0)) / 2

    not the full gap.
    """
    fbar = low_sector_mean(generator_fn, **generator_params)
    delta = low_sector_gap(generator_fn, **generator_params)

    assert np.isfinite(fbar)
    assert np.isfinite(delta)
    assert delta != 0.0


@pytest.mark.parametrize(
    ("generator_fn", "generator_params"),
    GENERATOR_CASES,
    ids=GENERATOR_CASE_IDS,
)
def test_generator_family_hamiltonian_is_hermitian(
    generator_fn: GeneratorFn,
    generator_params: dict[str, float],
) -> None:
    """
    Across generator families, the reduced axisymmetric Hamiltonian should
    remain Hermitian.
    """
    H = reduced_axisymmetric_hamiltonian(
        k_vector=np.array([0.2, -0.3, 0.4], dtype=float),
        step_length=1.0,
        time_step=1.0,
        generator_fn=generator_fn,
        **generator_params,
    )

    np.testing.assert_allclose(H, H.conj().T, rtol=1e-14, atol=1e-14)


@pytest.mark.parametrize(
    ("generator_fn", "generator_params"),
    GENERATOR_CASES,
    ids=GENERATOR_CASE_IDS,
)
def test_generator_family_unitary_is_unitary(
    generator_fn: GeneratorFn,
    generator_params: dict[str, float],
) -> None:
    """
    Across generator families, U = exp(-i tau H_red) should remain unitary.
    """
    U = reduced_axisymmetric_unitary(
        k_vector=np.array([0.2, -0.3, 0.4], dtype=float),
        step_length=1.0,
        time_step=0.7,
        generator_fn=generator_fn,
        **generator_params,
    )

    identity = np.eye(2, dtype=complex)

    np.testing.assert_allclose(U.conj().T @ U, identity, rtol=1e-14, atol=1e-14)
    np.testing.assert_allclose(U @ U.conj().T, identity, rtol=1e-14, atol=1e-14)


@pytest.mark.parametrize(
    ("generator_fn", "generator_params"),
    GENERATOR_CASES,
    ids=GENERATOR_CASE_IDS,
)
def test_generator_family_dispersion_matches_exact_formula(
    generator_fn: GeneratorFn,
    generator_params: dict[str, float],
) -> None:
    """
    Across generator families, the reduced energy eigenvalues should match:

        E_±(k) = fbar ± sqrt(Delta^2 + v^2 k^2)
    """
    step_length = 1.0
    time_step = 1.0

    fbar = low_sector_mean(generator_fn, **generator_params)
    delta = low_sector_gap(generator_fn, **generator_params)
    v = reduced_coupling_speed(
        step_length=step_length,
        time_step=time_step,
    )

    ks = np.linspace(0.0, 1.0, 8)

    for k in ks:
        E_minus, E_plus = reduced_energy_eigenvalues(
            k_vector=k_vector_x(float(k)),
            step_length=step_length,
            time_step=time_step,
            generator_fn=generator_fn,
            **generator_params,
        )

        spread = np.sqrt(delta**2 + (v * k) ** 2)

        np.testing.assert_allclose(E_minus, fbar - spread, rtol=1e-14, atol=1e-14)
        np.testing.assert_allclose(E_plus, fbar + spread, rtol=1e-14, atol=1e-14)


@pytest.mark.parametrize(
    ("generator_fn", "generator_params"),
    GENERATOR_CASES,
    ids=GENERATOR_CASE_IDS,
)
def test_generator_family_energy_is_isotropic_at_fixed_k_magnitude(
    generator_fn: GeneratorFn,
    generator_params: dict[str, float],
) -> None:
    """
    Across generator families, energy eigenvalues should depend on ||k_vector||,
    not the direction of k_vector.
    """
    step_length = 1.0
    time_step = 1.0
    k = 0.55

    spectra = []
    for k_vector in same_magnitude_k_vectors(k):
        eigvals = reduced_energy_eigenvalues(
            k_vector=k_vector,
            step_length=step_length,
            time_step=time_step,
            generator_fn=generator_fn,
            **generator_params,
        )
        spectra.append(np.array(eigvals, dtype=float))

    reference = spectra[0]
    for eigvals in spectra[1:]:
        np.testing.assert_allclose(eigvals, reference, rtol=1e-14, atol=1e-14)


@pytest.mark.parametrize(
    ("generator_fn", "generator_params"),
    GENERATOR_CASES,
    ids=GENERATOR_CASE_IDS,
)
def test_generator_family_group_speed_is_bounded_by_reduced_coupling_speed(
    generator_fn: GeneratorFn,
    generator_params: dict[str, float],
) -> None:
    """
    Across generator families, the group-speed magnitude should remain bounded
    by the reduced coupling speed:

        v = L / (tau sqrt(3))

    This is the reduced packet-centre speed limit, not the microscopic
    transport speed L / tau.
    """
    step_length = 1.0
    time_step = 1.0

    v = reduced_coupling_speed(
        step_length=step_length,
        time_step=time_step,
    )

    ks = np.linspace(0.0, 50.0, 32)

    for k in ks:
        speed = reduced_group_velocity_signed(
            k_vector=k_vector_x(float(k)),
            step_length=step_length,
            time_step=time_step,
            generator_fn=generator_fn,
            branch=1,
            **generator_params,
        )

        assert abs(speed) <= v + 1e-14


@pytest.mark.parametrize(
    ("generator_fn", "generator_params"),
    GENERATOR_CASES,
    ids=GENERATOR_CASE_IDS,
)
def test_generator_family_group_velocity_vector_direction_conventions(
    generator_fn: GeneratorFn,
    generator_params: dict[str, float],
) -> None:
    """
    Across generator families, the positive-branch group velocity should point
    along k_vector and the negative branch should point opposite k_vector.
    """
    step_length = 1.0
    time_step = 1.0
    k_vector = np.array([0.25, -0.5, 0.75], dtype=float)

    vg_plus = reduced_group_velocity_vector(
        k_vector=k_vector,
        step_length=step_length,
        time_step=time_step,
        generator_fn=generator_fn,
        branch=1,
        **generator_params,
    )
    vg_minus = reduced_group_velocity_vector(
        k_vector=k_vector,
        step_length=step_length,
        time_step=time_step,
        generator_fn=generator_fn,
        branch=-1,
        **generator_params,
    )

    np.testing.assert_allclose(
        np.cross(vg_plus, k_vector),
        np.zeros(3),
        rtol=1e-14,
        atol=1e-14,
    )
    np.testing.assert_allclose(
        np.cross(vg_minus, k_vector),
        np.zeros(3),
        rtol=1e-14,
        atol=1e-14,
    )

    assert float(np.dot(vg_plus, k_vector)) > 0.0
    assert float(np.dot(vg_minus, k_vector)) < 0.0


@pytest.mark.parametrize(
    ("generator_fn", "generator_params"),
    GENERATOR_CASES,
    ids=GENERATOR_CASE_IDS,
)
def test_generator_family_low_k_positive_branch_quadratic_coefficient(
    generator_fn: GeneratorFn,
    generator_params: dict[str, float],
) -> None:
    """
    Across generator families with nonzero Delta, the positive branch should
    have the small-k expansion:

        E_+(k) - E_+(0) ≈ (v^2 / (2 |Delta|)) k^2
    """
    step_length = 1.0
    time_step = 1.0

    delta = low_sector_gap(generator_fn, **generator_params)
    v = reduced_coupling_speed(
        step_length=step_length,
        time_step=time_step,
    )

    expected_slope = v**2 / (2.0 * abs(delta))

    ks = np.linspace(1.0e-5, 5.0e-4, 12)
    k_squared = ks**2

    _, E0_plus = reduced_energy_eigenvalues(
        k_vector=k_vector_x(0.0),
        step_length=step_length,
        time_step=time_step,
        generator_fn=generator_fn,
        **generator_params,
    )

    shifts = []
    for k in ks:
        _, E_plus = reduced_energy_eigenvalues(
            k_vector=k_vector_x(float(k)),
            step_length=step_length,
            time_step=time_step,
            generator_fn=generator_fn,
            **generator_params,
        )
        shifts.append(E_plus - E0_plus)

    energy_shift = np.array(shifts, dtype=float)

    fitted_slope, fitted_intercept = np.polyfit(k_squared, energy_shift, deg=1)

    np.testing.assert_allclose(fitted_slope, expected_slope, rtol=1e-6, atol=1e-12)
    np.testing.assert_allclose(fitted_intercept, 0.0, rtol=0.0, atol=1e-10)