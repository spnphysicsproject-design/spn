from __future__ import annotations

import numpy as np

from spn.evolution import (
    reduced_coupling_speed,
    reduced_energy_eigenvalues,
    reduced_group_velocity_signed,
)
from spn.kernels import (
    generator_laplace_beltrami,
    low_sector_gap,
    low_sector_mean,
)


def k_vector_x(k: float) -> np.ndarray:
    """
    Return a simple 3D wave-vector with magnitude k.

    The low-k dispersion test is about dependence on ||k||, not direction.
    Directional invariance is covered separately by the isotropy tests.
    """
    return np.array([k, 0.0, 0.0], dtype=float)


def positive_branch_energy(
    *,
    k: float,
    step_length: float,
    time_step: float,
    alpha: float,
) -> float:
    """
    Return E_+(k) from the reduced implementation.
    """
    _, E_plus = reduced_energy_eigenvalues(
        k_vector=k_vector_x(k),
        step_length=step_length,
        time_step=time_step,
        generator_fn=generator_laplace_beltrami,
        alpha=alpha,
    )
    return E_plus


def test_low_k_positive_branch_matches_quadratic_massive_expansion() -> None:
    """
    For small k and nonzero Delta:

        E_+(k) = fbar + sqrt(Delta^2 + v^2 k^2)

    has the low-k expansion:

        E_+(k) = fbar + |Delta| + (v^2 / (2 |Delta|)) k^2 + O(k^4)

    This checks the massive/Dirac-like small-k behaviour of the reduced model.
    """
    step_length = 1.0
    time_step = 1.0
    alpha = 0.4

    fbar = low_sector_mean(generator_laplace_beltrami, alpha=alpha)
    delta = low_sector_gap(generator_laplace_beltrami, alpha=alpha)
    v = reduced_coupling_speed(
        step_length=step_length,
        time_step=time_step,
    )

    assert delta > 0.0

    ks = np.array([1.0e-4, 2.0e-4, 3.0e-4, 4.0e-4, 5.0e-4], dtype=float)

    actual = np.array(
        [
            positive_branch_energy(
                k=float(k),
                step_length=step_length,
                time_step=time_step,
                alpha=alpha,
            )
            for k in ks
        ],
        dtype=float,
    )

    expected = fbar + abs(delta) + (v**2 / (2.0 * abs(delta))) * ks**2

    np.testing.assert_allclose(actual, expected, rtol=1e-7, atol=1e-14)


def test_low_k_negative_branch_matches_quadratic_massive_expansion() -> None:
    """
    For small k and nonzero Delta:

        E_-(k) = fbar - sqrt(Delta^2 + v^2 k^2)

    has the low-k expansion:

        E_-(k) = fbar - |Delta| - (v^2 / (2 |Delta|)) k^2 + O(k^4)

    This guards the sign convention for the negative branch.
    """
    step_length = 1.0
    time_step = 1.0
    alpha = 0.4

    fbar = low_sector_mean(generator_laplace_beltrami, alpha=alpha)
    delta = low_sector_gap(generator_laplace_beltrami, alpha=alpha)
    v = reduced_coupling_speed(
        step_length=step_length,
        time_step=time_step,
    )

    assert delta > 0.0

    ks = np.array([1.0e-4, 2.0e-4, 3.0e-4, 4.0e-4, 5.0e-4], dtype=float)

    actual = []
    for k in ks:
        E_minus, _ = reduced_energy_eigenvalues(
            k_vector=k_vector_x(float(k)),
            step_length=step_length,
            time_step=time_step,
            generator_fn=generator_laplace_beltrami,
            alpha=alpha,
        )
        actual.append(E_minus)

    actual = np.array(actual, dtype=float)
    expected = fbar - abs(delta) - (v**2 / (2.0 * abs(delta))) * ks**2

    np.testing.assert_allclose(actual, expected, rtol=1e-7, atol=1e-14)


def test_low_k_positive_branch_quadratic_coefficient_from_fit() -> None:
    """
    Fit E_+(k) - E_+(0) against k^2 at small k.

    The fitted slope should match:

        v^2 / (2 |Delta|)

    This is a numerical pressure test of the massive low-k dispersion.
    """
    step_length = 1.0
    time_step = 1.0
    alpha = 0.4

    delta = low_sector_gap(generator_laplace_beltrami, alpha=alpha)
    v = reduced_coupling_speed(
        step_length=step_length,
        time_step=time_step,
    )

    expected_slope = v**2 / (2.0 * abs(delta))

    ks = np.linspace(1.0e-5, 5.0e-4, 12)
    k_squared = ks**2

    E0 = positive_branch_energy(
        k=0.0,
        step_length=step_length,
        time_step=time_step,
        alpha=alpha,
    )

    energy_shift = np.array(
        [
            positive_branch_energy(
                k=float(k),
                step_length=step_length,
                time_step=time_step,
                alpha=alpha,
            )
            - E0
            for k in ks
        ],
        dtype=float,
    )

    fitted_slope, fitted_intercept = np.polyfit(k_squared, energy_shift, deg=1)

    np.testing.assert_allclose(fitted_slope, expected_slope, rtol=1e-6, atol=1e-12)
    np.testing.assert_allclose(fitted_intercept, 0.0, rtol=0.0, atol=1e-10)


def test_low_k_group_velocity_is_linear_in_k() -> None:
    """
    At small k, the positive-branch group velocity satisfies:

        dE_+/dk = v^2 k / sqrt(Delta^2 + v^2 k^2)

    and therefore:

        dE_+/dk ≈ (v^2 / |Delta|) k

    This checks the low-k packet-centre velocity scaling.
    """
    step_length = 1.0
    time_step = 1.0
    alpha = 0.4

    delta = low_sector_gap(generator_laplace_beltrami, alpha=alpha)
    v = reduced_coupling_speed(
        step_length=step_length,
        time_step=time_step,
    )

    expected_slope = v**2 / abs(delta)

    ks = np.array([1.0e-6, 2.0e-6, 3.0e-6, 4.0e-6, 5.0e-6], dtype=float)

    speeds = np.array(
        [
            reduced_group_velocity_signed(
                k_vector=k_vector_x(float(k)),
                step_length=step_length,
                time_step=time_step,
                generator_fn=generator_laplace_beltrami,
                branch=1,
                alpha=alpha,
            )
            for k in ks
        ],
        dtype=float,
    )

    expected = expected_slope * ks

    np.testing.assert_allclose(speeds, expected, rtol=1e-8, atol=1e-14)


def test_energy_gap_at_k_zero_is_two_delta() -> None:
    """
    At k = 0:

        E_+(0) - E_-(0) = 2 |Delta|

    This guards the half-gap convention:

        Delta = (f(1) - f(0)) / 2

    not the full gap.
    """
    step_length = 1.0
    time_step = 1.0
    alpha = 0.4

    delta = low_sector_gap(generator_laplace_beltrami, alpha=alpha)

    E_minus, E_plus = reduced_energy_eigenvalues(
        k_vector=k_vector_x(0.0),
        step_length=step_length,
        time_step=time_step,
        generator_fn=generator_laplace_beltrami,
        alpha=alpha,
    )

    actual_gap = E_plus - E_minus
    expected_gap = 2.0 * abs(delta)

    np.testing.assert_allclose(actual_gap, expected_gap, rtol=1e-14, atol=1e-14)


def test_reduced_dispersion_matches_exact_massive_formula() -> None:
    """
    Directly check the exact reduced dispersion formula over a small-k range:

        E_±(k) = fbar ± sqrt(Delta^2 + v^2 k^2)

    The low-k expansion tests are approximate; this test confirms the exact
    implementation convention used by the reduced model.
    """
    step_length = 1.0
    time_step = 1.0
    alpha = 0.4

    fbar = low_sector_mean(generator_laplace_beltrami, alpha=alpha)
    delta = low_sector_gap(generator_laplace_beltrami, alpha=alpha)
    v = reduced_coupling_speed(
        step_length=step_length,
        time_step=time_step,
    )

    ks = np.linspace(0.0, 0.01, 10)

    for k in ks:
        E_minus, E_plus = reduced_energy_eigenvalues(
            k_vector=k_vector_x(float(k)),
            step_length=step_length,
            time_step=time_step,
            generator_fn=generator_laplace_beltrami,
            alpha=alpha,
        )

        spread = np.sqrt(delta**2 + (v * k) ** 2)

        np.testing.assert_allclose(E_minus, fbar - spread, rtol=1e-14, atol=1e-14)
        np.testing.assert_allclose(E_plus, fbar + spread, rtol=1e-14, atol=1e-14)