from __future__ import annotations

import numpy as np
import pytest

from spn.kernels import (
    l2_eigenvalue,
    generator_laplace_beltrami,
    generator_linear,
    generator_poly2,
    generator_values,
    low_sector_mean,
    low_sector_gap,
    unitary_kernel_value,
    unitary_kernel_values,
)


# ---------------------------------------------------------------------
# L^2 spectrum
# ---------------------------------------------------------------------


@pytest.mark.parametrize(
    "l, expected",
    [
        (0, 0.0),
        (1, 2.0),
        (2, 6.0),
        (3, 12.0),
        (4, 20.0),
    ],
)
def test_l2_eigenvalue_returns_l_l_plus_one(l: int, expected: float) -> None:
    assert l2_eigenvalue(l) == expected


def test_l2_eigenvalue_rejects_negative_l() -> None:
    with pytest.raises(ValueError):
        l2_eigenvalue(-1)


# ---------------------------------------------------------------------
# Generator families
# ---------------------------------------------------------------------


def test_laplace_beltrami_generator_scales_l2_eigenvalue() -> None:
    alpha = 0.25

    assert generator_laplace_beltrami(0, alpha=alpha) == 0.0
    assert generator_laplace_beltrami(1, alpha=alpha) == alpha * 2.0
    assert generator_laplace_beltrami(2, alpha=alpha) == alpha * 6.0
    assert generator_laplace_beltrami(3, alpha=alpha) == alpha * 12.0


def test_laplace_beltrami_generator_rejects_negative_l() -> None:
    with pytest.raises(ValueError):
        generator_laplace_beltrami(-1, alpha=1.0)


def test_linear_generator_scales_l2_eigenvalue() -> None:
    kappa = 0.75

    assert generator_linear(0, kappa=kappa) == 0.0
    assert generator_linear(1, kappa=kappa) == 2.0 * kappa
    assert generator_linear(2, kappa=kappa) == 6.0 * kappa
    assert generator_linear(3, kappa=kappa) == 12.0 * kappa


def test_linear_generator_rejects_negative_l() -> None:
    with pytest.raises(ValueError):
        generator_linear(-1, kappa=1.0)


def test_poly2_generator_matches_quadratic_polynomial_in_l2() -> None:
    a = 0.5
    b = 0.1

    # l=2 -> L^2 eigenvalue = 6
    expected = a * 6.0 + b * 6.0**2
    assert generator_poly2(2, a=a, b=b) == pytest.approx(expected)


def test_poly2_generator_rejects_negative_l() -> None:
    with pytest.raises(ValueError):
        generator_poly2(-1, a=0.5, b=0.1)


# ---------------------------------------------------------------------
# Generator spectrum helper
# ---------------------------------------------------------------------


def test_generator_values_returns_expected_spectrum_for_laplace_beltrami() -> None:
    alpha = 0.5

    vals = generator_values(
        3,
        generator_laplace_beltrami,
        alpha=alpha,
    )

    expected = np.array(
        [
            0.0,
            1.0,
            3.0,
            6.0,
        ],
        dtype=float,
    )

    assert vals.shape == (4,)
    assert np.allclose(vals, expected)


def test_generator_values_rejects_negative_l_max() -> None:
    with pytest.raises(ValueError):
        generator_values(
            -1,
            generator_laplace_beltrami,
            alpha=0.5,
        )


# ---------------------------------------------------------------------
# Low-sector mean and gap
# ---------------------------------------------------------------------


def test_low_sector_mean_for_laplace_beltrami() -> None:
    alpha = 0.5

    assert low_sector_mean(
        generator_laplace_beltrami,
        alpha=alpha,
    ) == pytest.approx(alpha)


def test_low_sector_gap_for_laplace_beltrami_uses_half_gap_convention() -> None:
    alpha = 0.5

    assert low_sector_gap(
        generator_laplace_beltrami,
        alpha=alpha,
    ) == pytest.approx(alpha)


def test_low_sector_mean_for_linear_generator() -> None:
    kappa = 0.4

    assert low_sector_mean(
        generator_linear,
        kappa=kappa,
    ) == pytest.approx(kappa)


def test_low_sector_gap_for_linear_generator_uses_half_gap_convention() -> None:
    kappa = 0.4

    assert low_sector_gap(
        generator_linear,
        kappa=kappa,
    ) == pytest.approx(kappa)


# ---------------------------------------------------------------------
# One-step unitary kernel values
# ---------------------------------------------------------------------


def test_unitary_kernel_value_matches_expected_phase() -> None:
    f_l = 3.0
    tau = 0.25

    expected = np.exp(-1j * tau * f_l)
    assert unitary_kernel_value(f_l, tau) == pytest.approx(expected)


def test_unitary_kernel_values_returns_expected_phases() -> None:
    alpha = 0.5
    tau = 0.25
    l_max = 2

    U_vals = unitary_kernel_values(
        l_max,
        generator_laplace_beltrami,
        tau=tau,
        alpha=alpha,
    )

    expected = np.array(
        [
            np.exp(-1j * tau * generator_laplace_beltrami(l, alpha=alpha))
            for l in range(l_max + 1)
        ],
        dtype=complex,
    )

    assert U_vals.shape == (3,)
    assert np.allclose(U_vals, expected)


def test_unitary_kernel_values_have_unit_modulus() -> None:
    alpha = 0.5
    tau = 0.25
    l_max = 3

    U_vals = unitary_kernel_values(
        l_max,
        generator_laplace_beltrami,
        tau=tau,
        alpha=alpha,
    )

    assert np.allclose(np.abs(U_vals), np.ones(l_max + 1))


def test_unitary_kernel_values_rejects_negative_l_max() -> None:
    with pytest.raises(ValueError):
        unitary_kernel_values(
            -1,
            generator_laplace_beltrami,
            tau=1.0,
            alpha=0.5,
        )