from __future__ import annotations

from typing import Callable
import numpy as np


def l2_eigenvalue(l: int) -> float:
    """
    Return the dimensionless L^2 eigenvalue for angular momentum quantum number l:
        L^2 -> l(l+1)

    Note:
        This uses units where any overall hbar^2 factor has been absorbed.
    """
    if l < 0:
        raise ValueError("l must be non-negative")
    return float(l * (l + 1))


# ---------------------------------------------------------------------
# Free generator spectra H_Ω = f(L^2)
# ---------------------------------------------------------------------

def generator_laplace_beltrami(l: int, alpha: float) -> float:
    """
    Laplace-Beltrami spectral generator:
        H_Ω = alpha * (-Δ_{S^2})

    Since -Δ_{S^2} has eigenvalues l(l+1), this gives:
        f(l) = alpha * l(l+1)
    """
    return alpha * l2_eigenvalue(l)


def generator_linear(l: int, kappa: float) -> float:
    """
    Generic linear spectral generator:
        f(L^2) = kappa * L^2
    """
    return kappa * l2_eigenvalue(l)


def generator_poly2(l: int, a: float, b: float) -> float:
    """
    Quadratic polynomial spectral generator:
        f(L^2) = a * L^2 + b * (L^2)^2
    """
    lam = l2_eigenvalue(l)
    return a * lam + b * lam**2


def generator_values(
    l_max: int,
    generator_fn: Callable[..., float],
    **params,
) -> np.ndarray:
    """
    Evaluate the generator spectrum f(l) for l = 0, 1, ..., l_max.
    """
    if l_max < 0:
        raise ValueError("l_max must be non-negative")
    return np.array(
        [generator_fn(l, **params) for l in range(l_max + 1)],
        dtype=float,
    )


# ---------------------------------------------------------------------
# One-step unitary kernel K = exp(-i τ H_Ω)
# ---------------------------------------------------------------------

def unitary_kernel_value(generator_eigenvalue: float, tau: float) -> complex:
    """
    Convert a generator eigenvalue f(l) into the corresponding one-step
    unitary kernel eigenvalue:
        K_l = exp(-i τ f(l))
    """
    return np.exp(-1j * tau * generator_eigenvalue)


def unitary_kernel_values(
    l_max: int,
    generator_fn: Callable[..., float],
    tau: float,
    **params,
) -> np.ndarray:
    """
    Evaluate the one-step unitary kernel eigenvalues
        K_l = exp(-i τ f(l))
    for l = 0, 1, ..., l_max.
    """
    if l_max < 0:
        raise ValueError("l_max must be non-negative")
    return np.array(
        [
            unitary_kernel_value(generator_fn(l, **params), tau)
            for l in range(l_max + 1)
        ],
        dtype=complex,
    )


# ---------------------------------------------------------------------
# Low-sector quantities for the 3+1 l=0 / l=1 reduction
# ---------------------------------------------------------------------

def low_sector_gap(generator_fn: Callable[..., float], **params) -> float:
    """
    Compute the Paper-2 low-sector gap:
        Δ = (f(1) - f(0)) / 2
    """
    f0 = generator_fn(0, **params)
    f1 = generator_fn(1, **params)
    return 0.5 * (f1 - f0)


def low_sector_mean(generator_fn: Callable[..., float], **params) -> float:
    """
    Compute the low-sector mean:
        f̄ = (f(1) + f(0)) / 2
    """
    f0 = generator_fn(0, **params)
    f1 = generator_fn(1, **params)
    return 0.5 * (f1 + f0)