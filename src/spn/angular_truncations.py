"""
Finite angular-sector truncations for SPN reduced-model diagnostics.

This module contains helper functions for axisymmetric m=0 angular
truncations of the directional Hilbert space:

    span{Y_00, Y_10, ..., Y_lmax,0}

The main purpose is to test how the Paper-2 reduced low sector
span{Y_00, Y_10} embeds inside a larger angular truncation.

Conventions
-----------
- Basis ordering is natural angular order:

      [l=0, l=1, ..., l=l_max]

- The angular generator is diagonal:

      H_omega |l,0> = f(l) |l,0>

- The axisymmetric transport coupling is represented by multiplication
  by cos(theta), which couples only neighbouring angular sectors:

      l <-> l +/- 1

  with matrix elements

      <l+1,0|cos(theta)|l,0>
      =
      (l+1) / sqrt((2l+1)(2l+3))

- The scalar k is interpreted after aligning the wave vector with the
  polar axis of the axisymmetric truncation.

- The Paper-2 reduced Hamiltonian convention orders the two-level
  state so that the sigma_z splitting is represented as

      diag[f(1), f(0)]

  whereas the natural angular block is ordered as

      diag[f(0), f(1)]

  Therefore `reordered_low_block_from_axisymmetric(...)` permutes the
  natural [l=0,l=1] block into the Paper-2 reduced convention [l=1,l=0].

Scope
-----
This is an enlarged axisymmetric angular Hamiltonian prototype. It is
not the full configuration-space SPN field evolution on psi(x, Omega).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np


Array = np.ndarray


__all__ = [
    "axisymmetric_l_values",
    "low_sector_indices",
    "cos_theta_matrix_element",
    "cos_theta_coupling_matrix",
    "angular_generator_spectrum",
    "angular_generator_matrix",
    "axisymmetric_transport_matrix",
    "axisymmetric_enlarged_hamiltonian",
    "natural_low_block_from_axisymmetric",
    "reduced_order_permutation_matrix",
    "reordered_low_block_from_axisymmetric",
    "low_sector_mean_and_gap_from_spectrum",
    "expected_reduced_low_block",
    "low_block_difference_norm",
    "is_hermitian",
]


def axisymmetric_l_values(l_max: int) -> Array:
    """
    Return l-values for the axisymmetric m=0 truncation.

    Parameters
    ----------
    l_max:
        Maximum angular momentum sector retained. Must be at least 1.

    Returns
    -------
    numpy.ndarray
        Integer array [0, 1, ..., l_max].
    """
    if not isinstance(l_max, int):
        raise TypeError("l_max must be an integer")
    if l_max < 1:
        raise ValueError("l_max must be at least 1")

    return np.arange(l_max + 1, dtype=int)


def low_sector_indices() -> Array:
    """
    Return indices of the Paper-2 low sector in natural axisymmetric order.

    In the natural enlarged basis [l=0, l=1, ..., l_max], the low sector is
    represented by indices [0, 1].
    """
    return np.array([0, 1], dtype=int)


def cos_theta_matrix_element(l: int) -> float:
    """
    Return <l+1,0|cos(theta)|l,0> for axisymmetric spherical harmonics.

    The matrix element is

        (l+1) / sqrt((2l+1)(2l+3))

    Parameters
    ----------
    l:
        Non-negative angular momentum index.

    Returns
    -------
    float
        Nearest-neighbour cos(theta) coupling matrix element.
    """
    if not isinstance(l, int):
        raise TypeError("l must be an integer")
    if l < 0:
        raise ValueError("l must be non-negative")

    numerator = float(l + 1)
    denominator = np.sqrt(float((2 * l + 1) * (2 * l + 3)))
    return numerator / denominator


def cos_theta_coupling_matrix(l_max: int) -> Array:
    """
    Matrix representation of multiplication by cos(theta) in the
    axisymmetric m=0 spherical-harmonic basis.

    Basis ordering
    --------------
    [Y_00, Y_10, Y_20, ..., Y_lmax,0]

    Nonzero entries
    ---------------
    Only neighbouring sectors are coupled:

        <l+1,0|cos(theta)|l,0>
        =
        (l+1) / sqrt((2l+1)(2l+3))

    Parameters
    ----------
    l_max:
        Maximum angular momentum sector retained. Must be at least 1.

    Returns
    -------
    numpy.ndarray
        Real symmetric matrix of shape (l_max+1, l_max+1).
    """
    l_values = axisymmetric_l_values(l_max)
    dim = len(l_values)

    matrix = np.zeros((dim, dim), dtype=float)

    for l in range(l_max):
        element = cos_theta_matrix_element(l)
        matrix[l, l + 1] = element
        matrix[l + 1, l] = element

    return matrix


def angular_generator_spectrum(
    l_max: int,
    generator_fn: Callable[..., float],
    **generator_params: Any,
) -> Array:
    """
    Return the diagonal spectrum f(l) for l=0,...,l_max.

    Parameters
    ----------
    l_max:
        Maximum angular momentum sector retained.
    generator_fn:
        Callable returning the generator value f(l).
    **generator_params:
        Extra keyword parameters passed to generator_fn.

    Returns
    -------
    numpy.ndarray
        Float array [f(0), f(1), ..., f(l_max)].
    """
    if not callable(generator_fn):
        raise TypeError("generator_fn must be callable")

    l_values = axisymmetric_l_values(l_max)

    spectrum = np.array(
        [generator_fn(int(l), **generator_params) for l in l_values],
        dtype=float,
    )

    if not np.all(np.isfinite(spectrum)):
        raise ValueError("angular generator spectrum contains non-finite values")

    return spectrum


def angular_generator_matrix(
    l_max: int,
    generator_fn: Callable[..., float],
    **generator_params: Any,
) -> tuple[Array, Array]:
    """
    Build the diagonal angular generator matrix H_omega.

    Parameters
    ----------
    l_max:
        Maximum angular momentum sector retained.
    generator_fn:
        Callable returning f(l).
    **generator_params:
        Extra keyword parameters passed to generator_fn.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray]
        (H_omega, spectrum), where H_omega is diagonal and spectrum is
        [f(0), f(1), ..., f(l_max)].
    """
    spectrum = angular_generator_spectrum(
        l_max=l_max,
        generator_fn=generator_fn,
        **generator_params,
    )

    return np.diag(spectrum).astype(complex), spectrum


def axisymmetric_transport_matrix(
    l_max: int,
    k: float,
    L: float,
    tau: float,
    transport_coupling_multiplier: float = 1.0,
) -> tuple[Array, Array]:
    """
    Build the axisymmetric transport-coupling matrix.

    The prototype enlarged Hamiltonian uses

        H_transport = multiplier * (L/tau) * k * cos(theta)

    The scalar k is interpreted after aligning the wave vector with the
    polar axis of the axisymmetric truncation.

    Parameters
    ----------
    l_max:
        Maximum angular momentum sector retained.
    k:
        Wave-number magnitude. Must be non-negative.
    L:
        Microscopic step length. Must be positive.
    tau:
        Time step. Must be positive.
    transport_coupling_multiplier:
        Non-negative dimensionless multiplier used for exploratory scans.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray]
        (H_transport, C), where C is the cos(theta) coupling matrix.
    """
    k = _as_finite_float(k, "k")
    L = _as_finite_float(L, "L")
    tau = _as_finite_float(tau, "tau")
    transport_coupling_multiplier = _as_finite_float(
        transport_coupling_multiplier,
        "transport_coupling_multiplier",
    )

    if k < 0:
        raise ValueError("k must be non-negative")
    if L <= 0:
        raise ValueError("L must be positive")
    if tau <= 0:
        raise ValueError("tau must be positive")
    if transport_coupling_multiplier < 0:
        raise ValueError("transport_coupling_multiplier must be non-negative")

    C = cos_theta_coupling_matrix(l_max)
    prefactor = transport_coupling_multiplier * (L / tau) * k
    H_transport = prefactor * C

    return H_transport.astype(complex), C


def axisymmetric_enlarged_hamiltonian(
    l_max: int,
    k: float,
    L: float,
    tau: float,
    generator_fn: Callable[..., float],
    generator_params: dict[str, Any] | None = None,
    transport_coupling_multiplier: float = 1.0,
) -> tuple[Array, Array, Array, Array, Array]:
    """
    Build an enlarged axisymmetric angular Hamiltonian.

    The Hamiltonian is

        H = H_omega + H_transport

    where

        H_omega |l,0> = f(l) |l,0>

    and

        H_transport = multiplier * (L/tau) * k * cos(theta).

    The scalar k is interpreted after aligning the wave vector with the
    polar axis of the axisymmetric truncation.

    Parameters
    ----------
    l_max:
        Maximum angular momentum sector retained.
    k:
        Wave-number magnitude.
    L:
        Microscopic step length.
    tau:
        Time step.
    generator_fn:
        Callable returning f(l).
    generator_params:
        Optional dictionary of keyword parameters passed to generator_fn.
    transport_coupling_multiplier:
        Non-negative dimensionless multiplier for exploratory coupling scans.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray]
        (H, H_omega, H_transport, spectrum, C)

        H:
            Enlarged Hermitian Hamiltonian.
        H_omega:
            Diagonal angular generator.
        H_transport:
            Axisymmetric transport-coupling matrix.
        spectrum:
            Diagonal generator spectrum [f(0), ..., f(l_max)].
        C:
            cos(theta) coupling matrix.
    """
    if generator_params is None:
        generator_params = {}
    else:
        generator_params = dict(generator_params)

    H_omega, spectrum = angular_generator_matrix(
        l_max=l_max,
        generator_fn=generator_fn,
        **generator_params,
    )

    H_transport, C = axisymmetric_transport_matrix(
        l_max=l_max,
        k=k,
        L=L,
        tau=tau,
        transport_coupling_multiplier=transport_coupling_multiplier,
    )

    H = H_omega + H_transport

    if not is_hermitian(H):
        raise ValueError("axisymmetric enlarged Hamiltonian is not Hermitian")

    return H, H_omega, H_transport, spectrum, C


def natural_low_block_from_axisymmetric(H: Array) -> Array:
    """
    Extract the natural [l=0,l=1] low block from an axisymmetric Hamiltonian.

    Parameters
    ----------
    H:
        Enlarged axisymmetric Hamiltonian in natural basis order
        [l=0, l=1, ..., l_max].

    Returns
    -------
    numpy.ndarray
        2x2 block in natural order [l=0,l=1].
    """
    H = np.asarray(H, dtype=complex)

    if H.ndim != 2 or H.shape[0] != H.shape[1]:
        raise ValueError("H must be a square matrix")
    if H.shape[0] < 2:
        raise ValueError("H must have dimension at least 2")

    return H[:2, :2].copy()


def reduced_order_permutation_matrix() -> Array:
    """
    Return the 2x2 permutation matrix mapping [l=0,l=1] to [l=1,l=0].
    """
    return np.array([[0, 1], [1, 0]], dtype=complex)


def reordered_low_block_from_axisymmetric(H: Array) -> Array:
    """
    Extract the low block in the Paper-2 reduced basis convention.

    The natural angular basis order is

        [l=0, l=1]

    but the Paper-2 sigma_z convention uses the order

        [l=1, l=0]

    so that the free part appears as

        diag[f(1), f(0)] = fbar I + Delta sigma_z.

    Parameters
    ----------
    H:
        Enlarged axisymmetric Hamiltonian.

    Returns
    -------
    numpy.ndarray
        2x2 low block in reduced Paper-2 order [l=1,l=0].
    """
    natural_block = natural_low_block_from_axisymmetric(H)
    P = reduced_order_permutation_matrix()
    return P @ natural_block @ P.conj().T


def low_sector_mean_and_gap_from_spectrum(spectrum: Array) -> tuple[float, float]:
    """
    Return (fbar, delta) from the first two entries of an angular spectrum.

    delta is the Paper-2 half-gap:

        delta = (f(1) - f(0)) / 2

    not the full gap.

    Parameters
    ----------
    spectrum:
        Array whose first two entries are f(0), f(1).

    Returns
    -------
    tuple[float, float]
        (fbar, delta).
    """
    spectrum = np.asarray(spectrum, dtype=float)

    if spectrum.ndim != 1 or spectrum.shape[0] < 2:
        raise ValueError("spectrum must be a one-dimensional array with at least two entries")

    f0 = float(spectrum[0])
    f1 = float(spectrum[1])

    fbar = 0.5 * (f0 + f1)
    delta = 0.5 * (f1 - f0)

    return fbar, delta


def expected_reduced_low_block(
    fbar: float,
    delta: float,
    k: float,
    L: float,
    tau: float,
) -> Array:
    """
    Construct the expected Paper-2 reduced low-sector Hamiltonian.

    Convention:

        H_red = fbar I + delta sigma_z + v k sigma_x

    where

        v = L / (tau sqrt(3)).

    Parameters
    ----------
    fbar:
        Low-sector mean.
    delta:
        Paper-2 half-gap delta = (f(1)-f(0))/2.
    k:
        Wave-number magnitude.
    L:
        Step length. Must be positive.
    tau:
        Time step. Must be positive.

    Returns
    -------
    numpy.ndarray
        2x2 reduced Hamiltonian in Paper-2 basis convention [l=1,l=0].
    """
    fbar = _as_finite_float(fbar, "fbar")
    delta = _as_finite_float(delta, "delta")
    k = _as_finite_float(k, "k")
    L = _as_finite_float(L, "L")
    tau = _as_finite_float(tau, "tau")

    if k < 0:
        raise ValueError("k must be non-negative")
    if L <= 0:
        raise ValueError("L must be positive")
    if tau <= 0:
        raise ValueError("tau must be positive")

    I = np.eye(2, dtype=complex)
    sigma_x = np.array([[0, 1], [1, 0]], dtype=complex)
    sigma_z = np.array([[1, 0], [0, -1]], dtype=complex)

    reduced_coupling_speed = L / (tau * np.sqrt(3.0))

    H = fbar * I + delta * sigma_z + reduced_coupling_speed * k * sigma_x

    if not is_hermitian(H):
        raise ValueError("expected reduced low block is not Hermitian")

    return H


def low_block_difference_norm(
    H: Array,
    spectrum: Array,
    k: float,
    L: float,
    tau: float,
) -> float:
    """
    Compare the reordered enlarged low block with the expected H_red.

    Parameters
    ----------
    H:
        Enlarged axisymmetric Hamiltonian.
    spectrum:
        Angular generator spectrum [f(0), f(1), ...].
    k:
        Wave-number magnitude.
    L:
        Step length.
    tau:
        Time step.

    Returns
    -------
    float
        Frobenius norm of the difference.
    """
    fbar, delta = low_sector_mean_and_gap_from_spectrum(spectrum)

    low_block = reordered_low_block_from_axisymmetric(H)
    expected = expected_reduced_low_block(
        fbar=fbar,
        delta=delta,
        k=k,
        L=L,
        tau=tau,
    )

    return float(np.linalg.norm(low_block - expected))


def is_hermitian(matrix: Array, atol: float = 1e-12) -> bool:
    """
    Return True if matrix is Hermitian to the requested absolute tolerance.
    """
    matrix = np.asarray(matrix, dtype=complex)

    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        return False

    return bool(np.allclose(matrix, matrix.conj().T, atol=atol, rtol=0.0))


def _as_finite_float(value: float, name: str) -> float:
    """
    Convert value to float and validate finiteness.
    """
    try:
        value_float = float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be convertible to float") from exc

    if not np.isfinite(value_float):
        raise ValueError(f"{name} must be finite")

    return value_float