from __future__ import annotations

"""
General numerical and validation utilities for SPN models.

This module is intentionally modest. It provides small reusable helpers for
finite arrays, vector/matrix validation, normalisation, and basic numerical
checks.

It should not contain model-specific physics. Keep SPN-specific evolution,
kernel, observable, and topology logic in their own modules.
"""

from typing import Any

import numpy as np


# ---------------------------------------------------------------------
# Default tolerances
# ---------------------------------------------------------------------


DEFAULT_ATOL = 1e-12


# ---------------------------------------------------------------------
# Basic finite-value helpers
# ---------------------------------------------------------------------


def is_finite_array(x: np.ndarray) -> bool:
    """
    Return True if an array contains only finite values.

    Works for both real and complex arrays.
    """
    x = np.asarray(x)

    if np.iscomplexobj(x):
        return bool(np.isfinite(x.real).all() and np.isfinite(x.imag).all())

    return bool(np.isfinite(x).all())


def require_finite_array(x: np.ndarray, *, name: str = "array") -> np.ndarray:
    """
    Convert x to an array and require all entries to be finite.

    Returns
    -------
    np.ndarray
        The converted array.
    """
    x = np.asarray(x)

    if not is_finite_array(x):
        raise ValueError(f"{name} must contain only finite values")

    return x


def require_scalar_finite(x: float, *, name: str = "value") -> float:
    """
    Require a scalar value to be finite and return it as a float.
    """
    x = float(x)

    if not np.isfinite(x):
        raise ValueError(f"{name} must be finite")

    return x


def require_nonnegative(x: float, *, name: str = "value") -> float:
    """
    Require a scalar value to be finite and non-negative.
    """
    x = require_scalar_finite(x, name=name)

    if x < 0:
        raise ValueError(f"{name} must be non-negative")

    return x


def require_positive(x: float, *, name: str = "value") -> float:
    """
    Require a scalar value to be finite and strictly positive.
    """
    x = require_scalar_finite(x, name=name)

    if x <= 0:
        raise ValueError(f"{name} must be positive")

    return x


# ---------------------------------------------------------------------
# Vector and matrix validation
# ---------------------------------------------------------------------


def as_finite_vector(
    x: np.ndarray,
    *,
    name: str = "vector",
    dtype: Any = complex,
    nonempty: bool = True,
) -> np.ndarray:
    """
    Convert x to a finite 1D vector.

    Parameters
    ----------
    x:
        Input array-like object.

    name:
        Name used in validation errors.

    dtype:
        dtype passed to np.asarray.

    nonempty:
        If True, require the vector to have at least one element.
    """
    x = np.asarray(x, dtype=dtype)

    if x.ndim != 1:
        raise ValueError(f"{name} must be a 1D vector")

    if nonempty and x.size == 0:
        raise ValueError(f"{name} must be non-empty")

    if not is_finite_array(x):
        raise ValueError(f"{name} must contain only finite values")

    return x


def as_finite_vector3(
    x: np.ndarray,
    *,
    name: str = "vector",
    dtype: Any = float,
) -> np.ndarray:
    """
    Convert x to a finite 3-vector with shape (3,).
    """
    x = as_finite_vector(x, name=name, dtype=dtype, nonempty=True)

    if x.shape != (3,):
        raise ValueError(f"{name} must be a 3-vector with shape (3,)")

    return x


def as_finite_matrix(
    A: np.ndarray,
    *,
    name: str = "matrix",
    dtype: Any = complex,
) -> np.ndarray:
    """
    Convert A to a finite 2D matrix.
    """
    A = np.asarray(A, dtype=dtype)

    if A.ndim != 2:
        raise ValueError(f"{name} must be a 2D matrix")

    if not is_finite_array(A):
        raise ValueError(f"{name} must contain only finite values")

    return A


def as_square_matrix(
    A: np.ndarray,
    *,
    name: str = "matrix",
    dtype: Any = complex,
) -> np.ndarray:
    """
    Convert A to a finite square matrix.
    """
    A = as_finite_matrix(A, name=name, dtype=dtype)

    if A.shape[0] != A.shape[1]:
        raise ValueError(f"{name} must be a square matrix")

    return A


# ---------------------------------------------------------------------
# Norm and normalisation helpers
# ---------------------------------------------------------------------


def vector_norm(x: np.ndarray, *, name: str = "vector") -> float:
    """
    Return the Euclidean norm of a finite vector.
    """
    x = as_finite_vector(x, name=name, dtype=complex)
    return float(np.linalg.norm(x))


def normalize_vector(x: np.ndarray, *, name: str = "vector") -> np.ndarray:
    """
    Return a normalized copy of a finite vector.
    """
    x = as_finite_vector(x, name=name, dtype=complex)
    norm = np.linalg.norm(x)

    if norm == 0.0:
        raise ValueError(f"cannot normalize zero {name}")

    return x / norm


def safe_unit_vector(
    x: np.ndarray,
    *,
    name: str = "vector",
    zero: str = "raise",
) -> np.ndarray:
    """
    Return a unit vector in the direction of x.

    Parameters
    ----------
    x:
        Input vector.

    name:
        Name used in validation errors.

    zero:
        Behaviour when ||x|| = 0.

        - "raise": raise ValueError
        - "zero": return a zero vector of the same shape
    """
    x = as_finite_vector(x, name=name, dtype=float)
    norm = np.linalg.norm(x)

    if norm == 0.0:
        if zero == "raise":
            raise ValueError(f"cannot form unit vector from zero {name}")
        if zero == "zero":
            return np.zeros_like(x, dtype=float)
        raise ValueError("zero must be either 'raise' or 'zero'")

    return x / norm


# ---------------------------------------------------------------------
# Matrix property checks
# ---------------------------------------------------------------------


def is_hermitian(A: np.ndarray, *, atol: float = DEFAULT_ATOL) -> bool:
    """
    Return True if A is Hermitian to numerical tolerance.
    """
    A = np.asarray(A, dtype=complex)

    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        return False

    return bool(np.allclose(A, A.conj().T, atol=atol, rtol=0.0))


def require_hermitian(
    A: np.ndarray,
    *,
    name: str = "matrix",
    atol: float = DEFAULT_ATOL,
) -> np.ndarray:
    """
    Convert A to a square matrix and require it to be Hermitian.
    """
    A = as_square_matrix(A, name=name, dtype=complex)

    if not is_hermitian(A, atol=atol):
        raise ValueError(f"{name} must be Hermitian")

    return A


def is_unitary(U: np.ndarray, *, atol: float = DEFAULT_ATOL) -> bool:
    """
    Return True if U is unitary to numerical tolerance.
    """
    U = np.asarray(U, dtype=complex)

    if U.ndim != 2 or U.shape[0] != U.shape[1]:
        return False

    identity = np.eye(U.shape[0], dtype=complex)
    return bool(np.allclose(U.conj().T @ U, identity, atol=atol, rtol=0.0))


def require_unitary(
    U: np.ndarray,
    *,
    name: str = "matrix",
    atol: float = DEFAULT_ATOL,
) -> np.ndarray:
    """
    Convert U to a square matrix and require it to be unitary.
    """
    U = as_square_matrix(U, name=name, dtype=complex)

    if not is_unitary(U, atol=atol):
        raise ValueError(f"{name} must be unitary")

    return U


# ---------------------------------------------------------------------
# Probability and diagnostic helpers
# ---------------------------------------------------------------------


def probability_weights(psi: np.ndarray, *, normalize: bool = True) -> np.ndarray:
    """
    Return component probability weights |psi_i|^2.

    Parameters
    ----------
    psi:
        1D complex state vector.

    normalize:
        If True, normalize psi before computing weights.
    """
    psi = as_finite_vector(psi, name="psi", dtype=complex)

    if normalize:
        psi = normalize_vector(psi, name="psi")

    return np.abs(psi) ** 2


def sums_to_one(
    weights: np.ndarray,
    *,
    atol: float = DEFAULT_ATOL,
) -> bool:
    """
    Return True if a vector of weights sums to one within tolerance.
    """
    weights = as_finite_vector(weights, name="weights", dtype=float)

    return bool(np.isclose(np.sum(weights), 1.0, atol=atol, rtol=0.0))


def max_abs_difference(a: np.ndarray, b: np.ndarray) -> float:
    """
    Return max(abs(a - b)) after validating compatible finite arrays.
    """
    a = require_finite_array(a, name="a")
    b = require_finite_array(b, name="b")

    if a.shape != b.shape:
        raise ValueError("a and b must have the same shape")

    if a.size == 0:
        return 0.0

    return float(np.max(np.abs(a - b)))


def allclose_strict(
    a: np.ndarray,
    b: np.ndarray,
    *,
    atol: float = DEFAULT_ATOL,
) -> bool:
    """
    Return np.allclose(a, b) with rtol fixed to zero.

    This is useful for tests where only absolute numerical tolerance is wanted.
    """
    a = require_finite_array(a, name="a")
    b = require_finite_array(b, name="b")

    if a.shape != b.shape:
        return False

    return bool(np.allclose(a, b, atol=atol, rtol=0.0))