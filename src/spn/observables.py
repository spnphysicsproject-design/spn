from __future__ import annotations

"""
Observables for reduced and truncated SPN models.

This module is designed primarily for the Paper-2-style reduced and
finite-dimensional analyses:
    - 2-level Bloch diagnostics for the axisymmetric low sector
    - spectral gaps in reduced Hamiltonians
    - sector-weight and leakage diagnostics in enlarged-basis truncations
    - inter-sector coupling diagnostics

It is not a full observable layer for the continuum SPN field
psi(x, Omega) on L^2(S^2).
"""

from typing import Sequence

import numpy as np


# ---------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------


def _as_finite_vector(x: np.ndarray, *, name: str, dtype=complex) -> np.ndarray:
    """
    Convert input to a finite 1D numpy vector.
    """
    x = np.asarray(x, dtype=dtype)

    if x.ndim != 1:
        raise ValueError(f"{name} must be a 1D vector")

    if x.size == 0:
        raise ValueError(f"{name} must be non-empty")

    if np.iscomplexobj(x):
        finite = np.isfinite(x.real).all() and np.isfinite(x.imag).all()
    else:
        finite = np.isfinite(x).all()

    if not finite:
        raise ValueError(f"{name} must contain only finite values")

    return x


def _as_square_matrix(A: np.ndarray, *, name: str, dtype=complex) -> np.ndarray:
    """
    Convert input to a finite square numpy matrix.
    """
    A = np.asarray(A, dtype=dtype)

    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError(f"{name} must be a square matrix")

    if np.iscomplexobj(A):
        finite = np.isfinite(A.real).all() and np.isfinite(A.imag).all()
    else:
        finite = np.isfinite(A).all()

    if not finite:
        raise ValueError(f"{name} must contain only finite values")

    return A


def is_hermitian(A: np.ndarray, *, atol: float = 1e-12) -> bool:
    """
    Return True if A is Hermitian to numerical tolerance.
    """
    A = np.asarray(A, dtype=complex)

    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        return False

    return bool(np.allclose(A, A.conj().T, atol=atol))


# ---------------------------------------------------------------------
# Basic spectral observables
# ---------------------------------------------------------------------


def spectral_gap(eigenvalues: np.ndarray, i: int = 0, j: int = 1) -> float:
    """
    Return the absolute spectral gap |E_j - E_i| between two eigenvalues.

    Parameters
    ----------
    eigenvalues:
        1D array of eigenvalues.
    i, j:
        Indices of the two levels to compare.
    """
    eigenvalues = _as_finite_vector(eigenvalues, name="eigenvalues", dtype=float)

    if not (0 <= i < len(eigenvalues)) or not (0 <= j < len(eigenvalues)):
        raise IndexError("i and j must be valid eigenvalue indices")

    return float(abs(eigenvalues[j] - eigenvalues[i]))


def ordered_eigensystem(H: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute the ordered eigensystem of a Hermitian matrix.

    Returns
    -------
    eigenvalues:
        Eigenvalues sorted in ascending order.
    eigenvectors:
        Corresponding eigenvectors as columns.
    """
    H = _as_square_matrix(H, name="H", dtype=complex)

    if not is_hermitian(H):
        raise ValueError("H must be Hermitian")

    vals, vecs = np.linalg.eigh(H)
    return vals, vecs


def central_gap_4level(eigenvalues: np.ndarray, *, assume_ordered: bool = True) -> float:
    """
    For a 4-level toy spectrum, return the central gap:

        E_2 - E_1

    using zero-based indexing, i.e. eigenvalues[2] - eigenvalues[1].

    This is useful when a 4-level truncation is interpreted as a lower pair
    and an upper pair, and one wants the separation between those two pairs.

    Parameters
    ----------
    eigenvalues:
        1D array of eigenvalues with length at least 4.
    assume_ordered:
        If True, use the order provided. If False, sort ascending first.
    """
    eigenvalues = _as_finite_vector(eigenvalues, name="eigenvalues", dtype=float)

    if len(eigenvalues) < 4:
        raise ValueError("need at least 4 eigenvalues for central_gap_4level")

    if not assume_ordered:
        eigenvalues = np.sort(eigenvalues)

    return float(eigenvalues[2] - eigenvalues[1])


# ---------------------------------------------------------------------
# State normalization and expectation values
# ---------------------------------------------------------------------


def state_norm(psi: np.ndarray) -> float:
    """
    Return ||psi|| for a finite 1D complex state vector.
    """
    psi = _as_finite_vector(psi, name="psi", dtype=complex)
    return float(np.linalg.norm(psi))


def normalize_state(psi: np.ndarray) -> np.ndarray:
    """
    Normalize a state vector.

    Parameters
    ----------
    psi:
        1D complex state vector.
    """
    psi = _as_finite_vector(psi, name="psi", dtype=complex)

    norm = np.linalg.norm(psi)

    if norm == 0.0:
        raise ValueError("cannot normalize the zero vector")

    return psi / norm


def expectation_value(psi: np.ndarray, operator: np.ndarray) -> complex:
    """
    Compute <psi|A|psi> for a normalized or unnormalized state.

    Parameters
    ----------
    psi:
        1D state vector.
    operator:
        Square matrix acting on the same state space.
    """
    psi = normalize_state(psi)
    operator = _as_square_matrix(operator, name="operator", dtype=complex)

    if operator.shape[0] != len(psi):
        raise ValueError("operator dimension must match state dimension")

    return complex(np.vdot(psi, operator @ psi))


def expectation_value_real(
    psi: np.ndarray,
    operator: np.ndarray,
    *,
    require_hermitian: bool = True,
    atol: float = 1e-10,
) -> float:
    """
    Real part of <psi|A|psi>.

    For Hermitian observables, the expectation value should be real up to
    numerical noise. If require_hermitian=True, this function checks that
    the operator is Hermitian before returning the real part.
    """
    operator = _as_square_matrix(operator, name="operator", dtype=complex)

    if require_hermitian and not is_hermitian(operator, atol=atol):
        raise ValueError("operator must be Hermitian when require_hermitian=True")

    value = expectation_value(psi, operator)

    if require_hermitian and abs(value.imag) > atol:
        raise ValueError(
            "Hermitian expectation value has unexpectedly large imaginary part"
        )

    return float(value.real)


# ---------------------------------------------------------------------
# Bloch / two-level observables
# ---------------------------------------------------------------------


def pauli_matrices() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Return the Pauli matrices (sigma_x, sigma_y, sigma_z).
    """
    sigma_x = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)
    sigma_y = np.array([[0.0, -1j], [1j, 0.0]], dtype=complex)
    sigma_z = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex)
    return sigma_x, sigma_y, sigma_z


def bloch_vector(psi: np.ndarray) -> np.ndarray:
    """
    Compute the Bloch vector for a 2-level state:

        ( <sigma_x>, <sigma_y>, <sigma_z> )

    Parameters
    ----------
    psi:
        2-component state vector.
    """
    psi = normalize_state(psi)

    if psi.shape != (2,):
        raise ValueError("bloch_vector requires a 2-component state")

    sx, sy, sz = pauli_matrices()

    return np.array(
        [
            expectation_value_real(psi, sx),
            expectation_value_real(psi, sy),
            expectation_value_real(psi, sz),
        ],
        dtype=float,
    )


def bloch_radius(psi: np.ndarray) -> float:
    """
    Magnitude of the Bloch vector.

    For a pure normalized 2-level state this should be 1.
    """
    return float(np.linalg.norm(bloch_vector(psi)))


def bloch_vectors_from_states(states: Sequence[np.ndarray]) -> np.ndarray:
    """
    Compute Bloch vectors for a sequence of 2-component states.

    Returns
    -------
    np.ndarray
        Array of shape (n_states, 3).
    """
    return np.array([bloch_vector(psi) for psi in states], dtype=float)


# ---------------------------------------------------------------------
# Sector decomposition observables
# ---------------------------------------------------------------------


def _validate_sector_dims(sector_dims: Sequence[int], total_dim: int) -> None:
    """
    Validate sector sizes for contiguous-sector decompositions.

    Parameters
    ----------
    sector_dims:
        Positive sizes of contiguous sectors.
    total_dim:
        Total state or matrix dimension.
    """
    if len(sector_dims) == 0:
        raise ValueError("sector_dims must be non-empty")

    if any(dim <= 0 for dim in sector_dims):
        raise ValueError("all sector dimensions must be positive")

    if sum(sector_dims) != total_dim:
        raise ValueError("sector_dims must sum to the total dimension")


def sector_weights(psi: np.ndarray, sector_dims: Sequence[int]) -> np.ndarray:
    """
    Compute the probability weight of a state in each contiguous sector.

    Example
    -------
    For a 4D state with two contiguous 2D sectors, use:

        sector_dims = [2, 2]

    Important
    ---------
    This is a basis-dependent diagnostic: the sectors are assumed to
    correspond to contiguous blocks in the chosen basis ordering.

    Parameters
    ----------
    psi:
        1D state vector.
    sector_dims:
        Positive sizes of contiguous sectors summing to len(psi).

    Returns
    -------
    np.ndarray
        Sector weights summing to 1.
    """
    psi = normalize_state(psi)
    _validate_sector_dims(sector_dims, len(psi))

    weights = []
    start = 0

    for dim in sector_dims:
        block = psi[start : start + dim]
        weights.append(float(np.sum(np.abs(block) ** 2)))
        start += dim

    return np.array(weights, dtype=float)


def sector_purity(psi: np.ndarray, sector_dims: Sequence[int]) -> float:
    """
    Return the largest sector weight.

    Values near 1 mean the state is mostly concentrated in one sector.
    """
    return float(np.max(sector_weights(psi, sector_dims)))


def dominant_sector(psi: np.ndarray, sector_dims: Sequence[int]) -> int:
    """
    Return the index of the dominant sector.
    """
    return int(np.argmax(sector_weights(psi, sector_dims)))


def sector_mixing_ratio(psi: np.ndarray, sector_dims: Sequence[int]) -> float:
    """
    A simple mixing diagnostic for multi-sector states.

    Defined as:

        1 - max sector weight

    So:

        0   -> perfectly concentrated in one sector
        larger values -> more mixed across sectors
    """
    return float(1.0 - sector_purity(psi, sector_dims))


def low_sector_retention(psi: np.ndarray, low_dim: int) -> float:
    """
    Return the total probability weight in the first low_dim components.

    This is a convenience helper for common Paper-2-style truncations in
    which the low sector is represented by the first block of the basis.

    Parameters
    ----------
    psi:
        1D state vector.
    low_dim:
        Dimension of the contiguous low sector at the start of the basis.
    """
    psi = normalize_state(psi)

    if low_dim <= 0 or low_dim > len(psi):
        raise ValueError("low_dim must be between 1 and len(psi)")

    return float(np.sum(np.abs(psi[:low_dim]) ** 2))


def leakage_out_of_low_sector(psi: np.ndarray, low_dim: int) -> float:
    """
    Return the probability weight outside the first low_dim components.

    This is:

        1 - low_sector_retention(psi, low_dim)
    """
    return float(1.0 - low_sector_retention(psi, low_dim))


def sector_weights_from_states(
    states: Sequence[np.ndarray],
    sector_dims: Sequence[int],
) -> np.ndarray:
    """
    Compute sector weights across a sequence of states.

    Returns
    -------
    np.ndarray
        Array of shape (n_states, n_sectors).
    """
    return np.array([sector_weights(psi, sector_dims) for psi in states], dtype=float)


def low_sector_retention_from_states(
    states: Sequence[np.ndarray],
    low_dim: int,
) -> np.ndarray:
    """
    Compute low-sector retention across a sequence of states.
    """
    return np.array([low_sector_retention(psi, low_dim) for psi in states], dtype=float)


def leakage_from_states(
    states: Sequence[np.ndarray],
    low_dim: int,
) -> np.ndarray:
    """
    Compute leakage out of the low sector across a sequence of states.
    """
    return np.array(
        [leakage_out_of_low_sector(psi, low_dim) for psi in states],
        dtype=float,
    )


# ---------------------------------------------------------------------
# Matrix-level inter-sector diagnostics
# ---------------------------------------------------------------------


def block_frobenius_norm(H: np.ndarray, row_slice: slice, col_slice: slice) -> float:
    """
    Frobenius norm of a matrix block.

    Useful for measuring inter-sector coupling strength.
    """
    H = _as_square_matrix(H, name="H", dtype=complex)

    block = H[row_slice, col_slice]
    return float(np.linalg.norm(block))


def inter_sector_coupling_norm(
    H: np.ndarray,
    sector_dims: Sequence[int],
    a: int,
    b: int,
) -> float:
    """
    Compute the Frobenius norm of the block coupling sector a to sector b.

    Important
    ---------
    This is a basis-dependent diagnostic: sectors are assumed to be contiguous
    blocks in the chosen matrix basis ordering.

    Parameters
    ----------
    H:
        Square matrix.
    sector_dims:
        Positive contiguous sector sizes.
    a, b:
        Sector indices.
    """
    H = _as_square_matrix(H, name="H", dtype=complex)
    _validate_sector_dims(sector_dims, H.shape[0])

    if not (0 <= a < len(sector_dims)) or not (0 <= b < len(sector_dims)):
        raise IndexError("sector indices out of range")

    bounds = np.cumsum([0] + list(sector_dims))
    row_slice = slice(bounds[a], bounds[a + 1])
    col_slice = slice(bounds[b], bounds[b + 1])

    return block_frobenius_norm(H, row_slice, col_slice)


def coupling_to_gap_ratio(
    eigenvalues: np.ndarray,
    coupling_norm: float,
    i: int,
    j: int,
) -> float:
    """
    Compute the simple ratio:

        R = coupling_norm / |E_j - E_i|

    Small R suggests weak mixing relative to spectral separation.

    This is a crude diagnostic, not a perturbation-theory proof.
    """
    if coupling_norm < 0:
        raise ValueError("coupling_norm must be non-negative")

    gap = spectral_gap(eigenvalues, i=i, j=j)

    if gap == 0.0:
        return np.inf

    return float(coupling_norm / gap)


def gap_to_coupling_ratio(
    eigenvalues: np.ndarray,
    coupling_norm: float,
    i: int,
    j: int,
) -> float:
    """
    Backwards-compatible alias for coupling_to_gap_ratio.

    Note:
        Despite the historical name, this returns coupling_norm / gap,
        not gap / coupling_norm.
    """
    return coupling_to_gap_ratio(
        eigenvalues=eigenvalues,
        coupling_norm=coupling_norm,
        i=i,
        j=j,
    )