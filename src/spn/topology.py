from __future__ import annotations

"""
Topological and phase diagnostics for reduced SPN models.

This module is intentionally modest. It provides numerical diagnostics that
may be useful for reduced and truncated SPN analyses, especially exploratory
Paper-2 / charge-roadmap work:

    - phase wrapping
    - phase differences on S^1
    - winding diagnostics for phase paths
    - accumulated discrete phase around a path
    - simple integer-stability checks

It does not assert that these diagnostics are physical charge, gauge flux,
or a complete topological classification of the full SPN field on L^2(S^2).
"""

import numpy as np


# Numerical tolerance used when checking whether a complex value is too small
# for a stable phase to be defined.
_PHASE_TOL = 1e-14


# ---------------------------------------------------------------------
# Phase helpers
# ---------------------------------------------------------------------


def wrap_phase(angle: float | np.ndarray) -> float | np.ndarray:
    """
    Wrap phase angle(s) to the interval (-pi, pi].

    Parameters
    ----------
    angle:
        A scalar angle or array of angles in radians.

    Returns
    -------
    float | np.ndarray
        Wrapped angle(s).
    """
    wrapped = (np.asarray(angle) + np.pi) % (2.0 * np.pi) - np.pi

    # Map -pi to +pi for the advertised interval (-pi, pi].
    wrapped = np.where(np.isclose(wrapped, -np.pi), np.pi, wrapped)

    if np.isscalar(angle):
        return float(wrapped)

    return wrapped


def phase_difference(a: float, b: float) -> float:
    """
    Return the wrapped phase difference b - a in (-pi, pi].

    Parameters
    ----------
    a:
        Initial phase angle in radians.

    b:
        Final phase angle in radians.
    """
    if not np.isfinite(a) or not np.isfinite(b):
        raise ValueError("phases must be finite")

    return float(wrap_phase(b - a))


def phase_differences(phases: np.ndarray, *, closed: bool = False) -> np.ndarray:
    """
    Return wrapped nearest-neighbour phase differences.

    Parameters
    ----------
    phases:
        1D array of phase angles in radians.

    closed:
        If True, include the final difference from the last phase back to
        the first phase. This is useful for closed loops.

    Returns
    -------
    np.ndarray
        Wrapped phase increments.

    Notes
    -----
    This diagnostic assumes the path is sampled finely enough that wrapped
    nearest-neighbour phase differences correctly track the underlying
    continuous phase evolution.
    """
    phases = np.asarray(phases, dtype=float)

    if phases.ndim != 1:
        raise ValueError("phases must be a 1D array")

    if phases.size < 2:
        raise ValueError("phases must contain at least two values")

    if not np.isfinite(phases).all():
        raise ValueError("phases must contain only finite values")

    if closed:
        next_phases = np.roll(phases, -1)
        return wrap_phase(next_phases - phases)

    return wrap_phase(np.diff(phases))


def unwrap_phases(phases: np.ndarray) -> np.ndarray:
    """
    Return a continuous unwrapped version of a 1D phase path.

    This is a small wrapper around numpy.unwrap with validation.
    """
    phases = np.asarray(phases, dtype=float)

    if phases.ndim != 1:
        raise ValueError("phases must be a 1D array")

    if phases.size == 0:
        raise ValueError("phases must be non-empty")

    if not np.isfinite(phases).all():
        raise ValueError("phases must contain only finite values")

    return np.unwrap(phases)


# ---------------------------------------------------------------------
# Winding diagnostics on S^1
# ---------------------------------------------------------------------


def total_phase_accumulation(phases: np.ndarray, *, closed: bool = True) -> float:
    """
    Return the total wrapped phase accumulation along a phase path.

    Parameters
    ----------
    phases:
        1D array of phase angles in radians.

    closed:
        If True, treat the path as closed and include the final segment from
        the last phase back to the first phase.

    Returns
    -------
    float
        Total accumulated phase in radians.

    Notes
    -----
    Reliable winding extraction requires the phase path to be sufficiently
    well resolved. Poorly sampled paths can produce misleading totals.
    """
    diffs = phase_differences(phases, closed=closed)
    return float(np.sum(diffs))


def winding_number(phases: np.ndarray, *, closed: bool = True) -> float:
    """
    Return the winding diagnostic of a phase path around S^1.

    The diagnostic is computed as:

        total wrapped phase accumulation / (2 pi)

    For a well-resolved closed loop, this should be close to an integer.

    Parameters
    ----------
    phases:
        1D array of phase angles in radians.

    closed:
        If True, include the closing segment from last phase back to first.

    Returns
    -------
    float
        Numerical winding diagnostic.

    Notes
    -----
    If closed=True, this is the appropriate loop-based winding diagnostic
    for a closed phase path on S^1.

    If closed=False, this instead returns normalized net phase accumulation
    along an open path. In that case it is not, in general, a topological
    invariant.
    """
    return float(total_phase_accumulation(phases, closed=closed) / (2.0 * np.pi))


def nearest_integer_winding(phases: np.ndarray, *, closed: bool = True) -> int:
    """
    Return the nearest integer to the numerical winding diagnostic.

    This is useful when the winding is expected to be quantized but numerical
    error produces a value like 0.9999999998.
    """
    return int(np.rint(winding_number(phases, closed=closed)))


def winding_error(phases: np.ndarray, *, closed: bool = True) -> float:
    """
    Return the absolute distance between the winding diagnostic and the
    nearest integer winding.

    Values near zero indicate a numerically stable integer winding.
    """
    w = winding_number(phases, closed=closed)
    return float(abs(w - np.rint(w)))


def is_integer_winding(
    phases: np.ndarray,
    *,
    closed: bool = True,
    atol: float = 1e-8,
) -> bool:
    """
    Return True if the numerical winding diagnostic is close to an integer.

    Parameters
    ----------
    phases:
        1D array of phase angles in radians.

    closed:
        If True, include the closing segment from last phase back to first.

    atol:
        Absolute tolerance for integer closeness.
    """
    if atol < 0:
        raise ValueError("atol must be non-negative")

    return bool(winding_error(phases, closed=closed) <= atol)


# ---------------------------------------------------------------------
# Complex phase-path diagnostics
# ---------------------------------------------------------------------


def complex_phases(z: np.ndarray) -> np.ndarray:
    """
    Return phases arg(z) for a 1D complex path.

    Parameters
    ----------
    z:
        1D complex-valued path.

    Returns
    -------
    np.ndarray
        Phase angles in radians.
    """
    z = np.asarray(z, dtype=complex)

    if z.ndim != 1:
        raise ValueError("z must be a 1D complex array")

    if z.size == 0:
        raise ValueError("z must be non-empty")

    if not np.isfinite(z.real).all() or not np.isfinite(z.imag).all():
        raise ValueError("z must contain only finite values")

    if np.any(np.abs(z) <= _PHASE_TOL):
        raise ValueError(
            "complex phases are undefined for zero or numerically tiny entries"
        )

    return np.angle(z)


def winding_number_complex(z: np.ndarray, *, closed: bool = True) -> float:
    """
    Return the phase winding diagnostic of a complex path around the origin.

    This computes arg(z) and then applies winding_number(...).

    Parameters
    ----------
    z:
        1D complex-valued path.

    closed:
        If True, include the closing segment from last point back to first.
    """
    return winding_number(complex_phases(z), closed=closed)


def nearest_integer_winding_complex(z: np.ndarray, *, closed: bool = True) -> int:
    """
    Return the nearest integer phase winding number of a complex path.
    """
    return nearest_integer_winding(complex_phases(z), closed=closed)


def winding_error_complex(z: np.ndarray, *, closed: bool = True) -> float:
    """
    Return the distance of a complex path's winding from the nearest integer.
    """
    return winding_error(complex_phases(z), closed=closed)


def is_integer_winding_complex(
    z: np.ndarray,
    *,
    closed: bool = True,
    atol: float = 1e-8,
) -> bool:
    """
    Return True if a complex path's phase winding is close to an integer.
    """
    return is_integer_winding(complex_phases(z), closed=closed, atol=atol)


# ---------------------------------------------------------------------
# Spinor phase diagnostics
# ---------------------------------------------------------------------


def relative_phase(spinor: np.ndarray, i: int = 0, j: int = 1) -> float:
    """
    Return the relative phase arg(spinor[j]) - arg(spinor[i]).

    This is useful for two-component reduced states, but works for any
    1D complex state vector with nonzero selected components.

    Parameters
    ----------
    spinor:
        1D complex state vector.

    i, j:
        Component indices.
    """
    spinor = np.asarray(spinor, dtype=complex)

    if spinor.ndim != 1:
        raise ValueError("spinor must be a 1D complex vector")

    if spinor.size == 0:
        raise ValueError("spinor must be non-empty")

    if not np.isfinite(spinor.real).all() or not np.isfinite(spinor.imag).all():
        raise ValueError("spinor must contain only finite values")

    if not (0 <= i < spinor.size) or not (0 <= j < spinor.size):
        raise IndexError("component indices out of range")

    if abs(spinor[i]) <= _PHASE_TOL or abs(spinor[j]) <= _PHASE_TOL:
        raise ValueError(
            "relative phase is undefined for zero or numerically tiny components"
        )

    return phase_difference(np.angle(spinor[i]), np.angle(spinor[j]))


def relative_phases_from_spinors(
    spinors: np.ndarray,
    *,
    i: int = 0,
    j: int = 1,
) -> np.ndarray:
    """
    Return relative phases across a sequence of spinors.

    Parameters
    ----------
    spinors:
        2D array of shape (n_states, n_components).

    i, j:
        Component indices for the relative phase.
    """
    spinors = np.asarray(spinors, dtype=complex)

    if spinors.ndim != 2:
        raise ValueError("spinors must be a 2D array")

    if spinors.shape[0] == 0:
        raise ValueError("spinors must contain at least one state")

    return np.array(
        [relative_phase(spinor, i=i, j=j) for spinor in spinors],
        dtype=float,
    )


def relative_phase_winding_from_spinors(
    spinors: np.ndarray,
    *,
    i: int = 0,
    j: int = 1,
    closed: bool = False,
) -> float:
    """
    Compute winding of the relative phase across a spinor trajectory.

    This is a diagnostic for reduced-state phase evolution. It should not be
    interpreted by itself as a physical charge, gauge flux, or invariant of
    the full SPN field theory.
    """
    phases = relative_phases_from_spinors(spinors, i=i, j=j)
    return winding_number(phases, closed=closed)