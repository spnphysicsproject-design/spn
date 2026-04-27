from __future__ import annotations

import numpy as np
import pytest

from spn.utils import (
    DEFAULT_ATOL,
    is_finite_array,
    require_finite_array,
    require_scalar_finite,
    require_nonnegative,
    require_positive,
    as_finite_vector,
    as_finite_vector3,
    as_finite_matrix,
    as_square_matrix,
    vector_norm,
    normalize_vector,
    safe_unit_vector,
    is_hermitian,
    require_hermitian,
    is_unitary,
    require_unitary,
    probability_weights,
    sums_to_one,
    max_abs_difference,
    allclose_strict,
)


# ---------------------------------------------------------------------
# Default tolerance
# ---------------------------------------------------------------------


def test_default_tolerance_is_positive() -> None:
    assert DEFAULT_ATOL > 0.0


# ---------------------------------------------------------------------
# Basic finite-value helpers
# ---------------------------------------------------------------------


def test_is_finite_array_true_for_real_finite_array() -> None:
    x = np.array([1.0, 2.0, 3.0])

    assert is_finite_array(x)


def test_is_finite_array_false_for_real_nonfinite_array() -> None:
    x = np.array([1.0, np.inf, 3.0])

    assert not is_finite_array(x)


def test_is_finite_array_true_for_complex_finite_array() -> None:
    x = np.array([1.0 + 1.0j, 2.0 - 3.0j])

    assert is_finite_array(x)


def test_is_finite_array_false_for_complex_nonfinite_array() -> None:
    x = np.array([1.0 + 1.0j, np.nan + 0.0j])

    assert not is_finite_array(x)


def test_require_finite_array_returns_array() -> None:
    x = require_finite_array([1.0, 2.0, 3.0], name="x")

    assert isinstance(x, np.ndarray)
    assert np.allclose(x, np.array([1.0, 2.0, 3.0]))


def test_require_finite_array_accepts_complex_finite_array() -> None:
    x = require_finite_array(
        np.array([1.0 + 1.0j, 2.0 - 3.0j]),
        name="x",
    )

    assert isinstance(x, np.ndarray)
    assert x.shape == (2,)
    assert np.allclose(x, np.array([1.0 + 1.0j, 2.0 - 3.0j]))


def test_require_finite_array_rejects_nonfinite_values() -> None:
    with pytest.raises(ValueError):
        require_finite_array([1.0, np.inf], name="x")


def test_require_scalar_finite_returns_float() -> None:
    value = require_scalar_finite(3, name="value")

    assert isinstance(value, float)
    assert value == pytest.approx(3.0)


def test_require_scalar_finite_rejects_nonfinite_value() -> None:
    with pytest.raises(ValueError):
        require_scalar_finite(np.inf, name="value")


def test_require_nonnegative_accepts_zero_and_positive_values() -> None:
    assert require_nonnegative(0.0, name="x") == pytest.approx(0.0)
    assert require_nonnegative(2.0, name="x") == pytest.approx(2.0)


def test_require_nonnegative_rejects_negative_value() -> None:
    with pytest.raises(ValueError):
        require_nonnegative(-1.0, name="x")


def test_require_positive_accepts_positive_value() -> None:
    assert require_positive(2.0, name="x") == pytest.approx(2.0)


def test_require_positive_rejects_zero_and_negative_values() -> None:
    with pytest.raises(ValueError):
        require_positive(0.0, name="x")

    with pytest.raises(ValueError):
        require_positive(-1.0, name="x")


# ---------------------------------------------------------------------
# Vector and matrix validation
# ---------------------------------------------------------------------


def test_as_finite_vector_returns_1d_vector() -> None:
    x = as_finite_vector([1.0, 2.0], name="x", dtype=float)

    assert isinstance(x, np.ndarray)
    assert x.shape == (2,)
    assert x.dtype == float


def test_as_finite_vector_rejects_non_1d_input() -> None:
    with pytest.raises(ValueError):
        as_finite_vector(np.eye(2), name="x")


def test_as_finite_vector_rejects_empty_when_nonempty_true() -> None:
    with pytest.raises(ValueError):
        as_finite_vector(np.array([]), name="x", nonempty=True)


def test_as_finite_vector_allows_empty_when_nonempty_false() -> None:
    x = as_finite_vector(np.array([]), name="x", nonempty=False)

    assert x.shape == (0,)


def test_as_finite_vector_rejects_nonfinite_values() -> None:
    with pytest.raises(ValueError):
        as_finite_vector(np.array([1.0, np.nan]), name="x", dtype=float)


def test_as_finite_vector3_returns_3_vector() -> None:
    x = as_finite_vector3([1.0, 2.0, 3.0], name="x")

    assert x.shape == (3,)
    assert np.allclose(x, np.array([1.0, 2.0, 3.0]))


def test_as_finite_vector3_rejects_non_3_vector() -> None:
    with pytest.raises(ValueError):
        as_finite_vector3([1.0, 2.0], name="x")


def test_as_finite_matrix_returns_2d_matrix() -> None:
    A = as_finite_matrix([[1.0, 2.0], [3.0, 4.0]], name="A", dtype=float)

    assert A.shape == (2, 2)
    assert A.dtype == float


def test_as_finite_matrix_rejects_non_2d_input() -> None:
    with pytest.raises(ValueError):
        as_finite_matrix(np.array([1.0, 2.0]), name="A")


def test_as_finite_matrix_rejects_nonfinite_values() -> None:
    with pytest.raises(ValueError):
        as_finite_matrix(np.array([[1.0, np.inf]]), name="A", dtype=float)


def test_as_square_matrix_returns_square_matrix() -> None:
    A = as_square_matrix([[1.0, 0.0], [0.0, 1.0]], name="A")

    assert A.shape == (2, 2)


def test_as_square_matrix_rejects_non_square_matrix() -> None:
    with pytest.raises(ValueError):
        as_square_matrix(np.ones((2, 3)), name="A")


# ---------------------------------------------------------------------
# Norm and normalisation helpers
# ---------------------------------------------------------------------


def test_vector_norm_returns_euclidean_norm() -> None:
    x = np.array([3.0 + 0.0j, 4.0 + 0.0j])

    assert vector_norm(x, name="x") == pytest.approx(5.0)


def test_normalize_vector_returns_unit_vector() -> None:
    x = np.array([3.0 + 0.0j, 4.0 + 0.0j])

    normalized = normalize_vector(x, name="x")

    assert np.linalg.norm(normalized) == pytest.approx(1.0)
    assert np.allclose(normalized, x / 5.0)


def test_normalize_vector_rejects_zero_vector() -> None:
    with pytest.raises(ValueError):
        normalize_vector(np.zeros(3, dtype=complex), name="x")


def test_safe_unit_vector_returns_unit_direction() -> None:
    x = np.array([3.0, 4.0, 0.0])

    unit = safe_unit_vector(x, name="x")

    assert np.allclose(unit, np.array([0.6, 0.8, 0.0]))
    assert np.linalg.norm(unit) == pytest.approx(1.0)


def test_safe_unit_vector_rejects_zero_by_default() -> None:
    with pytest.raises(ValueError):
        safe_unit_vector(np.zeros(3), name="x")


def test_safe_unit_vector_can_return_zero_for_zero_vector() -> None:
    unit = safe_unit_vector(np.zeros(3), name="x", zero="zero")

    assert np.allclose(unit, np.zeros(3))


def test_safe_unit_vector_rejects_invalid_zero_mode() -> None:
    with pytest.raises(ValueError):
        safe_unit_vector(np.zeros(3), name="x", zero="bad-mode")


# ---------------------------------------------------------------------
# Matrix property checks
# ---------------------------------------------------------------------


def test_is_hermitian_true_for_hermitian_matrix() -> None:
    H = np.array(
        [
            [1.0, 1.0 + 2.0j],
            [1.0 - 2.0j, 3.0],
        ],
        dtype=complex,
    )

    assert is_hermitian(H)


def test_is_hermitian_false_for_non_hermitian_matrix() -> None:
    A = np.array(
        [
            [1.0, 2.0],
            [3.0, 4.0],
        ],
        dtype=complex,
    )

    assert not is_hermitian(A)


def test_is_hermitian_false_for_non_square_matrix() -> None:
    A = np.ones((2, 3), dtype=complex)

    assert not is_hermitian(A)


def test_require_hermitian_returns_matrix_for_hermitian_input() -> None:
    H = np.array([[1.0, 0.0], [0.0, 2.0]], dtype=complex)

    result = require_hermitian(H, name="H")

    assert np.allclose(result, H)


def test_require_hermitian_preserves_shape_and_dtype() -> None:
    H = np.array(
        [
            [1.0 + 0.0j, 0.5 - 0.25j],
            [0.5 + 0.25j, 2.0 + 0.0j],
        ],
        dtype=complex,
    )

    result = require_hermitian(H, name="H")

    assert result.shape == H.shape
    assert result.dtype == complex
    assert np.allclose(result, H)


def test_require_hermitian_rejects_non_hermitian_input() -> None:
    A = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=complex)

    with pytest.raises(ValueError):
        require_hermitian(A, name="A")


def test_is_unitary_true_for_unitary_matrix() -> None:
    U = np.array(
        [
            [1.0, 0.0],
            [0.0, np.exp(1.0j)],
        ],
        dtype=complex,
    )

    assert is_unitary(U)


def test_is_unitary_false_for_non_unitary_matrix() -> None:
    U = np.array(
        [
            [2.0, 0.0],
            [0.0, 1.0],
        ],
        dtype=complex,
    )

    assert not is_unitary(U)


def test_is_unitary_false_for_non_square_matrix() -> None:
    U = np.ones((2, 3), dtype=complex)

    assert not is_unitary(U)


def test_require_unitary_returns_matrix_for_unitary_input() -> None:
    U = np.eye(2, dtype=complex)

    result = require_unitary(U, name="U")

    assert np.allclose(result, U)


def test_require_unitary_preserves_shape_and_dtype() -> None:
    phase = np.exp(1.0j)
    U = np.array(
        [
            [1.0 + 0.0j, 0.0 + 0.0j],
            [0.0 + 0.0j, phase],
        ],
        dtype=complex,
    )

    result = require_unitary(U, name="U")

    assert result.shape == U.shape
    assert result.dtype == complex
    assert np.allclose(result, U)


def test_require_unitary_rejects_non_unitary_input() -> None:
    U = np.array([[2.0, 0.0], [0.0, 1.0]], dtype=complex)

    with pytest.raises(ValueError):
        require_unitary(U, name="U")


# ---------------------------------------------------------------------
# Probability and diagnostic helpers
# ---------------------------------------------------------------------


def test_probability_weights_normalizes_by_default() -> None:
    psi = np.array([1.0 + 0.0j, 1.0 + 0.0j])

    weights = probability_weights(psi)

    assert np.allclose(weights, np.array([0.5, 0.5]))
    assert np.sum(weights) == pytest.approx(1.0)


def test_probability_weights_can_skip_normalization() -> None:
    psi = np.array([2.0 + 0.0j, 1.0 + 0.0j])

    weights = probability_weights(psi, normalize=False)

    assert np.allclose(weights, np.array([4.0, 1.0]))


def test_probability_weights_handles_complex_state() -> None:
    psi = np.array([1.0 + 1.0j, 1.0j])

    weights = probability_weights(psi, normalize=False)

    assert np.allclose(weights, np.array([2.0, 1.0]))


def test_probability_weights_rejects_zero_vector_when_normalizing() -> None:
    with pytest.raises(ValueError):
        probability_weights(np.zeros(2, dtype=complex), normalize=True)


def test_sums_to_one_true_for_normalized_weights() -> None:
    weights = np.array([0.25, 0.75])

    assert sums_to_one(weights)


def test_sums_to_one_false_for_unnormalized_weights() -> None:
    weights = np.array([0.25, 0.5])

    assert not sums_to_one(weights)


def test_sums_to_one_rejects_non_vector() -> None:
    with pytest.raises(ValueError):
        sums_to_one(np.eye(2))


def test_max_abs_difference_returns_max_difference() -> None:
    a = np.array([1.0, 2.0, 3.0])
    b = np.array([1.0, 2.5, 1.0])

    assert max_abs_difference(a, b) == pytest.approx(2.0)


def test_max_abs_difference_rejects_shape_mismatch() -> None:
    with pytest.raises(ValueError):
        max_abs_difference(np.ones(2), np.ones(3))


def test_max_abs_difference_returns_zero_for_empty_arrays() -> None:
    assert max_abs_difference(np.array([]), np.array([])) == pytest.approx(0.0)


def test_allclose_strict_true_for_close_arrays() -> None:
    a = np.array([1.0, 2.0])
    b = np.array([1.0 + 1e-13, 2.0 - 1e-13])

    assert allclose_strict(a, b, atol=1e-12)


def test_allclose_strict_false_for_arrays_outside_absolute_tolerance() -> None:
    a = np.array([1.0, 2.0])
    b = np.array([1.0 + 1e-6, 2.0])

    assert not allclose_strict(a, b, atol=1e-12)


def test_allclose_strict_false_for_shape_mismatch() -> None:
    assert not allclose_strict(np.ones(2), np.ones(3))