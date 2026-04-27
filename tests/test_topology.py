from __future__ import annotations

import numpy as np
import pytest

from spn.topology import (
    wrap_phase,
    phase_difference,
    phase_differences,
    unwrap_phases,
    total_phase_accumulation,
    winding_number,
    nearest_integer_winding,
    winding_error,
    is_integer_winding,
    complex_phases,
    winding_number_complex,
    nearest_integer_winding_complex,
    winding_error_complex,
    is_integer_winding_complex,
    relative_phase,
    relative_phases_from_spinors,
    relative_phase_winding_from_spinors,
)


# ---------------------------------------------------------------------
# Phase helpers
# ---------------------------------------------------------------------


def test_wrap_phase_scalar_maps_to_expected_interval() -> None:
    assert wrap_phase(0.0) == pytest.approx(0.0)
    assert wrap_phase(np.pi) == pytest.approx(np.pi)
    assert wrap_phase(-np.pi) == pytest.approx(np.pi)
    assert wrap_phase(3.0 * np.pi) == pytest.approx(np.pi)


def test_wrap_phase_array() -> None:
    phases = np.array([0.0, np.pi, -np.pi, 3.0 * np.pi])
    wrapped = wrap_phase(phases)

    expected = np.array([0.0, np.pi, np.pi, np.pi])

    assert isinstance(wrapped, np.ndarray)
    assert np.allclose(wrapped, expected)


def test_phase_difference_wraps_b_minus_a() -> None:
    a = 3.0 * np.pi / 4.0
    b = -3.0 * np.pi / 4.0

    # Raw b - a = -3pi/2, wrapped to +pi/2.
    assert phase_difference(a, b) == pytest.approx(np.pi / 2.0)


def test_phase_difference_rejects_nonfinite_values() -> None:
    with pytest.raises(ValueError):
        phase_difference(np.inf, 0.0)

    with pytest.raises(ValueError):
        phase_difference(0.0, np.nan)


def test_phase_differences_open_path() -> None:
    phases = np.array([0.0, np.pi / 2.0, np.pi])

    diffs = phase_differences(phases, closed=False)

    assert np.allclose(diffs, np.array([np.pi / 2.0, np.pi / 2.0]))


def test_phase_differences_closed_path_includes_closing_segment() -> None:
    phases = np.array([0.0, np.pi / 2.0, np.pi])

    diffs = phase_differences(phases, closed=True)

    # 0 -> pi/2, pi/2 -> pi, pi -> 0 gives +pi after wrapping
    # because wrap_phase(-pi) maps to +pi.
    expected = np.array([np.pi / 2.0, np.pi / 2.0, np.pi])

    assert np.allclose(diffs, expected)


def test_phase_differences_rejects_non_1d_input() -> None:
    with pytest.raises(ValueError):
        phase_differences(np.eye(2))


def test_phase_differences_rejects_too_few_values() -> None:
    with pytest.raises(ValueError):
        phase_differences(np.array([0.0]))


def test_phase_differences_rejects_nonfinite_values() -> None:
    with pytest.raises(ValueError):
        phase_differences(np.array([0.0, np.inf]))


def test_unwrap_phases_returns_continuous_path() -> None:
    phases = np.array([0.0, 0.9 * np.pi, -0.9 * np.pi])

    unwrapped = unwrap_phases(phases)

    assert unwrapped.shape == phases.shape
    assert unwrapped[2] > unwrapped[1]


def test_unwrap_phases_rejects_empty_input() -> None:
    with pytest.raises(ValueError):
        unwrap_phases(np.array([]))


def test_unwrap_phases_rejects_non_1d_input() -> None:
    with pytest.raises(ValueError):
        unwrap_phases(np.eye(2))


# ---------------------------------------------------------------------
# Winding diagnostics on S^1
# ---------------------------------------------------------------------


def test_total_phase_accumulation_open_path() -> None:
    phases = np.array([0.0, np.pi / 2.0, np.pi])

    assert total_phase_accumulation(phases, closed=False) == pytest.approx(np.pi)


def test_winding_number_open_path_is_normalized_net_accumulation() -> None:
    phases = np.array([0.0, np.pi / 2.0, np.pi])

    assert winding_number(phases, closed=False) == pytest.approx(0.5)


def test_winding_number_closed_unit_loop() -> None:
    phases = np.array(
        [
            0.0,
            np.pi / 2.0,
            np.pi,
            3.0 * np.pi / 2.0,
        ]
    )

    assert winding_number(phases, closed=True) == pytest.approx(1.0)


def test_nearest_integer_winding_rounds_winding() -> None:
    phases = np.array(
        [
            0.0,
            np.pi / 2.0,
            np.pi,
            3.0 * np.pi / 2.0,
        ]
    )

    assert nearest_integer_winding(phases, closed=True) == 1


def test_winding_error_is_small_for_integer_winding() -> None:
    phases = np.array(
        [
            0.0,
            np.pi / 2.0,
            np.pi,
            3.0 * np.pi / 2.0,
        ]
    )

    assert winding_error(phases, closed=True) < 1e-12


def test_is_integer_winding_true_for_closed_unit_loop() -> None:
    phases = np.array(
        [
            0.0,
            np.pi / 2.0,
            np.pi,
            3.0 * np.pi / 2.0,
        ]
    )

    assert is_integer_winding(phases, closed=True)


def test_is_integer_winding_rejects_negative_tolerance() -> None:
    phases = np.array([0.0, np.pi])

    with pytest.raises(ValueError):
        is_integer_winding(phases, atol=-1.0)


# ---------------------------------------------------------------------
# Complex phase-path diagnostics
# ---------------------------------------------------------------------


def test_complex_phases_returns_angles() -> None:
    z = np.array([1.0 + 0.0j, 1.0j, -1.0 + 0.0j])

    phases = complex_phases(z)

    assert np.allclose(phases, np.array([0.0, np.pi / 2.0, np.pi]))


def test_complex_phases_rejects_non_1d_input() -> None:
    with pytest.raises(ValueError):
        complex_phases(np.eye(2, dtype=complex))


def test_complex_phases_rejects_empty_input() -> None:
    with pytest.raises(ValueError):
        complex_phases(np.array([], dtype=complex))


def test_complex_phases_rejects_nonfinite_input() -> None:
    with pytest.raises(ValueError):
        complex_phases(np.array([1.0 + 0.0j, np.inf + 0.0j]))


def test_complex_phases_rejects_zero_or_tiny_entries() -> None:
    with pytest.raises(ValueError):
        complex_phases(np.array([1.0 + 0.0j, 0.0 + 0.0j]))

    with pytest.raises(ValueError):
        complex_phases(np.array([1.0 + 0.0j, 1e-16 + 0.0j]))


def test_winding_number_complex_open_path_is_normalized_net_accumulation() -> None:
    z = np.array(
        [
            1.0 + 0.0j,
            0.0 + 1.0j,
            -1.0 + 0.0j,
        ],
        dtype=complex,
    )

    # Phases: 0 -> pi/2 -> pi, total open accumulation = pi.
    assert winding_number_complex(z, closed=False) == pytest.approx(0.5)


def test_winding_number_complex_for_unit_circle_loop() -> None:
    z = np.array(
        [
            1.0 + 0.0j,
            0.0 + 1.0j,
            -1.0 + 0.0j,
            0.0 - 1.0j,
        ],
        dtype=complex,
    )

    assert winding_number_complex(z, closed=True) == pytest.approx(1.0)


def test_nearest_integer_winding_complex_for_unit_circle_loop() -> None:
    z = np.array(
        [
            1.0 + 0.0j,
            0.0 + 1.0j,
            -1.0 + 0.0j,
            0.0 - 1.0j,
        ],
        dtype=complex,
    )

    assert nearest_integer_winding_complex(z, closed=True) == 1


def test_winding_error_complex_is_small_for_unit_circle_loop() -> None:
    z = np.array(
        [
            1.0 + 0.0j,
            0.0 + 1.0j,
            -1.0 + 0.0j,
            0.0 - 1.0j,
        ],
        dtype=complex,
    )

    assert winding_error_complex(z, closed=True) < 1e-12


def test_is_integer_winding_complex_for_unit_circle_loop() -> None:
    z = np.array(
        [
            1.0 + 0.0j,
            0.0 + 1.0j,
            -1.0 + 0.0j,
            0.0 - 1.0j,
        ],
        dtype=complex,
    )

    assert is_integer_winding_complex(z, closed=True)


# ---------------------------------------------------------------------
# Spinor phase diagnostics
# ---------------------------------------------------------------------


def test_relative_phase_between_two_components() -> None:
    spinor = np.array([1.0 + 0.0j, 1.0j])

    assert relative_phase(spinor, i=0, j=1) == pytest.approx(np.pi / 2.0)


def test_relative_phase_wraps_result() -> None:
    spinor = np.array(
        [
            np.exp(1j * 3.0 * np.pi / 4.0),
            np.exp(-1j * 3.0 * np.pi / 4.0),
        ]
    )

    # Relative phase is -3pi/2, wrapped to +pi/2.
    assert relative_phase(spinor, i=0, j=1) == pytest.approx(np.pi / 2.0)


def test_relative_phase_rejects_non_1d_spinor() -> None:
    with pytest.raises(ValueError):
        relative_phase(np.eye(2, dtype=complex))


def test_relative_phase_rejects_empty_spinor() -> None:
    with pytest.raises(ValueError):
        relative_phase(np.array([], dtype=complex))


def test_relative_phase_rejects_nonfinite_spinor() -> None:
    with pytest.raises(ValueError):
        relative_phase(np.array([1.0 + 0.0j, np.nan + 0.0j]))


def test_relative_phase_rejects_bad_indices() -> None:
    spinor = np.array([1.0 + 0.0j, 1.0j])

    with pytest.raises(IndexError):
        relative_phase(spinor, i=0, j=2)


def test_relative_phase_rejects_zero_or_tiny_components() -> None:
    with pytest.raises(ValueError):
        relative_phase(np.array([1.0 + 0.0j, 0.0 + 0.0j]))

    with pytest.raises(ValueError):
        relative_phase(np.array([1.0 + 0.0j, 1e-16 + 0.0j]))


def test_relative_phases_from_spinors_returns_expected_values() -> None:
    spinors = np.array(
        [
            [1.0 + 0.0j, 1.0 + 0.0j],
            [1.0 + 0.0j, 1.0j],
            [1.0 + 0.0j, -1.0 + 0.0j],
        ],
        dtype=complex,
    )

    phases = relative_phases_from_spinors(spinors)

    assert np.allclose(phases, np.array([0.0, np.pi / 2.0, np.pi]))


def test_relative_phases_from_spinors_rejects_non_2d_input() -> None:
    with pytest.raises(ValueError):
        relative_phases_from_spinors(np.ones(2, dtype=complex))


def test_relative_phases_from_spinors_rejects_empty_sequence() -> None:
    with pytest.raises(ValueError):
        relative_phases_from_spinors(np.empty((0, 2), dtype=complex))


def test_relative_phases_from_spinors_rejects_tiny_component() -> None:
    spinors = np.array(
        [
            [1.0 + 0.0j, 1.0 + 0.0j],
            [1.0 + 0.0j, 1e-16 + 0.0j],
        ],
        dtype=complex,
    )

    with pytest.raises(ValueError):
        relative_phases_from_spinors(spinors)


def test_relative_phase_winding_from_spinors_open_path() -> None:
    spinors = np.array(
        [
            [1.0 + 0.0j, 1.0 + 0.0j],
            [1.0 + 0.0j, 1.0j],
            [1.0 + 0.0j, -1.0 + 0.0j],
        ],
        dtype=complex,
    )

    assert relative_phase_winding_from_spinors(spinors, closed=False) == pytest.approx(
        0.5
    )


def test_relative_phase_winding_from_spinors_closed_loop() -> None:
    spinors = np.array(
        [
            [1.0 + 0.0j, 1.0 + 0.0j],
            [1.0 + 0.0j, 1.0j],
            [1.0 + 0.0j, -1.0 + 0.0j],
            [1.0 + 0.0j, -1.0j],
        ],
        dtype=complex,
    )

    assert relative_phase_winding_from_spinors(spinors, closed=True) == pytest.approx(
        1.0
    )