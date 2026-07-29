"""Tests for run-length resolution: fixed iterations vs. a circuit-measurement budget."""

import pytest

from src.qcbm import resolve_iterations


def test_iterations_mode_passes_through():
    assert resolve_iterations("iterations", 1000, 0, 3_554_000) == 1000
    # the budget is ignored entirely in this mode, even an unaffordable one
    assert resolve_iterations("iterations", 7, 1, 3_554_000) == 7


def test_budget_mode_floors_to_whole_iterations():
    """A final iteration that would exceed the budget is cut off, not partially run."""
    per_step = 1000
    assert resolve_iterations("measurements", 10, 10_000, per_step) == 10
    assert resolve_iterations("measurements", 10, 10_999, per_step) == 10   # remainder dropped
    assert resolve_iterations("measurements", 10, 11_000, per_step) == 11


def test_budget_mode_ignores_the_iterations_setting():
    assert resolve_iterations("measurements", 5, 100_000, 1000) == 100
    assert resolve_iterations("measurements", 500, 100_000, 1000) == 100


def test_budget_mode_penalizes_more_parameters():
    """The point of the mode: equal cost, so a denser circuit gets fewer iterations."""
    budget, shots = 10_000_000, 1000
    sparse = resolve_iterations("measurements", 0, budget, (2 * 100 + 1) * shots)
    dense = resolve_iterations("measurements", 0, budget, (2 * 400 + 1) * shots)
    assert sparse > dense
    assert sparse * (2 * 100 + 1) * shots <= budget
    assert dense * (2 * 400 + 1) * shots <= budget


def test_budget_too_small_for_one_iteration_is_an_error():
    with pytest.raises(ValueError, match="does not cover a single training iteration"):
        resolve_iterations("measurements", 10, 999, 1000)


def test_invalid_inputs():
    with pytest.raises(ValueError, match="qcbm.mode must be"):
        resolve_iterations("budget", 10, 1000, 1000)
    with pytest.raises(ValueError, match="iterations must be >= 1"):
        resolve_iterations("iterations", 0, 1000, 1000)
