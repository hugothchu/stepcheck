"""The worker process that puts a time limit on SymPy calls."""

import time

import pytest
from sympy import Eq, Integer, Rational, Sum, symbols

from stepcheck.timeout import CheckTimeout, run_with_timeout

k, n = symbols("k n")


def test_without_a_limit_the_call_runs_here() -> None:
    assert run_with_timeout(pow, 2, 10, seconds=None) == 1024


def test_a_slow_call_times_out_and_the_worker_recovers() -> None:
    with pytest.raises(CheckTimeout):
        run_with_timeout(time.sleep, 5, seconds=0.5)
    assert run_with_timeout(pow, 2, 10, seconds=30) == 1024  # a fresh worker


def test_errors_are_raised_here() -> None:
    with pytest.raises(ValueError, match="invalid literal"):
        run_with_timeout(int, "not a number", seconds=30)


@pytest.mark.parametrize(
    "value",
    [
        Eq(Integer(12), Integer(13), evaluate=False),  # pickling alone gives False
        Eq(Rational(1, 10), Rational(1, 5), evaluate=False),
        Sum(k**2, (k, 1, n)),
        [Eq(Integer(1), Integer(2), evaluate=False), Eq(k, 3)],
    ],
)
def test_sympy_objects_arrive_unchanged(value: object) -> None:
    assert run_with_timeout(_identity, value, seconds=30) == value


def _identity(value: object) -> object:
    return value
