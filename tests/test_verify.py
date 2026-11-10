"""The two checkers on their own: equivalence (simplify) and consistency (solve)."""

import pytest
from sympy import Symbol, simplify, sympify

from stepcheck.models import Method, Verdict
from stepcheck.parse import parse_expression, parse_statement
from stepcheck.verify import check_identity, check_statement

# ------------------------------------------------------------------- identities


@pytest.mark.parametrize(
    ("left", "right"),
    [
        (r"(x + 1)^2", r"x^2 + 2x + 1"),
        (r"\sin^{2} x + \cos^{2} x", r"1"),
        (r"\sin 2\theta", r"2 \sin\theta \cos\theta"),
        (r"\cos 2x", r"1 - 2\sin^{2} x"),
        (r"\frac{x^2 - 1}{x - 1}", r"x + 1"),  # equal wherever both sides are defined
        (r"\ln(e^{x})", r"x"),
        (r"\frac{d}{dx} x^{3}", r"3x^{2}"),  # calculus is evaluated first
        (r"\int_{0}^{1} x \, dx", r"\frac{1}{2}"),
        (r"\sum_{k=1}^{n} k", r"\frac{n(n+1)}{2}"),
        (r"0.1 + 0.2", r"0.3"),  # exact, not floating point
    ],
)
def test_true_identities_are_correct(left: str, right: str) -> None:
    finding = check_identity(parse_expression(left), parse_expression(right))
    assert finding.verdict == Verdict.CORRECT
    assert finding.method == Method.SIMPLIFY


@pytest.mark.parametrize(
    ("left", "right"),
    [
        (r"(x + 1)^2", r"x^2 + 1"),
        (r"(x + y)^2", r"x^2 + y^2"),
        (r"\sqrt{x^2}", r"x"),  # false for negative x
        (r"\sin 2x", r"2 \sin x"),
        (r"\frac{1}{x} + \frac{1}{y}", r"\frac{1}{x + y}"),
    ],
)
def test_false_identities_come_with_a_counterexample(left: str, right: str) -> None:
    left_side, right_side = parse_expression(left), parse_expression(right)
    finding = check_identity(left_side, right_side)
    assert finding.verdict == Verdict.WRONG
    assert finding.witness is not None
    # The counterexample is exact and can be rechecked independently.
    values = {Symbol(name): sympify(value) for name, value in finding.witness.items()}
    assert simplify((left_side - right_side).subs(values.items())) != 0


@pytest.mark.parametrize(
    ("left", "right", "verdict"),
    [(r"3 \cdot 4", "12", Verdict.CORRECT), (r"3 \cdot 4", "13", Verdict.WRONG)],
)
def test_arithmetic(left: str, right: str, verdict: Verdict) -> None:
    finding = check_identity(parse_expression(left), parse_expression(right))
    assert finding.verdict == verdict
    assert finding.witness is None  # nothing to substitute


# ------------------------------------------------------------------- statements


def check(line: str, *earlier: str) -> Verdict:
    return check_statement(parse_statement(line), [parse_statement(e) for e in earlier]).verdict


@pytest.mark.parametrize(
    ("line", "earlier", "verdict"),
    [
        # one variable: compare solution sets
        (r"(x - 2)(x - 3) = 0", [r"x^2 - 5x + 6 = 0"], Verdict.CORRECT),
        (r"x = 2 \text{ or } x = 3", [r"x^2 - 5x + 6 = 0"], Verdict.CORRECT),
        (r"x = \pm 2", [r"x^2 = 4"], Verdict.CORRECT),
        (r"x = 2", [r"x^2 = 4"], Verdict.WARNING),  # loses x = -2
        (r"x^2 = 4", [r"x = 2"], Verdict.WARNING),  # adds x = -2
        (r"x = 2 \text{ or } x = 4", [r"x^2 - 5x + 6 = 0"], Verdict.WARNING),  # both
        (r"x = 5", [r"x^2 = 4"], Verdict.WRONG),
        (r"x = 2", [r"x^2 = 4", r"x > 0"], Verdict.CORRECT),  # x > 0 rules out -2
        (r"-2 < x, x < 2", [r"x^2 < 4"], Verdict.CORRECT),
        (r"x < 2", [r"x^2 < 4"], Verdict.WARNING),  # adds x <= -2
        (r"x > 5", [r"x^2 < 4"], Verdict.WRONG),
        # several variables: solve the earlier system, substitute
        (r"x = 2", [r"x + y = 3", r"x - y = 1"], Verdict.CORRECT),
        (r"y = 5", [r"x + y = 3", r"x - y = 1"], Verdict.WRONG),
        (r"y = 3 - x", [r"x + y = 3"], Verdict.CORRECT),
        (r"x = 2", [r"x + y = 3"], Verdict.WARNING),  # consistent, doesn't follow
        (r"x + y = 4", [r"x + y = 3"], Verdict.WRONG),
    ],
)
def test_conditions(line: str, earlier: list[str], verdict: Verdict) -> None:
    assert check(line, *earlier) == verdict


@pytest.mark.parametrize(
    ("line", "earlier", "reason"),
    [
        (r"x^2 - 5x + 6 = 0", [], "first condition"),
        (r"y = 2x", [r"x = 3"], "introduces y"),
        (r"t = 5", [r"x = 3"], "shares no symbol"),
    ],
)
def test_new_conditions_are_given(line: str, earlier: list[str], reason: str) -> None:
    finding = check_statement(parse_statement(line), [parse_statement(e) for e in earlier])
    assert finding.verdict == Verdict.GIVEN
    assert reason in finding.reason


def test_unrelated_earlier_conditions_are_ignored() -> None:
    # a > 0 has two variables' worth of trouble for the solver, but nothing to do with x.
    assert check(r"x = 2", r"a + b > 0", r"x^2 = 4", r"x > 0") == Verdict.CORRECT


@pytest.mark.parametrize(
    ("line", "verdict"),
    [
        (r"\sin^{2} x + \cos^{2} x = 1", Verdict.CORRECT),  # an identity, any earlier lines
        (r"3 \cdot 4 = 13", Verdict.WRONG),
        (r"2 < 5", Verdict.CORRECT),
        (r"5 \le 2", Verdict.WRONG),
    ],
)
def test_statements_without_unknowns_are_decided_directly(line: str, verdict: Verdict) -> None:
    finding = check_statement(parse_statement(line), [parse_statement(r"x = 1")])
    assert finding.verdict == verdict
    assert finding.method == Method.SIMPLIFY


def test_wrong_condition_names_an_earlier_solution_it_rejects() -> None:
    finding = check_statement(parse_statement(r"x = 5"), [parse_statement(r"x^2 = 4")])
    assert finding.witness in ({"x": "-2"}, {"x": "2"})


def test_inequalities_in_several_variables_are_unknown() -> None:
    assert check(r"x + y > 2", r"x - y > 0") == Verdict.UNKNOWN
