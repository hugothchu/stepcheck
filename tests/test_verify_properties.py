"""Property-based tests for the checkers: their verdicts must be *sound*.

A checker may say "unknown" whenever it likes, but it must never call a true claim
wrong or a false claim correct. Hypothesis builds claims whose truth we know by
construction (a rewrite SymPy guarantees, roots we picked, a system with a known
solution) and checks the verdict against it.
"""

# SymPy's type stubs are loose (Eq() may "return" an Integer, Add/Mul take Basic).
# pyright: reportArgumentType=false

from collections.abc import Callable
from fractions import Fraction

from hypothesis import given, reject
from hypothesis import strategies as st
from strategies import expressions, x, y
from sympy import (
    Eq,
    Expr,
    Mul,
    Or,
    Rational,
    Symbol,
    cancel,
    expand,
    factor,
    simplify,
    sympify,
    together,
)

from stepcheck.models import Finding, Verdict
from stepcheck.timeout import CheckTimeout, run_with_timeout
from stepcheck.verify import check_identity, check_statement

TIME_LIMIT = 20.0  # seconds; a timeout is "unknown", so it is skipped, not failed


def run(check: Callable[..., Finding], *args: object) -> Finding:
    try:
        return run_with_timeout(check, *args, seconds=TIME_LIMIT)
    except CheckTimeout:
        reject()  # too slow to tell: try another example


# --------------------------------------------------------------- equivalence


rewrites = st.sampled_from([expand, factor, cancel, together])


@given(expressions, rewrites)
def test_a_true_identity_is_never_wrong(expression: Expr, rewrite: Callable[[Expr], Expr]) -> None:
    # expand(e) = e (and so on) holds wherever both sides are defined.
    finding = run(check_identity, expression, rewrite(expression))
    assert finding.verdict != Verdict.WRONG, (expression, finding.reason)


nonzero = st.fractions(min_value=-5, max_value=5, max_denominator=6).filter(bool)


@given(expressions, nonzero, st.sampled_from([1, x, x**2, y]))
def test_a_false_identity_is_never_correct(expression: Expr, size: Fraction, shape: Expr) -> None:
    # expand(e) and e + c·shape differ by c·shape, which isn't 0 for every x and y.
    left, right = (
        expand(expression),
        expression + Rational(size.numerator, size.denominator) * shape,
    )
    finding = run(check_identity, left, right)
    assert finding.verdict != Verdict.CORRECT, (expression, finding.reason)
    if finding.witness:
        # The counterexample is genuine: at those values the two sides differ.
        values = {
            Symbol(name, real=True): sympify(value) for name, value in finding.witness.items()
        }
        assert simplify(_as_real(left - right).subs(values.items())) != 0


def _as_real(expression: Expr) -> Expr:
    """Symbols declared real, as the checker does, so substituting values matches."""
    return expression.subs([(s, Symbol(str(s), real=True)) for s in expression.free_symbols])


# --------------------------------------------------------------- consistency


roots = st.sets(st.integers(-9, 9), min_size=1, max_size=3)


@given(roots, roots)
def test_claimed_roots_against_the_true_roots(true_roots: set[int], claimed: set[int]) -> None:
    # Given (x - r1)(x - r2)... = 0, expanded; the line claims x = c1 or x = c2 ...
    given_line = Eq(expand(Mul(*(x - root for root in true_roots))), 0)
    claim = Or(*(Eq(x, value) for value in sorted(claimed)))
    finding = run(check_statement, claim, [given_line])
    if claimed == true_roots:
        expected = Verdict.CORRECT
    elif claimed.isdisjoint(true_roots):
        expected = Verdict.WRONG
    else:
        expected = Verdict.WARNING  # loses or adds roots
    assert finding.verdict == expected, finding.reason


small = st.integers(-9, 9)


@given(small, small, st.tuples(small, small, small, small), small.filter(bool))
def test_a_system_with_a_known_solution(
    a: int, b: int, matrix: tuple[int, int, int, int], offset: int
) -> None:
    p, q, r, s = matrix
    if p * s - q * r == 0:
        reject()  # no unique solution
    system = [Eq(p * x + q * y, p * a + q * b), Eq(r * x + s * y, r * a + s * b)]
    assert run(check_statement, Eq(x, a), system).verdict == Verdict.CORRECT
    assert run(check_statement, Eq(y, b + offset), system).verdict == Verdict.WRONG
