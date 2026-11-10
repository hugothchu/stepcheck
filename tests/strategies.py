"""Hypothesis strategies shared by the property tests: random SymPy expressions."""

from fractions import Fraction

from hypothesis import strategies as st
from sympy import (
    Add,
    Basic,
    Expr,
    I,
    Integer,
    Mul,
    Pow,
    Rational,
    cos,
    exp,
    log,
    nan,
    oo,
    sin,
    sqrt,
    symbols,
    tan,
    zoo,
)

x, y, z, theta, m_1, v_0 = symbols("x y z theta m_1 v_0")


def _rational(fraction: Fraction) -> Expr:
    return Rational(fraction.numerator, fraction.denominator)


leaves = st.one_of(
    st.sampled_from([x, y, z, theta, m_1, v_0]),
    st.integers(-12, 12).map(Integer),
    st.fractions(min_value=-5, max_value=5, max_denominator=9).map(_rational),
)


def combine(children: st.SearchStrategy[Expr]) -> st.SearchStrategy[Expr]:
    return st.one_of(
        st.tuples(children, children).map(lambda pair: Add(*pair)),
        st.tuples(children, children).map(lambda pair: Mul(*pair)),
        st.tuples(children, st.integers(-3, 4)).map(lambda pair: Pow(pair[0], pair[1])),
        children.map(sin),
        children.map(cos),
        children.map(tan),
        children.map(exp),
        children.map(log),
        children.map(sqrt),
    )


def is_finite_and_real(expression: Basic) -> bool:
    """Drop 1/0, log(0), sqrt(-4) = 2i and friends.

    The imaginary unit is a known gap (`i` parses as a symbol), not what's under test.
    """
    return not expression.has(zoo, oo, -oo, nan, I) and expression.is_real is not False


expressions = st.recursive(leaves, combine, max_leaves=6).filter(is_finite_and_real)
