"""Property-based tests: Hypothesis generates the cases, we state what must always hold.

The main property is a round trip. Build a random SymPy expression, render it with
SymPy's own LaTeX printer, parse it back, and it must be the same mathematics. The
rest check that the clean-up rules hold for every number, not just the examples.
"""

# SymPy's type stubs are loose (Eq() may "return" an Integer, Add/Mul take Basic).
# pyright: reportArgumentType=false

from decimal import Decimal
from fractions import Fraction

from hypothesis import example, given, settings
from hypothesis import strategies as st
from sympy import (
    Add,
    Basic,
    Eq,
    Expr,
    I,
    Integer,
    Mul,
    Or,
    Pow,
    Rational,
    cos,
    exp,
    latex,
    log,
    nan,
    oo,
    pi,
    simplify,
    sin,
    sqrt,
    symbols,
    tan,
    zoo,
)

from stepcheck.models import Line
from stepcheck.parse import LineKind, parse_expression, parse_statement, split_lines

x, y, z, theta, m_1, v_0 = symbols("x y z theta m_1 v_0")

# ----------------------------------------------------------------------- strategies


def _rational(fraction: Fraction) -> Expr:
    return Rational(fraction.numerator, fraction.denominator)


leaves = st.one_of(
    st.sampled_from([x, y, z, theta, m_1, v_0]),
    st.integers(-12, 12).map(Integer),
    st.fractions(min_value=-5, max_value=5, max_denominator=9).map(_rational),
)


def _combine(children: st.SearchStrategy[Expr]) -> st.SearchStrategy[Expr]:
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


def _is_finite_and_real(expression: Basic) -> bool:
    """Drop 1/0, log(0), sqrt(-4) = 2i and friends.

    The imaginary unit is a known gap (`i` parses as a symbol), not what's under test.
    """
    return not expression.has(zoo, oo, -oo, nan, I) and expression.is_real is not False


expressions = st.recursive(leaves, _combine, max_leaves=6).filter(_is_finite_and_real)


def _same_maths(parsed: Basic, expected: Expr) -> bool:
    # Cheap structural check first; simplify only when the printer reordered things.
    return parsed == expected or simplify(expected - parsed) == 0


# ------------------------------------------------------------------- round trips


@given(expressions)
def test_sympy_latex_round_trips(expression: Expr) -> None:
    # ln_notation: SymPy prints natural log as \log, which LaTeX readers take as log_10.
    rendered = latex(expression, ln_notation=True)
    assert _same_maths(parse_expression(rendered), expression), rendered


@given(expressions, expressions)
@example(x, log(log(x) ** 2))  # found by Hypothesis: \ln{\left(x\right)}^{2} read as ln(x²)
def test_equations_round_trip(left: Expr, right: Expr) -> None:
    rendered = latex(Eq(left, right, evaluate=False), ln_notation=True)
    parsed = parse_statement(rendered)
    assert isinstance(parsed, Eq), rendered
    assert _same_maths(parsed.lhs, left), rendered
    assert _same_maths(parsed.rhs, right), rendered


# ------------------------------------------------------------- clean-up rules


@given(st.integers(min_value=1_000, max_value=10**15))
def test_thousands_separators(number: int) -> None:
    with_commas = f"{number:,}"  # 1,234,567
    assert parse_expression(with_commas) == number
    assert parse_expression(with_commas.replace(",", "{,}")) == number


@given(st.decimals(min_value=-(10**6), max_value=10**6, places=4, allow_nan=False))
def test_decimals_are_exact(number: Decimal) -> None:
    written = f"{number:f}"
    assert parse_expression(written) == Rational(written)


@given(st.integers(1, 1000))
def test_plus_minus_gives_both_roots(root: int) -> None:
    assert parse_statement(rf"x = \pm {root}") == Or(Eq(x, -root), Eq(x, root))


@given(st.integers(0, 360))
def test_degrees_are_converted_to_radians(angle: int) -> None:
    assert parse_expression(rf"{angle}^{{\circ}}") == Rational(angle, 180) * pi


# --------------------------------------------------------------- structure


statement_lines = st.lists(
    st.builds("{} = {}".format, st.sampled_from("xyz"), st.integers(-50, 50)),
    min_size=1,
    max_size=8,
)


@given(statement_lines, st.sampled_from([r" \\ ", "\n", r" \implies "]))
def test_split_lines_keeps_every_line(lines: list[str], separator: str) -> None:
    assert split_lines(separator.join(lines)) == lines


# Fragments that are common in solutions, plus some that are common in broken ones.
# fmt: off
_FRAGMENTS = [
    "x", "y", "2", "0.5", "=", "+", "-", "^", "_", "{", "}", "(", ")", ",", "\\\\", "&",
    "$", "\\$", "\\frac", "\\sqrt", "\\sin", "\\pm", "\\text{or}", "\\cdot", "\\le",
    "\\Delta", "^{\\circ}", "\\%", "\\boxed", "\\begin{cases}", "\\end{cases}", "'", " ",
]
# fmt: on


@settings(max_examples=2 * settings().max_examples)
@given(st.lists(st.sampled_from(_FRAGMENTS), min_size=1, max_size=12).map("".join))
def test_any_text_becomes_a_line_without_crashing(text: str) -> None:
    # The whole point of UNPARSED: garbage in, a recorded reason out, never an exception.
    line = Line.from_latex(text)
    assert (line.kind == LineKind.UNPARSED) == (line.parse_error is not None)
