"""Property-based tests: Hypothesis generates the cases, we state what must always hold.

The main property is a round trip. Build a random SymPy expression, render it with
SymPy's own LaTeX printer, parse it back, and it must be the same mathematics. The
same goes for sums and products, whose bounds (`\\sum_{k=1}^{n}`) contain an `=` that
must not make the line a statement. The rest check that the clean-up rules hold for
every number, not just the examples.
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
    Ge,
    Gt,
    I,
    Integer,
    Le,
    Lt,
    Mul,
    Or,
    Pow,
    Product,
    Rational,
    Sum,
    Symbol,
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
from sympy.core.relational import Relational

from stepcheck.models import Line
from stepcheck.parse import (
    LatexParseError,
    LineKind,
    parse_expression,
    parse_line,
    parse_statement,
    split_lines,
)

x, y, z, theta, m_1, v_0 = symbols("x y z theta m_1 v_0")
k, n = symbols("k n")  # the index and upper bound of sums and products
NESTED_INDICES = symbols("i j l")  # outermost to innermost
K, N = symbols("K N")

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

# A sum or product: \sum_{k=lower}^{upper} body, where the body uses the index k.
_bodies = (
    st.recursive(st.one_of(st.just(k), leaves), _combine, max_leaves=4)
    .filter(lambda body: body.has(k))
    .filter(_is_finite_and_real)
)
_lower_bounds = st.integers(-3, 3).map(Integer)
_upper_bounds = st.one_of(
    st.integers(4, 12).map(Integer),
    st.integers(0, 3).map(lambda offset: n + offset),
)
big_operators = st.builds(
    lambda operator, body, lower, upper: operator(body, (k, lower, upper)),
    st.sampled_from([Sum, Product]),
    _bodies,
    _lower_bounds,
    _upper_bounds,
)


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


@given(expressions, expressions)
def test_boxing_changes_nothing(left: Expr, right: Expr) -> None:
    # A boxed final answer means the same as the unboxed line, wherever the box is.
    left_latex, right_latex = latex(left, ln_notation=True), latex(right, ln_notation=True)
    plain = parse_line(f"{left_latex} = {right_latex}")
    assert parse_line(rf"\boxed{{{left_latex} = {right_latex}}}") == plain
    assert parse_line(rf"{left_latex} = \boxed{{{right_latex}}}") == plain


def _same_big_operator(parsed: Basic, expected: Sum | Product) -> bool:
    """Same operator and bounds, and a body with the same maths.

    Compared piece by piece: simplify() on a whole Sum may try to evaluate it.
    """
    return (
        type(parsed) is type(expected)
        and parsed.limits == expected.limits
        and _same_maths(parsed.function, expected.function)
    )


@given(big_operators)
def test_sums_and_products_round_trip_as_expressions(operator: Sum | Product) -> None:
    rendered = latex(operator, ln_notation=True)
    kind, content = parse_line(rendered)
    assert kind == LineKind.EXPRESSION, rendered  # the `=` in `_{k=1}` is a bound
    assert _same_big_operator(content, operator), rendered


@given(big_operators, expressions, st.sampled_from([Eq, Le, Lt, Ge, Gt]))
def test_relations_with_sums_and_products_are_statements(
    operator: Sum | Product, other_side: Expr, relation: type
) -> None:
    rendered = latex(relation(operator, other_side, evaluate=False), ln_notation=True)
    kind, content = parse_line(rendered)
    assert kind == LineKind.STATEMENT, rendered
    assert isinstance(content, Relational) and type(content) is relation, rendered
    assert _same_big_operator(content.lhs, operator), rendered
    assert _same_maths(content.rhs, other_side), rendered


# ------------------------------------------------------- nested sums and products
#
# SymPy prints a nested Sum with \substack, which the parser can't read (a known
# gap), so these LaTeX strings are written by hand: one \sum_{i=a}^{b} per level,
# with inner bounds that may depend on outer indices (\sum_{j=i}^{K}).

Level = tuple[type, Symbol, Expr, Expr]  # (Sum or Product, index, lower, upper)


@st.composite
def nested_levels(draw: st.DrawFn) -> list[Level]:
    depth = draw(st.integers(2, 3))
    levels: list[Level] = []
    for index in NESTED_INDICES[:depth]:
        outer = [level[1] for level in levels]
        lower = draw(st.one_of(st.integers(-2, 2).map(Integer), *map(st.just, outer)))
        upper = draw(
            st.one_of(
                st.integers(3, 8).map(Integer),
                st.sampled_from([K, N]),
                *(st.integers(1, 3).map(lambda c, o=o: o + c) for o in outer),
            )
        )
        levels.append((draw(st.sampled_from([Sum, Product])), index, lower, upper))
    return levels


def _nested_body(innermost: Symbol, indices: tuple[Symbol, ...]) -> st.SearchStrategy[Expr]:
    """A body that uses the innermost index (and possibly the outer ones)."""
    atoms = st.one_of(st.sampled_from(indices), leaves)
    return (
        st.recursive(atoms, _combine, max_leaves=4)
        .filter(lambda body: body.has(innermost))
        .filter(_is_finite_and_real)
    )


def _render(levels: list[Level], body_latex: str) -> str:
    operators = "".join(
        rf"\{'sum' if operator is Sum else 'prod'}_{{{index}={latex(lower)}}}^{{{latex(upper)}}} "
        for operator, index, lower, upper in levels
    )
    return operators + body_latex


def _build(levels: list[Level], body: Expr) -> Expr:
    for operator, index, lower, upper in reversed(levels):  # innermost first
        body = operator(body, (index, lower, upper))
    return body


def _flatten(expression: Basic) -> tuple[list[tuple[type, tuple]], Basic]:
    """Operators and ranges outermost first, plus the body: undoes SymPy's merging.

    Sum(Sum(f, (j, ...)), (i, ...)) and Sum(f, (j, ...), (i, ...)) both give
    [(Sum, (i, ...)), (Sum, (j, ...))] and f.
    """
    ranges: list[tuple[type, tuple]] = []
    while isinstance(expression, (Sum, Product)):
        ranges += [(type(expression), limit) for limit in reversed(expression.limits)]
        expression = expression.function
    return ranges, expression


@st.composite
def nested_operators(draw: st.DrawFn) -> tuple[list[Level], Expr]:
    levels = draw(nested_levels())
    indices = tuple(level[1] for level in levels)
    return levels, draw(_nested_body(indices[-1], indices))


@given(nested_operators())
def test_nested_sums_and_products_round_trip(case: tuple[list[Level], Expr]) -> None:
    levels, body = case
    rendered = _render(levels, rf"\left({latex(body, ln_notation=True)}\right)")
    kind, content = parse_line(rendered)
    assert kind == LineKind.EXPRESSION, rendered
    parsed_ranges, parsed_body = _flatten(content)
    expected_ranges, expected_body = _flatten(_build(levels, body))
    assert parsed_ranges == expected_ranges, rendered
    assert isinstance(expected_body, Expr)
    assert _same_maths(parsed_body, expected_body), rendered


@st.composite
def ambiguous_nested_operators(draw: st.DrawFn) -> tuple[list[Level], Expr]:
    """A body that's a sum of terms, each using the innermost index: k + k x + ..."""
    levels = draw(nested_levels())
    indices = tuple(level[1] for level in levels)
    innermost = indices[-1]
    factors = st.one_of(st.sampled_from(indices), leaves)
    terms = st.lists(factors.map(lambda factor: factor * innermost), min_size=2, max_size=3)
    return levels, Add(*draw(terms))


@given(ambiguous_nested_operators())
def test_unbracketed_nested_bodies_are_never_misread(case: tuple[list[Level], Expr]) -> None:
    # Without brackets the parser may end the body early. If it does, a term with
    # the innermost index lands outside its sum, so the line must be rejected; if it
    # doesn't, the parse must be the intended one. Never a different reading.
    levels, body = case
    rendered = _render(levels, latex(body))
    try:
        content = parse_expression(rendered)
    except LatexParseError:
        return
    parsed_ranges, parsed_body = _flatten(content)
    expected_ranges, expected_body = _flatten(_build(levels, body))
    assert parsed_ranges == expected_ranges, rendered
    assert isinstance(expected_body, Expr)
    assert _same_maths(parsed_body, expected_body), rendered


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
    "\\{", "\\}", "\\sum_{", "\\prod_{", "}^{", "k=1", "k", "n", "\\,", "dx", "\\int",
]
# fmt: on


@settings(max_examples=2 * settings().max_examples)
@example("k=1k=1k")  # found by Hypothesis: a chained relation crashed SymPy's And()
@given(st.lists(st.sampled_from(_FRAGMENTS), min_size=1, max_size=12).map("".join))
def test_any_text_becomes_a_line_without_crashing(text: str) -> None:
    # The whole point of UNPARSED: garbage in, a recorded reason out, never an exception.
    line = Line.from_latex(text)
    assert (line.kind == LineKind.UNPARSED) == (line.parse_error is not None)
