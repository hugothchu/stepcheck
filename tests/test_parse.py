"""What the LaTeX layer turns into, including the parser's known traps."""

# SymPy's type stubs say Eq() may return an Integer, so Or(Eq(...), ...) in the
# expected values below would all be flagged. That's a stub limitation, not a bug.
# pyright: reportArgumentType=false

import pytest
from sympy import (
    And,
    Basic,
    Derivative,
    Eq,
    Integral,
    Ne,
    Or,
    Rational,
    Symbol,
    atan,
    cos,
    log,
    sin,
    symbols,
    sympify,
)

from stepcheck.parse import (
    LatexParseError,
    LineKind,
    parse_expression,
    parse_line,
    parse_statement,
    split_lines,
)

x, y, A, a, theta = symbols("x y A a theta")


def same(actual: Basic, expected: Basic) -> bool:
    """Structural equality after evaluating leftovers like `1*5*x` from the parser."""
    return sympify(str(actual)) == sympify(str(expected))


# ------------------------------------------------------------------- statements


@pytest.mark.parametrize(
    ("latex", "expected"),
    [
        (r"x^2 - 5x + 6 = 0", Eq(x**2 - 5 * x + 6, 0)),
        (r"(x-2)(x-3) = 0", Eq((x - 2) * (x - 3), 0)),
        (r"2x + 3y = 7", Eq(2 * x + 3 * y, 7)),
        (r"x \le 5", x <= 5),
        (r"x \neq 0", Ne(x, 0)),
        (r"|x - 1| = 3", Eq(abs(x - 1), 3)),
    ],
)
def test_single_relations(latex: str, expected: Basic) -> None:
    assert same(parse_statement(latex), expected)


@pytest.mark.parametrize(
    ("latex", "expected"),
    [
        (r"x = \pm 2", Or(Eq(x, -2), Eq(x, 2))),
        (r"x = 2 \text{ or } x = 3", Or(Eq(x, 2), Eq(x, 3))),
        (r"x = 2 \lor x = 3", Or(Eq(x, 2), Eq(x, 3))),
        (r"x^2 = 4, x > 0", And(Eq(x**2, 4), x > 0)),
        (r"x = 1 \text{ and } y = 2", And(Eq(x, 1), Eq(y, 2))),
        (r"\begin{cases} x + y = 3 \\ x - y = 1 \end{cases}", And(Eq(x + y, 3), Eq(x - y, 1))),
    ],
    ids=["plus-minus", "text-or", "lor", "comma-is-and", "text-and", "cases-is-a-system"],
)
def test_and_or_are_explicit(latex: str, expected: Basic) -> None:
    # The parser alone returns the same unordered set for "x = 2, x = 3" and
    # "x = 2 or x = 3", so "and" vs "or" must be decided before parsing.
    assert same(parse_statement(latex), expected)


def test_mixing_and_with_or_is_rejected() -> None:
    with pytest.raises(LatexParseError, match="ambiguous"):
        parse_statement(r"x = 1, y = 2 \text{ or } y = 3")


@pytest.mark.parametrize(
    "latex",
    [r"x = 1 \text{ or }", r"x = 1,", r"\lor x = 1", ""],
    ids=["trailing-or", "trailing-comma", "leading-or", "empty"],
)
def test_incomplete_statements_are_rejected(latex: str) -> None:
    # Dropping the empty side would turn `x = 1 or ...` into the claim `x = 1`.
    with pytest.raises(LatexParseError):
        parse_statement(latex)


@pytest.mark.parametrize(
    ("latex", "rhs"),
    [
        (r"3 \cdot 4 = 13", 13),
        (r"0.1 = 0.2", Rational(1, 5)),  # decimals become rationals without evaluating `=`
        (r"I = I", Symbol("I")),  # nor does replacing I
    ],
    ids=["arithmetic", "decimals", "capital-i"],
)
def test_relations_stay_unevaluated(latex: str, rhs: Basic) -> None:
    # A line must keep both sides, not collapse to `True` / `False`: a following
    # `= ...` line continues from its right-hand side.
    relation = parse_statement(latex)
    assert isinstance(relation, Eq)
    assert relation.rhs == rhs


# ------------------------------------------------------------------ trigonometry


@pytest.mark.parametrize(
    ("latex", "expected"),
    [
        (r"\sin^{2} x + \cos^{2} x = 1", Eq(sin(x) ** 2 + cos(x) ** 2, 1)),
        (
            r"\sin 2 \theta = 2 \sin \theta \cos \theta",
            Eq(sin(2 * theta), 2 * sin(theta) * cos(theta)),
        ),
        (
            r"\sin(2\theta) = 2\sin(\theta)\cos(\theta)",
            Eq(sin(2 * theta), 2 * sin(theta) * cos(theta)),
        ),
    ],
)
def test_trig_identities(latex: str, expected: Basic) -> None:
    assert same(parse_statement(latex), expected)


@pytest.mark.parametrize(
    ("latex", "expected"),
    [
        (r"\sin 2\theta", sin(2 * theta)),
        (r"\sin 2x \cos x", sin(2 * x) * cos(x)),
        (r"\tan^{-1} x", atan(x)),
    ],
)
def test_trig_without_brackets(latex: str, expected: Basic) -> None:
    assert same(parse_expression(latex), expected)


@pytest.mark.parametrize(
    ("latex", "expected"),
    [
        (r"\ln{\left(x \right)}^{2}", log(x) ** 2),  # how SymPy's printer writes (ln x)²
        (r"\sin{\left(x \right)}^{2}", sin(x) ** 2),
        (r"\sin{x}^{2}", sin(x**2)),  # braces around a bare argument: sin(x²), as written
        (r"\ln x^{2}", log(x**2)),
    ],
)
def test_power_after_a_function(latex: str, expected: Basic) -> None:
    assert parse_expression(latex) == expected


@pytest.mark.xfail(strict=True, reason=r"\cdot doesn't end a bracketless argument: sin(3x)")
def test_trig_argument_ends_at_cdot() -> None:
    assert same(parse_expression(r"2 \sin x \cdot 3"), 6 * sin(x))


# -------------------------------------------------------------------- expressions


def test_calculus() -> None:
    assert parse_expression(r"\frac{d}{dx} x^3") == Derivative(x**3, x)
    assert parse_expression(r"\int_0^1 x^2 \, dx") == Integral(x**2, (x, 0, 1))


@pytest.mark.parametrize(
    ("latex", "expected"),
    [
        (r"\boxed{42}", 42),
        (r"1{,}000 \times 3", 3000),
        (r"1,000 \times 3", 3000),
        (r"50\% \cdot 80", 40),
        (r"\$5 \times 3", 15),
        (r"3\frac{1}{2}", Rational(7, 2)),  # a mixed number, not 3 * 1/2
    ],
)
def test_word_problem_notation(latex: str, expected: int | Rational) -> None:
    assert parse_expression(latex).doit() == expected


def test_decimals_become_exact_rationals() -> None:
    assert parse_expression(r"0.1 + 0.2").doit() == Rational(3, 10)


def test_symbol_case_is_preserved() -> None:
    # The parser's default lowercases symbols, which would turn A + a into 2a.
    assert parse_expression(r"A + a") == A + a


# ----------------------------------------------------------- misparse protection


@pytest.mark.parametrize(
    "latex",
    [
        r"\vec{v} \cdot \vec{w}",
        r"1,5",
        r"a{,}b",
        r"x^2 = ",
        r"\frac{1}{",
        r"f'(x) = 2x",
        r"5 \text{ km} = 5000 \text{ m}",
        r"\frac{d^2 y}{dx^2}",
        r"x^2 \degree",
        r"x_1^\circ",
        r"(x + 1)^{\circ}",
        r"k = 1 k = 1 k",
        r"0 < x < 1",
    ],
    ids=[
        "vector-command",
        "stray-comma",
        "comma-between-letters",
        "dangling-equals",
        "unbalanced",
        "prime-would-vanish",
        "units-would-be-stripped",
        "dx-would-be-a-symbol",
        "degree-on-an-exponent",
        "degree-on-a-subscript",
        "degree-on-a-bracket",
        "chained-equals",
        "chained-inequality",
    ],
)
def test_untrustworthy_input_is_rejected(latex: str) -> None:
    with pytest.raises(LatexParseError):
        parse_line(latex)


def test_capital_i_is_a_symbol_not_the_imaginary_unit() -> None:
    assert parse_expression(r"\mu_{0} I") == Symbol("mu_0") * Symbol("I")


@pytest.mark.parametrize(
    ("latex", "expected"),
    [
        (r"$x = 5$", Eq(x, 5)),
        (r"\(x = 5\)", Eq(x, 5)),
        (r"5 \$ = 5", Eq(5, 5, evaluate=False)),  # an escaped \$ is currency, not math mode
        (r"\$1{,}200 + \$300", 1500),
    ],
)
def test_dollar_signs(latex: str, expected: Basic) -> None:
    assert parse_line(latex)[1] == expected


# ------------------------------------------------------------------ line handling


@pytest.mark.parametrize(
    ("latex", "kind"),
    [
        (r"(x + 1)^2", LineKind.EXPRESSION),
        (r"x^2 + 2x + 1 = 0", LineKind.STATEMENT),
        (r"= x^2 + 2x + 1", LineKind.CONTINUATION),
        (r"&= x^2 + 2x + 1", LineKind.CONTINUATION),
        (r"$x \le 5$", LineKind.STATEMENT),
    ],
)
def test_line_kinds(latex: str, kind: LineKind) -> None:
    assert parse_line(latex)[0] == kind


@pytest.mark.parametrize(
    ("latex", "kind"),
    [
        (r"\sum_{k=1}^{n} k", LineKind.EXPRESSION),
        (r"\prod_{i=1}^{n} i", LineKind.EXPRESSION),
        (r"\sum_{k=1}^{n} k^{2} + \prod_{i=1}^{3} i", LineKind.EXPRESSION),
        (r"\sum_{k=1}^{n} k = \frac{n(n+1)}{2}", LineKind.STATEMENT),
        (r"\prod_{i=1}^{3} i = 6", LineKind.STATEMENT),
        (r"\sum_{k=1}^{n} k^{2} \le n^{3}", LineKind.STATEMENT),
        (r"= \sum_{k=1}^{n} k", LineKind.CONTINUATION),
    ],
)
def test_operator_bounds_are_not_relations(latex: str, kind: LineKind) -> None:
    # The `=` in `\sum_{k=1}` is part of the bound, not a claim about the line.
    assert parse_line(latex)[0] == kind


def test_thin_space_is_not_an_and() -> None:
    # `\,` contains a comma, which must not split `x \, dx` into "x and dx".
    assert parse_statement(r"\int_0^1 x \, dx = \frac{1}{2}") == Eq(
        Integral(x, (x, 0, 1)), Rational(1, 2)
    )


def test_split_lines() -> None:
    text = r"""\begin{aligned}
    (x+1)^2 &= x^2 + 2x + 1 \\
    &= x(x + 2) + 1. \implies y = 2
    \end{aligned}"""
    assert split_lines(text) == ["(x+1)^2 = x^2 + 2x + 1", "= x(x + 2) + 1", "y = 2"]


def test_split_lines_keeps_cases_together() -> None:
    text = r"\begin{cases} x + y = 3 \\ x - y = 1 \end{cases} \\ x = 2"
    assert split_lines(text) == ["x + y = 3 , x - y = 1", "x = 2"]


def test_split_lines_keeps_cases_together_across_source_lines() -> None:
    text = "\\begin{cases}\n  x + y = 3 \\\\\n  x - y = 1\n\\end{cases}"
    assert len(split_lines(text)) == 1
    assert parse_statement(split_lines(text)[0]) == And(Eq(x + y, 3), Eq(x - y, 1))


@pytest.mark.parametrize(
    ("text", "lines"),
    [
        (r"\begin{aligned}{x+1}^2 &= y\end{aligned}", ["{x+1}^2 = y"]),  # braces are maths
        (r"\begin{array}{cc} x = 1 \end{array}", ["x = 1"]),  # {cc} is a column spec
    ],
)
def test_environment_markers_are_removed(text: str, lines: list[str]) -> None:
    assert split_lines(text) == lines
