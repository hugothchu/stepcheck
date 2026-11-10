r"""Notation from physics, chemistry and more advanced maths, parsed line by line.

These pin down what the parser produces for each discipline's habits: subscripted
quantities (M_{1}, v_{\text{max}}), \Delta quantities, degrees, logarithm bases,
calculus operators. Lines it can't parse faithfully must be rejected, not guessed.
"""

# SymPy's type stubs say Eq() may return an Integer, so expected values built from it
# would all be flagged. That's a stub limitation, not a bug.
# pyright: reportArgumentType=false

import pytest
from sympy import (
    Basic,
    Derivative,
    E,
    Eq,
    Integral,
    Limit,
    Rational,
    Sum,
    Symbol,
    asin,
    cos,
    cot,
    csc,
    exp,
    log,
    oo,
    pi,
    sin,
    sqrt,
    symbols,
    tan,
)

from stepcheck.parse import LatexParseError, parse_expression, parse_line, parse_statement

a, b, g, k, m, n, r, s, t, u, v, x, y = symbols("a b g k m n r s t u v x y")


def sym(name: str) -> Symbol:
    """A symbol named the way the parser names it, e.g. sym("M_1") for M_{1}."""
    return Symbol(name)


# ------------------------------------------------------------------------ physics


@pytest.mark.parametrize(
    ("latex", "expected"),
    [
        (r"v = u + a t", Eq(v, u + a * t)),
        (r"s = u t + \frac{1}{2} a t^{2}", Eq(s, u * t + a * t**2 / 2)),
        (r"v^{2} = u^{2} + 2 a s", Eq(v**2, u**2 + 2 * a * s)),
        (r"M_{1} v_{1} = m_{2} v_{2}", Eq(sym("M_1") * sym("v_1"), sym("m_2") * sym("v_2"))),
        (
            r"F = G \frac{m_{1} m_{2}}{r^{2}}",
            Eq(sym("F"), sym("G") * sym("m_1") * sym("m_2") / r**2),
        ),
        (r"v_{\text{max}} = \omega_{0} A", Eq(sym(r"v_{\text{max}}"), sym("omega_0") * sym("A"))),
        (r"F_{net} = m a", Eq(sym("F_{net}"), m * a)),
        (r"E_{k} = \frac{1}{2} m v^{2}", Eq(sym("E_k"), m * v**2 / 2)),
        (r"E_{1} - E_{0} = h f", Eq(sym("E_1") - sym("E_0"), sym("h") * sym("f"))),
        (r"T = 2\pi\sqrt{\frac{\ell}{g}}", Eq(sym("T"), 2 * pi * sqrt(sym("ell") / g))),
        (r"E = \hbar \omega", Eq(sym("E"), sym("hbar") * sym("omega"))),
        (r"B = \mu_{0} n I", Eq(sym("B"), sym("mu_0") * n * sym("I"))),  # I is a current
        (r"\omega_{0} = \sqrt{\frac{k}{m}}", Eq(sym("omega_0"), sqrt(k / m))),
        (
            r"\bar{v} = \frac{\Delta x}{\Delta t}",
            Eq(sym("bar{v}"), sym("Delta_x") / sym("Delta_t")),
        ),
        (r"x_{i,j} = T_{1/2}", Eq(sym("x_{i,j}"), sym("T_{1/2}"))),
    ],
    ids=lambda case: case if isinstance(case, str) else "",
)
def test_physics(latex: str, expected: Basic) -> None:
    assert parse_statement(latex) == expected


@pytest.mark.parametrize(
    ("latex", "expected"),
    [
        (r"\Delta x = v \Delta t", Eq(sym("Delta_x"), v * sym("Delta_t"))),
        (r"\Delta \theta", sym(r"Delta_{\theta}")),
        (r"\Delta = b^{2} - 4ac", Eq(sym("Delta"), b**2 - 4 * a * sym("c"))),  # discriminant
    ],
)
def test_delta_is_part_of_the_quantity(latex: str, expected: Basic) -> None:
    assert parse_line(latex)[1] == expected


@pytest.mark.parametrize(
    "latex",
    [
        r"\vec{F} = m \vec{a}",
        r"v' = a",
    ],
    ids=["vectors", "prime"],
)
def test_unsupported_physics_is_rejected(latex: str) -> None:
    with pytest.raises(LatexParseError):
        parse_line(latex)


# ---------------------------------------------------------------------- chemistry


@pytest.mark.parametrize(
    ("latex", "expected"),
    [
        (r"P V = n R T", Eq(sym("P") * sym("V"), n * sym("R") * sym("T"))),
        (r"n = \frac{m}{M}", Eq(n, m / sym("M"))),
        (r"c_{1} V_{1} = c_{2} V_{2}", Eq(sym("c_1") * sym("V_1"), sym("c_2") * sym("V_2"))),
        (
            r"\Delta G = \Delta H - T \Delta S",
            Eq(sym("Delta_G"), sym("Delta_H") - sym("T") * sym("Delta_S")),
        ),
        # Square brackets are concentrations; as plain symbols the algebra is still right.
        (r"K_{c} = \frac{[C]^{2}}{[A][B]}", Eq(sym("K_c"), sym("C") ** 2 / (sym("A") * sym("B")))),
    ],
)
def test_chemistry_formulas(latex: str, expected: Basic) -> None:
    assert parse_statement(latex) == expected


@pytest.mark.parametrize(
    ("latex", "expected"),
    [
        (r"2 \times 1.008 + 16.00", Rational("18.016")),  # molar mass of water, exactly
        (r"6.022 \times 10^{23}", 602_200_000_000_000_000_000_000),
        (r"1.5 \times 10^{-3}", Rational(3, 2000)),
        (r"\frac{36.04}{18.02}", 2),
    ],
)
def test_chemistry_arithmetic_is_exact(latex: str, expected: Rational | int) -> None:
    assert parse_expression(latex) == expected


@pytest.mark.parametrize(
    "latex",
    [r"pH = -\log_{10}[H^+]", r"\mathrm{pH} = 7"],
    ids=["ion-concentration", "multi-letter-name"],
)
def test_unsupported_chemistry_is_rejected(latex: str) -> None:
    with pytest.raises(LatexParseError):
        parse_line(latex)


# --------------------------------------------------------------------- logarithms


@pytest.mark.parametrize(
    ("latex", "expected"),
    [
        (r"\log_{2} 8 = 3", Eq(3, 3, evaluate=False)),  # exact logs are evaluated
        (r"\ln(e^{2}) = 2", Eq(2, 2, evaluate=False)),
        (r"\log x + \log y = \log(xy)", Eq(log(x, 10) + log(y, 10), log(x * y, 10))),  # base 10
        (r"\ln x - \ln y = \ln \frac{x}{y}", Eq(log(x) - log(y), log(x / y))),
        (r"\log_{a} b = \frac{\ln b}{\ln a}", Eq(log(b, a), log(b) / log(a), evaluate=False)),
        (r"2^{x} = 8", Eq(2**x, 8)),
    ],
)
def test_logarithms(latex: str, expected: Basic) -> None:
    assert parse_statement(latex) == expected


# ------------------------------------------------------------------- trigonometry


@pytest.mark.parametrize(
    ("latex", "expected"),
    [
        (r"\tan x = \frac{\sin x}{\cos x}", Eq(tan(x), sin(x) / cos(x))),
        (r"\cot^{2} x + 1 = \csc^{2} x", Eq(cot(x) ** 2 + 1, csc(x) ** 2)),
        (r"\cos 2x = 1 - 2\sin^{2} x", Eq(cos(2 * x), 1 - 2 * sin(x) ** 2)),
        (
            r"\sin(a + b) = \sin a \cos b + \cos a \sin b",
            Eq(sin(a + b), sin(a) * cos(b) + cos(a) * sin(b)),
        ),
        (r"\sin^{-1} x = \arcsin x", Eq(asin(x), asin(x), evaluate=False)),
    ],
)
def test_trig_identities(latex: str, expected: Basic) -> None:
    assert parse_statement(latex) == expected


@pytest.mark.parametrize(
    ("latex", "expected"),
    [
        (r"\sin 30^{\circ}", Rational(1, 2)),  # not sin(30 radians)
        (r"\tan(45^{\circ})", 1),
        (r"\cos 60^\circ", Rational(1, 2)),
        (r"\sin 30 \degree", Rational(1, 2)),
        (r"x^{\circ} + 90^{\circ}", pi * x / 180 + pi / 2),
    ],
)
def test_degrees_become_radians(latex: str, expected: Basic) -> None:
    assert parse_expression(latex) == expected


# ----------------------------------------------------------------------- calculus


@pytest.mark.parametrize(
    ("latex", "expected"),
    [
        (r"\frac{d}{dx} \sin x", Derivative(sin(x), x)),
        (r"\frac{d}{dx}(x^{2} e^{x})", Derivative(x**2 * exp(x), x)),
        (r"\frac{\partial}{\partial x}(x^{2} y)", Derivative(x**2 * y, x)),
        (r"\int e^{x} \, dx", Integral(exp(x), x)),
        (r"\int_{0}^{\pi} \sin x \, dx", Integral(sin(x), (x, 0, pi))),
        (r"\sum_{k=1}^{n} k", Sum(k, (k, 1, n))),
        (r"\lim_{x \to 0} \frac{\sin x}{x}", Limit(sin(x) / x, x, 0)),
        (r"\lim_{x \to \infty} \frac{1}{x}", Limit(1 / x, x, oo, dir="-")),
        (r"e^{x}", exp(x)),
        (r"e", E),  # a lone e is Euler's number (clashes with the electron charge)
    ],
)
def test_calculus_stays_unevaluated(latex: str, expected: Basic) -> None:
    # Operators are kept as written, so a checker can evaluate them deliberately.
    assert parse_expression(latex) == expected


@pytest.mark.xfail(strict=True, reason="f is parsed as a constant symbol, so df/dx = 0")
def test_partial_derivative_of_a_named_function() -> None:
    assert parse_expression(r"\frac{\partial f}{\partial x}").doit() != 0
