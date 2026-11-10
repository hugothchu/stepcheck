"""Turn LaTeX into SymPy objects.

The heavy lifting is done by latex2sympy2_extended. This module wraps it with the
pre- and post-processing that step-by-step checking needs:

* clean-up of notation the parser mishandles (thousands separators, percentages,
  degrees, ``\\Delta x``, alignment markers, ``cases`` blocks);
* explicit "and" / "or" between relations, because the parser turns both
  ``x = 2, x = 3`` and ``x = 2 \\text{ or } x = 3`` into the same unordered set;
* ``\\pm`` expanded into a disjunction;
* decimals converted to exact rationals, so the solver works exactly;
* a guard against *silent* misparses (``\\boxed`` becoming the symbol ``boxed``,
  ``I`` becoming the imaginary unit, the prime in ``f'(x)`` vanishing).

Public functions:

* :func:`split_lines` splits a multi-line solution into one string per line;
* :func:`parse_expression` parses a single expression, e.g. ``(x+1)^2``;
* :func:`parse_statement` parses one or more relations, e.g. ``x^2 = 4, x > 0``;
* :func:`parse_line` detects which of those a solution line is.
"""

import re
from enum import StrEnum
from string import ascii_letters

from latex2sympy2_extended import NormalizationConfig, latex2sympy
from latex2sympy2_extended.latex2sympy2 import ConversionConfig
from sympy import And, Basic, Eq, Expr, FiniteSet, Float, I, Or, Product, Rational, Sum, Symbol
from sympy.core.relational import Relational
from sympy.logic.boolalg import Boolean, BooleanAtom


class LatexParseError(ValueError):
    """The LaTeX could not be turned into a trustworthy SymPy object."""


class LineKind(StrEnum):
    EXPRESSION = "expression"  # (x + 1)^2
    STATEMENT = "statement"  # x^2 - 5x + 6 = 0      or   x = 2 \text{ or } x = 3
    CONTINUATION = "continuation"  # = x^2 + 2x + 1   (continues the previous line)
    UNPARSED = "unparsed"  # could not be parsed; checkers report it as "unknown"


# --------------------------------------------------------------------------- config

_CONVERSION = ConversionConfig(
    lowercase_symbols=False,  # the default turns `A + a` into `2a`
    interpret_as_mixed_fractions=True,  # 3\frac{1}{2} = 7/2
)
_NORMALIZATION = NormalizationConfig(
    basic_latex=True,
    # Never strip units: the parser's list is a dataset heuristic that turns
    # `5 \text{ km} = 5000 \text{ m}` into `5` and `s = 5 m` into `s = 5`.
    units=False,
    malformed_operators=False,
    nits=False,
    boxed="all",
    equations=False,
)

_GREEK_NAMES = (
    "alpha beta gamma delta epsilon varepsilon zeta eta theta vartheta iota kappa lambda mu "
    "nu xi pi rho varrho sigma varsigma tau upsilon phi varphi chi psi omega "
    "Gamma Delta Theta Lambda Xi Pi Sigma Upsilon Phi Psi Omega "
    "ell hbar"  # not Greek, but single-letter physics symbols
)
GREEK_LETTERS = frozenset(_GREEK_NAMES.split())

# How the parser names symbols: base letter, optional accent, optional subscript.
#   m_1   v_{\text{max}}   x_{i,j}   omega_0   bar{x}   hat{n}_1   Delta_x
_SUBSCRIPT = re.compile(r"_(\w+|\{.*\})$")
_ACCENT = re.compile(r"^(bar|hat|tilde|dot|ddot)\{(?P<base>.+)\}$")

# `30^{\circ}` must become π/6: the parser reads `\sin 30^{\circ}` as sin(30 radians).
# The angle must be a whole atom: in `x^2\degree` the 2 is an exponent, so that line is
# left alone, and `_convert` rejects the leftover degree sign (the parser would drop it).
_DEGREES = re.compile(
    r"(?<![\^_\d.])(?P<angle>\d+(?:\.\d+)?|[A-Za-z]|\\[A-Za-z]+)"
    r"\s*(?:\^\s*\{?\s*\\circ\s*\}?|\\degree\b)"
)

# The grammar reserves E and e for Euler's number, so `E_{k}` is a syntax error.
_SUBSCRIPTED_E = re.compile(r"(?<!\\mathit\{)(?<![\\A-Za-z])(?P<letter>[Ee])_")

# SymPy prints log(x)**2 as `\ln{\left(x \right)}^{2}`, which the parser reads as ln(x²).
_FUNCTION_THEN_BRACE = re.compile(r"\\(?:(?:arc)?(?:sin|cos|tan|sec|csc|cot)h?|ln|log|exp)\s*\{")

# `\Delta x` is one quantity (a change in x), not Delta times x.
_DELTA_QUANTITY = re.compile(r"\\Delta\s*(?P<quantity>[A-Za-z](?![A-Za-z])|\\[A-Za-z]+\b)")

_OR_SEPARATOR = re.compile(r"\\text\{\s*or\s*\}|\\lor\b|\\vee\b")
_AND_SEPARATOR = re.compile(r",|\\text\{\s*and\s*\}|\\land\b|\\wedge\b")
_LINE_BREAK = re.compile(r"\\\\|\n|\\implies\b|\\Rightarrow\b|\\iff\b|\\Leftrightarrow\b")
_CASES = re.compile(r"\\begin\{cases\}(?P<body>.*?)\\end\{cases\}", re.DOTALL)
_ALIGN_ENVIRONMENT = re.compile(
    r"\\(?:begin|end)\{(?:aligned|align\*?|gathered|split)\}"
    r"|\\begin\{array\}\{[^}]*\}|\\end\{array\}"  # only array has a column spec: {cc}
)
# Spacing commands. `\,` must go before splitting on commas: `x \, dx` is not "x and dx".
_SPACING = re.compile(r"(?<!\\)\\[,;:! ]|\\q?quad\b")


# ------------------------------------------------------------------ public functions


def split_lines(text: str) -> list[str]:
    """Split a solution into lines: on ``\\\\``, newlines, ``\\implies`` and similar."""
    text = _flatten_cases(text)
    text = _ALIGN_ENVIRONMENT.sub("", text)
    pieces = (_strip_delimiters(piece) for piece in _LINE_BREAK.split(text))
    return [piece for piece in pieces if piece]


def parse_expression(latex: str) -> Expr:
    """Parse a single expression such as ``\\frac{d}{dx} x^3`` or ``(x+1)^2``."""
    result = _convert(_clean(latex))
    if not isinstance(result, Expr):
        raise LatexParseError(f"expected an expression, got {type(result).__name__}: {latex!r}")
    return result


def parse_statement(latex: str) -> Boolean:
    """Parse relations joined by "and" (``,``) or "or" (``\\text{or}``).

    ``x^2 = 4, x > 0``          → And(Eq(x**2, 4), x > 0)
    ``x = 2 \\text{ or } x = 3`` → Or(Eq(x, 2), Eq(x, 3))
    ``x = \\pm 2``               → Or(Eq(x, -2), Eq(x, 2))
    """
    cleaned = _clean(latex)
    or_parts = _split_top_level(cleaned, _OR_SEPARATOR)
    and_parts = _split_top_level(cleaned, _AND_SEPARATOR)
    if "" in or_parts or "" in and_parts:
        raise LatexParseError(f"incomplete statement, an 'and'/'or' has a missing side: {latex!r}")
    if len(or_parts) > 1 and len(and_parts) > 1:
        raise LatexParseError(f"mixing 'and' and 'or' in one line is ambiguous: {latex!r}")

    if len(or_parts) > 1:
        return Or(*(_parse_relation(part) for part in or_parts))
    return And(*(_parse_relation(part) for part in and_parts))


def parse_line(latex: str) -> tuple[LineKind, Basic]:
    """Parse one solution line and say what kind of line it is."""
    cleaned = _clean(latex)
    if cleaned.startswith("="):
        return LineKind.CONTINUATION, parse_expression(cleaned.removeprefix("="))
    if _has_relation(cleaned):
        return LineKind.STATEMENT, parse_statement(cleaned)
    return LineKind.EXPRESSION, parse_expression(cleaned)


# ------------------------------------------------------------------------- cleaning


def _clean(latex: str) -> str:
    latex = _flatten_cases(latex)
    latex = _strip_delimiters(latex)
    latex = re.sub(r"(?<=\d)\{,\}(?=\d{3}(?!\d))", "", latex)  # 1{,}000 → 1000, not a{,}b
    if r"\{" not in latex:  # outside set notation, 1,000 is a thousands separator
        latex = re.sub(r"(?<=\d),(?=\d{3}(?!\d))", "", latex)
    latex = re.sub(r"(\d+(?:\.\d+)?)\s*\\%", r"\\frac{\1}{100}", latex)  # 50\% → 1/2
    latex = _SPACING.sub(" ", latex)
    latex = _DEGREES.sub(r"(\g<angle> \\cdot \\frac{\\pi}{180})", latex)  # 30^{\circ} → π/6
    latex = _unwrap_bracketed_arguments(latex)
    latex = _DELTA_QUANTITY.sub(_delta_as_subscript, latex)
    latex = _SUBSCRIPTED_E.sub(r"\\mathit{\g<letter>}_", latex)  # E_{k} → \mathit{E}_{k}
    return latex.strip()


def _unwrap_bracketed_arguments(latex: str) -> str:
    r"""``\ln{\left(x \right)}^{2}`` → ``\ln\left(x \right)^{2}``, which parses as (ln x)².

    Only braces holding exactly one bracketed group are removed: ``\sin{x}^{2}`` is
    sin(x²) as written, and stays that way.
    """
    position = 0
    while match := _FUNCTION_THEN_BRACE.search(latex, position):
        open_brace = match.end() - 1
        close_brace = _closing_index(latex, open_brace, "{", "}")
        if close_brace is None:
            break
        inner = latex[open_brace + 1 : close_brace].strip()
        if _is_one_bracketed_group(inner):
            latex = latex[:open_brace] + inner + latex[close_brace + 1 :]
        position = open_brace + 1
    return latex


def _is_one_bracketed_group(text: str) -> bool:
    """``(x)`` and ``\\left(x + 1\\right)`` are; ``x``, ``(x)(y)`` and ``(x) + 1`` aren't."""
    if not text.startswith(("(", r"\left(")):
        return False
    return _closing_index(text, text.index("("), "(", ")") == len(text) - 1


def _closing_index(text: str, start: int, opening: str, closing: str) -> int | None:
    """Index of the bracket that closes the one at `start`, or None if it's unbalanced."""
    depth = 0
    for index in range(start, len(text)):
        if text[index] == opening:
            depth += 1
        elif text[index] == closing:
            depth -= 1
            if depth == 0:
                return index
    return None


def _delta_as_subscript(match: re.Match[str]) -> str:
    r"""``\Delta x`` → ``\Delta_{x}`` and ``\Delta \theta`` → ``\Delta_{\theta}``."""
    quantity = match["quantity"]
    is_letter = len(quantity) == 1 or quantity.removeprefix("\\") in GREEK_LETTERS
    return rf"\Delta_{{{quantity}}}" if is_letter else match[0]


def _strip_delimiters(latex: str) -> str:
    """Remove math-mode delimiters, alignment markers and trailing punctuation."""
    latex = re.sub(r"(?<!\\)\$", "", latex)  # math-mode $...$, but keep currency \$
    latex = re.sub(r"^\\[\[(]|\\[\])]$", "", latex)
    latex = re.sub(r"^\\therefore\b", "", latex)
    latex = latex.replace("&", "")
    return latex.strip().rstrip(".").strip()


def _flatten_cases(latex: str) -> str:
    r"""``\begin{cases} a \\ b \end{cases}`` → ``a, b`` (a system: all must hold)."""
    return _CASES.sub(lambda m: m["body"].replace(r"\\", ",").replace("\n", " "), latex)


def _split_top_level(text: str, separator: re.Pattern[str]) -> list[str]:
    """Split on `separator`, ignoring matches inside brackets: f(1,2), \\{1,2\\}, {...}.

    Empty parts are kept (`x = 1 \\text{or}` → ["x = 1", ""]), so an incomplete
    statement is rejected rather than quietly shortened.
    """
    parts: list[str] = []
    depth = start = position = 0
    while position < len(text):
        char = text[position]
        if char in "({[":
            depth += 1
        elif char in ")}]":
            depth -= 1
        elif depth == 0 and (match := separator.match(text, position)):
            parts.append(text[start:position])
            start = position = match.end()
            continue
        position += 1
    parts.append(text[start:])
    return [part.strip() for part in parts]


def _has_relation(latex: str) -> bool:
    r"""Is there a relation outside braces? ``\sum_{k=1}^{n} k`` is an expression."""
    return bool(re.search(r"=|<|>|\\[lg]eq?\b|\\neq?\b", _outside_braces(latex)))


def _outside_braces(latex: str) -> str:
    r"""``\sum_{k=1}^{n} k = 3`` → ``\sum_^ k = 3``: drop everything inside ``{...}``."""
    kept: list[str] = []
    depth = 0
    for char in latex:
        if char == "{":
            depth += 1
        elif char == "}":
            depth = max(depth - 1, 0)
        elif depth == 0:
            kept.append(char)
    return "".join(kept)


# ----------------------------------------------------------------------- conversion


def _parse_relation(latex: str) -> Boolean:
    result = _convert(latex)
    result = _expand_plus_minus(result)
    if not isinstance(result, (Relational, Or, BooleanAtom)):
        raise LatexParseError(f"expected a relation such as `a = b`, got: {latex!r}")
    return result


def _convert(latex: str) -> Basic:
    """Call the parser, then reject results that are likely misparses."""
    if not latex:
        raise LatexParseError("empty LaTeX")
    if "'" in latex:  # the parser silently drops primes: f'(x) → f(x)
        raise LatexParseError(f"primes are not supported yet: {latex!r}")
    if re.search(r"\\circ\b|\\degree\b", latex):  # left by _DEGREES: x_1^\circ → x_1
        raise LatexParseError(f"degrees on a compound angle are not supported: {latex!r}")
    try:
        result = latex2sympy(
            latex, normalization_config=_NORMALIZATION, conversion_config=_CONVERSION
        )
    except Exception as error:  # the parser raises bare Exceptions
        raise LatexParseError(f"cannot parse {latex!r}: {str(error).splitlines()[0]}") from error

    if not isinstance(result, Basic):  # the parser can also return a list or None
        raise LatexParseError(f"unexpected parser output {type(result).__name__}: {latex!r}")
    if isinstance(result, (And, Or)):  # `a = b = c` arrives as And(Eq(a, b), Eq(b, c))
        raise LatexParseError(f"chained relations are not supported yet: {latex!r}")
    result = _rebuild(result)
    _reject_suspicious_symbols(result, latex)
    _reject_escaped_indices(result, latex)
    if result.has(FiniteSet) and r"\{" not in latex and r"\pm" not in latex:
        raise LatexParseError(f"parsed as a set, probably a stray comma: {latex!r}")
    return result


def _reject_suspicious_symbols(result: Basic, latex: str) -> None:
    r"""Every symbol must be a single letter, plus an optional accent and subscript.

    Allowed: ``x``, ``M_{1}``, ``v_{\text{max}}``, ``\omega_0``, ``\bar{x}``, ``\Delta x``.
    Rejected: misread commands (``boxed``, ``vec{v}``) and multi-letter names, which
    are usually units (``\text{km}``) or a misparse (``dx`` in ``\frac{d^2y}{dx^2}``).
    """
    for name in map(str, result.free_symbols):
        if not _is_symbol_name(name):
            raise LatexParseError(f"suspicious symbol {name!r} in {latex!r}")


def _reject_escaped_indices(result: Basic, latex: str) -> None:
    r"""A sum or product index must not also appear outside that sum or product.

    Without brackets the body ends early: ``\sum_{j=1}^{K} i - j`` parses as
    ``Sum(i, (j, 1, K)) - j``, leaving a meaningless ``j`` outside. Brackets fix it:
    ``\sum_{j=1}^{K} (i - j)``. (This also rejects reusing a name on purpose, as in
    ``j + \sum_{j=1}^{n} j``; that's rare, and it becomes "unknown", not "wrong".)
    """
    for operator in result.atoms(Sum, Product):
        escaped = set(operator.variables) & result.free_symbols
        if escaped:
            names = ", ".join(sorted(map(str, escaped)))
            raise LatexParseError(
                f"index {names} is used outside its sum or product; "
                f"put the body in brackets: {latex!r}"
            )


def _is_symbol_name(name: str) -> bool:
    name = _SUBSCRIPT.sub("", name)
    if accent := _ACCENT.match(name):
        name = accent["base"]
    return (len(name) == 1 and name in ascii_letters) or name in GREEK_LETTERS


def _expand_plus_minus(result: Basic) -> Basic:
    r"""``Eq(x, {-2, 2})`` (from ``x = \pm 2``) → ``Or(Eq(x, -2), Eq(x, 2))``."""
    if isinstance(result, Eq) and isinstance(result.rhs, FiniteSet):
        equations = [Eq(result.lhs, value, evaluate=False) for value in result.rhs]
        return Or(*equations)  # pyright: ignore[reportArgumentType]  (SymPy's stubs type Eq() as a union)
    return result


def _rebuild(result: Basic) -> Basic:
    """Rebuild the parser's tree bottom-up, fixing atoms and evaluating arithmetic.

    * floats become exact rationals: 0.5 → 1/2, so solving stays exact;
    * `I` becomes a plain symbol: it's a current or an integral, not √-1;
    * arithmetic is evaluated as usual: 1*5*x → 5*x.

    Relations are rebuilt *unevaluated*, so a wrong line such as 3·4 = 13 or
    0.1 = 0.2 keeps both sides instead of collapsing to `False`. Derivatives and
    integrals stay unevaluated too (rebuilding them doesn't compute them).
    """
    if isinstance(result, Float):
        return Rational(str(result))
    if result is I:
        return Symbol("I")
    if not result.args:
        return result
    args = [_rebuild(arg) for arg in result.args]
    if isinstance(result, Relational):
        return result.func(*args, evaluate=False)
    return result.func(*args)
