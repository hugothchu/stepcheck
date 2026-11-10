import pytest
from pydantic import ValidationError
from sympy import Basic, Eq, symbols

from stepcheck.models import (
    CheckReport,
    Finding,
    Line,
    LineCheck,
    Method,
    Solution,
    Verdict,
)
from stepcheck.parse import LineKind

x, y = symbols("x y")


# --------------------------------------------------------------------------- Line


def test_line_is_parsed_on_construction() -> None:
    line = Line.from_latex(r"x^2 = 4")
    assert line.kind == LineKind.STATEMENT
    assert line.content == Eq(x**2, 4)
    assert line.parse_error is None


def test_line_from_a_continuation() -> None:
    assert Line.from_latex(r"= x + 1").kind == LineKind.CONTINUATION


def test_unparseable_line_is_recorded_not_raised() -> None:
    line = Line.from_latex(r"\vec{v} \cdot \vec{w}")
    assert line.kind == LineKind.UNPARSED
    assert line.content is None
    assert line.parse_error is not None and "suspicious symbol" in line.parse_error


def test_empty_line_is_invalid() -> None:
    with pytest.raises(ValidationError):
        Line.from_latex("")


@pytest.mark.parametrize("data", [{}, {"latex": None}, {"latex": 3}])
def test_missing_or_non_text_latex_is_a_validation_error(data: dict[str, object]) -> None:
    with pytest.raises(ValidationError, match="latex"):
        Line.model_validate(data)


def test_kind_and_content_must_agree() -> None:
    with pytest.raises(ValidationError, match="exactly when it was parsed"):
        Line(latex="x", kind=LineKind.UNPARSED, content=x)


@pytest.mark.parametrize(
    ("kind", "content"),
    [
        (LineKind.STATEMENT, x),  # a Symbol is a SymPy Boolean, but not a statement
        (LineKind.EXPRESSION, Eq(x, 1)),
        (LineKind.CONTINUATION, Eq(x, 1)),
    ],
)
def test_content_type_must_fit_the_kind(kind: LineKind, content: Basic) -> None:
    with pytest.raises(ValidationError, match="cannot hold"):
        Line(latex="x", kind=kind, content=content)


def test_lines_are_immutable() -> None:
    line = Line.from_latex("x = 1")
    with pytest.raises(ValidationError):
        line.latex = "x = 2"  # type: ignore[misc]


def test_line_serializes_to_json() -> None:
    dumped = Line.from_latex(r"x = \pm 2").model_dump(mode="json")
    assert dumped == {
        "latex": r"x = \pm 2",
        "kind": "statement",
        "content": "Eq(x, -2) | Eq(x, 2)",
        "parse_error": None,
    }


@pytest.mark.parametrize(
    ("latex", "value"),
    [
        (r"(x + 1)^2", (x + 1) ** 2),
        (r"y = 3x + 1", 3 * x + 1),
        (r"= 3x + 1", 3 * x + 1),
        (r"x > 2", None),
        (r"x = 1, y = 2", None),
        (r"\vec{v}", None),
    ],
)
def test_line_value(latex: str, value: object) -> None:
    assert Line.from_latex(latex).value == value


# ----------------------------------------------------------------------- Solution


SOLVE_QUADRATIC = r"""
x^2 - 5x + 6 = 0 \\
(x - 2)(x - 3) = 0 \\
x = 2 \text{ or } x = 3
"""


def test_solution_from_latex() -> None:
    solution = Solution.from_latex(SOLVE_QUADRATIC)
    assert [line.latex for line in solution.lines] == [
        "x^2 - 5x + 6 = 0",
        "(x - 2)(x - 3) = 0",
        r"x = 2 \text{ or } x = 3",
    ]
    assert all(line.kind == LineKind.STATEMENT for line in solution.lines)


def test_empty_solution_is_invalid() -> None:
    with pytest.raises(ValidationError):
        Solution.from_latex("  \\\\  ")


def test_solution_cannot_start_with_a_continuation() -> None:
    with pytest.raises(ValidationError, match="nothing to continue"):
        Solution.from_latex(r"= x + 1")


def test_one_bad_line_does_not_hide_the_others() -> None:
    solution = Solution.from_latex(r"x = 1 \\ \vec{v} \\ y = 2")
    assert [line.kind for line in solution.lines] == [
        LineKind.STATEMENT,
        LineKind.UNPARSED,
        LineKind.STATEMENT,
    ]


def test_identity_claims_from_a_rewrite_chain() -> None:
    solution = Solution.from_latex(r"(x + 1)^2 \\ = x^2 + 2x + 1 \\ = x(x + 2) + 1")
    assert solution.identity_claim(0) is None
    assert solution.identity_claim(1) == Eq((x + 1) ** 2, x**2 + 2 * x + 1)
    assert solution.identity_claim(2) == Eq(x**2 + 2 * x + 1, x * (x + 2) + 1)


def test_no_identity_claim_after_an_unparsed_line() -> None:
    solution = Solution.from_latex(r"\vec{v} \\ = x + 1")
    assert solution.identity_claim(1) is None


# ----------------------------------------------------------------------- verdicts


def test_a_decided_verdict_needs_a_method() -> None:
    with pytest.raises(ValidationError, match="needs a method"):
        Finding(verdict=Verdict.WRONG, method=Method.NONE, reason="because")


def test_only_a_wrong_verdict_has_a_witness() -> None:
    with pytest.raises(ValidationError, match="only a wrong verdict"):
        Finding(verdict=Verdict.CORRECT, method=Method.SIMPLIFY, reason="ok", witness={"x": "1"})


def _line_check(index: int) -> LineCheck:
    return LineCheck(
        index=index,
        latex="x",
        kind=LineKind.EXPRESSION,
        verdict=Verdict.GIVEN,
        method=Method.NONE,
        reason="an expression",
    )


def test_report_lines_must_be_in_order() -> None:
    with pytest.raises(ValidationError, match="in order"):
        CheckReport(lines=(_line_check(0), _line_check(2)))
