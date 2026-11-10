r"""Pydantic models for a worked solution.

Parsing and structural rules live here, in validators, so that a `Solution` you hold is
always well-formed. The mathematics (simplifying, solving) lives in separate functions
that take these models and return typed results.

    >>> solution = Solution.from_latex(r"(x + 1)^2 \\ = x^2 + 2x + 1 \\ x^2 + 2x + 1 = 0")
    >>> [line.kind.value for line in solution.lines]
    ['expression', 'continuation', 'statement']
    >>> solution.identity_claim(1)
    Eq((x + 1)**2, x**2 + 2*x + 1)
"""

from typing import Any, Self

from pydantic import BaseModel, ConfigDict, Field, field_serializer, model_validator
from sympy import Basic, Eq, Expr

from stepcheck.parse import LatexParseError, LineKind, parse_line, split_lines


class Line(BaseModel):
    """One line of a solution, parsed from LaTeX.

    Build one from LaTeX; `kind` and `content` are filled in by the parser:

        Line.from_latex(r"\\sin^{2} x + \\cos^{2} x = 1")

    A line the parser can't handle doesn't raise: it becomes `LineKind.UNPARSED` with
    the reason in `parse_error`, so one bad line never hides the rest of a solution.
    """

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    latex: str = Field(min_length=1)
    kind: LineKind
    content: Basic | None = None  # a SymPy Expr (expressions) or Boolean (statements)
    parse_error: str | None = None

    @classmethod
    def from_latex(cls, latex: str) -> Self:
        return cls.model_validate(latex)

    @model_validator(mode="before")
    @classmethod
    def parse_latex(cls, data: Any) -> Any:
        if isinstance(data, str):
            data = {"latex": data}
        if not isinstance(data, dict) or "kind" in data:
            return data  # already parsed (e.g. copied from another Line)
        if not isinstance(data.get("latex"), str):
            return data  # missing or not text: let field validation report it
        try:
            kind, content = parse_line(data["latex"])
        except LatexParseError as error:
            return {**data, "kind": LineKind.UNPARSED, "parse_error": str(error)}
        return {**data, "kind": kind, "content": content}

    @model_validator(mode="after")
    def content_matches_kind(self) -> Self:
        if (self.kind == LineKind.UNPARSED) != (self.content is None):
            raise ValueError("a line has content exactly when it was parsed")
        return self

    @field_serializer("content")
    def serialize_content(self, content: Basic | None) -> str | None:
        return None if content is None else str(content)

    @property
    def value(self) -> Expr | None:
        """The expression a following `= ...` line continues from, if there is one.

        `(x + 1)^2` → (x + 1)**2;   `y = 3x + 1` → 3*x + 1;   `x > 2` → None
        """
        if isinstance(self.content, Expr):  # expression and continuation lines
            return self.content
        if isinstance(self.content, Eq) and isinstance(self.content.rhs, Expr):
            return self.content.rhs  # a single equation: continue from its right side
        return None


class Solution(BaseModel):
    """A worked solution: an ordered, non-empty sequence of lines."""

    model_config = ConfigDict(frozen=True)

    lines: tuple[Line, ...] = Field(min_length=1)

    @classmethod
    def from_latex(cls, text: str) -> Self:
        r"""Split on ``\\``, newlines and ``\implies``, then parse each line."""
        return cls.model_validate({"lines": split_lines(text)})

    @model_validator(mode="after")
    def first_line_is_not_a_continuation(self) -> Self:
        if self.lines[0].kind == LineKind.CONTINUATION:
            raise ValueError("the first line cannot start with '=': there is nothing to continue")
        return self

    def identity_claim(self, index: int) -> Eq | None:
        """For a continuation line, the claim it makes: previous value = this value.

        Lines `(x + 1)^2` then `= x^2 + 2x + 1` claim Eq((x + 1)**2, x**2 + 2*x + 1).
        Returns None if the line isn't a continuation, or the previous line has no value.
        """
        line = self.lines[index]
        previous = self.lines[index - 1].value if index > 0 else None
        if line.kind != LineKind.CONTINUATION or previous is None or line.value is None:
            return None
        return Eq(previous, line.value, evaluate=False)  # pyright: ignore[reportReturnType]  (SymPy's stubs type Eq() as a union)
