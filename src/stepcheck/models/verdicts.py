"""What the checkers conclude about each line of a solution.

A `Finding` is one checker's conclusion; a `LineCheck` places it on a line; a
`CheckReport` holds one `LineCheck` per line, in order.

    >>> finding = Finding(verdict=Verdict.WRONG, method=Method.SIMPLIFY,
    ...                   reason="(x + 1)**2 - (x**2 + 1) = 2*x, not 0",
    ...                   witness={"x": "1"})
    >>> finding.verdict.symbol
    '✗'
"""

from enum import StrEnum
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from stepcheck.parse import LineKind


class Verdict(StrEnum):
    GIVEN = "given"  # starts something (an expression, the problem's equation): not judged
    CORRECT = "correct"  # proved: an identity holds, or the line follows from earlier ones
    WARNING = "warning"  # consistent with earlier lines, but loses some of their solutions
    WRONG = "wrong"  # disproved: a counterexample, or no solution shared with earlier lines
    UNKNOWN = "unknown"  # couldn't decide: unparsed, unsupported, or timed out

    @property
    def symbol(self) -> str:
        return {"given": "·", "correct": "✓", "warning": "!", "wrong": "✗", "unknown": "?"}[self]


class Method(StrEnum):
    SIMPLIFY = "simplify"  # equivalence: simplify(a - b) == 0, or a counterexample
    SOLVE = "solve"  # consistency: solution sets of this line and the earlier conditions
    NONE = "none"  # nothing was computed (given or unparsed lines)


class Finding(BaseModel):
    """A checker's conclusion about one claim, with the evidence for it."""

    model_config = ConfigDict(frozen=True)

    verdict: Verdict
    method: Method
    reason: str = Field(min_length=1)
    # Exact values at which the claim fails, e.g. {"x": "1"}: a counterexample to an
    # identity, or an earlier solution the line rejects. A model can recheck them.
    witness: dict[str, str] | None = None

    @model_validator(mode="after")
    def evidence_fits_the_verdict(self) -> Self:
        decided = self.verdict in (Verdict.CORRECT, Verdict.WARNING, Verdict.WRONG)
        if decided and self.method == Method.NONE:
            raise ValueError(f"a {self.verdict.value} verdict needs a method that decided it")
        if self.witness is not None and self.verdict != Verdict.WRONG:
            raise ValueError("only a wrong verdict has a counterexample")
        return self


class LineCheck(Finding):
    """The finding for one line of a solution."""

    index: int = Field(ge=0)
    latex: str
    kind: LineKind


class CheckReport(BaseModel):
    """One `LineCheck` per line of a solution, in order."""

    model_config = ConfigDict(frozen=True)

    lines: tuple[LineCheck, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def lines_are_in_order(self) -> Self:
        if [line.index for line in self.lines] != list(range(len(self.lines))):
            raise ValueError("line checks must be numbered 0, 1, 2, ... in order")
        return self

    @property
    def first_error(self) -> LineCheck | None:
        """The first wrong line: where the reasoning breaks."""
        return next((line for line in self.lines if line.verdict == Verdict.WRONG), None)

    @property
    def counts(self) -> dict[Verdict, int]:
        return {verdict: sum(line.verdict == verdict for line in self.lines) for verdict in Verdict}
