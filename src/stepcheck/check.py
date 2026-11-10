r"""Check a whole solution, line by line.

Each line gets one `LineCheck`:

* **expression** lines (``(x + 1)^2``) start a chain of rewrites: given;
* **continuation** lines (``= x^2 + 2x + 1``) claim an identity with the value before
  them: checked with `check_identity`;
* **statement** lines (``x^2 - 5x + 6 = 0``) are checked with `check_statement`
  against the *system*: the earlier statements that were given or correct. Wrong,
  warning and unknown lines are left out, so one doubtful step doesn't decide the
  verdict of every line after it. Identities (``\sin^2 x + \cos^2 x = 1``) say nothing about
  the unknowns, so they don't join the system either;
* **unparsed** lines are unknown, with the parser's reason.

    >>> report = check_latex(r"x^2 - 5x + 6 = 0 \\ (x - 2)(x - 4) = 0 \\ x = 5", timeout=None)
    >>> [line.verdict.value for line in report.lines]
    ['given', 'warning', 'wrong']
"""

from collections.abc import Callable
from typing import Any

from sympy.logic.boolalg import Boolean

from stepcheck.models import CheckReport, Finding, Line, LineCheck, Method, Solution, Verdict
from stepcheck.parse import LineKind
from stepcheck.timeout import CheckAborted, run_with_timeout
from stepcheck.verify import check_identity, check_statement

DEFAULT_TIMEOUT = 10.0  # seconds per line


def check_latex(text: str, *, timeout: float | None = DEFAULT_TIMEOUT) -> CheckReport:
    """Parse a LaTeX solution and check every line."""
    return check_solution(Solution.from_latex(text), timeout=timeout)


def check_solution(solution: Solution, *, timeout: float | None = DEFAULT_TIMEOUT) -> CheckReport:
    """Check every line of `solution`; each check may take up to `timeout` seconds.

    `timeout=None` runs the checks in this process with no time limit (faster to
    start, for tests and trusted input).
    """
    return _Checker(solution, timeout).run()


class _Checker:
    """Walks a solution once, keeping two pieces of state between lines.

    * `system`: the statements later conditions are solved against;
    * `anchor`: the line a `= ...` continuation is compared with: the start of the
      chain, or its last *correct* step. So in ``(x+1)^2 \\ = x^2 + 1 \\ = x^2 + 2x + 1``
      the wrong second line doesn't make the correct third line wrong too.
    """

    def __init__(self, solution: Solution, timeout: float | None) -> None:
        self.solution = solution
        self.timeout = timeout
        self.system: list[Boolean] = []
        self.anchor: int | None = None

    def run(self) -> CheckReport:
        checks = [self._check(index, line) for index, line in enumerate(self.solution.lines)]
        return CheckReport(lines=tuple(checks))

    def _check(self, index: int, line: Line) -> LineCheck:
        finding = self._find(index, line)
        self._remember(index, line, finding)
        return LineCheck(index=index, latex=line.latex, kind=line.kind, **finding.model_dump())

    def _find(self, index: int, line: Line) -> Finding:
        match line.kind:
            case LineKind.UNPARSED:
                return _unknown(f"not parsed: {line.parse_error}")
            case LineKind.EXPRESSION:
                reason = "an expression: the '= ...' lines after it are checked against it"
                return Finding(verdict=Verdict.GIVEN, method=Method.NONE, reason=reason)
            case LineKind.CONTINUATION:
                claim = self.solution.identity_claim(index, against=self.anchor)
                if claim is None:
                    return _unknown("nothing to continue from: no earlier line has a value")
                return _run(check_identity, claim.lhs, claim.rhs, timeout=self.timeout)
            case LineKind.STATEMENT:
                return _run(check_statement, line.content, self.system, timeout=self.timeout)

    def _remember(self, index: int, line: Line, finding: Finding) -> None:
        match line.kind:
            case LineKind.UNPARSED:
                self.anchor = None  # the chain is broken: nothing to continue from
            case LineKind.EXPRESSION:
                self.anchor = index
            case LineKind.CONTINUATION:
                if finding.verdict == Verdict.CORRECT:
                    self.anchor = index
            case LineKind.STATEMENT:
                self.anchor = index
                if isinstance(line.content, Boolean) and _joins_the_system(finding):
                    self.system.append(line.content)


def _joins_the_system(finding: Finding) -> bool:
    """Given and solver-checked lines constrain the unknowns; identities don't."""
    trusted = finding.verdict in (Verdict.GIVEN, Verdict.CORRECT)
    return trusted and finding.method != Method.SIMPLIFY


def _run(check: Callable[..., Finding], *args: Any, timeout: float | None) -> Finding:
    try:
        return run_with_timeout(check, *args, seconds=timeout)
    except CheckAborted as error:
        return _unknown(f"stopped: {error}")
    except Exception as error:  # SymPy raises many kinds: NotImplementedError, PolynomialError…
        return _unknown(f"SymPy failed: {type(error).__name__}: {error}")


def _unknown(reason: str) -> Finding:
    return Finding(verdict=Verdict.UNKNOWN, method=Method.NONE, reason=reason)
