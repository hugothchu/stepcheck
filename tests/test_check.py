r"""Whole solutions: how verdicts flow from line to line."""

import pytest

from stepcheck.check import check_latex
from stepcheck.models import Verdict

G, C, W, X, U = Verdict.GIVEN, Verdict.CORRECT, Verdict.WARNING, Verdict.WRONG, Verdict.UNKNOWN


def verdicts(text: str) -> list[Verdict]:
    return [line.verdict for line in check_latex(text, timeout=None).lines]


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        (r"x^2 - 5x + 6 = 0 \\ (x - 2)(x - 3) = 0 \\ x = 2 \text{ or } x = 3", [G, C, C]),
        (r"(x + 1)^2 \\ = x^2 + 2x + 1 \\ = x(x + 2) + 1", [G, C, C]),
        (
            r"\begin{cases} x + y = 3 \\ x - y = 1 \end{cases} \\ 2x = 4 \\ x = 2 \\ y = 1",
            [G, C, C, C],
        ),
        (r"y = 3x + 1 \\ = 3(x + 1) - 2", [G, C]),  # continues from the right-hand side
        (r"\sin 2\theta \\ = 2\sin\theta\cos\theta", [G, C]),
        (r"3 \cdot 4 = 12 \\ 12 + 3 = 15 \\ \boxed{15}", [C, C, G]),
    ],
    ids=["quadratic", "expansion", "system", "rewrite-a-definition", "trig", "arithmetic"],
)
def test_correct_solutions(text: str, expected: list[Verdict]) -> None:
    assert verdicts(text) == expected


def test_a_wrong_rewrite_does_not_spread_along_the_chain() -> None:
    # Line 3 is compared with line 1 (the last correct value), not with wrong line 2.
    assert verdicts(r"(x + 1)^2 \\ = x^2 + 1 \\ = x^2 + 2x + 1") == [G, X, C]


def test_a_wrong_condition_does_not_join_the_system() -> None:
    # After the wrong y = 5, y = 1 is still checked against the original system.
    text = r"\begin{cases} x + y = 3 \\ x - y = 1 \end{cases} \\ y = 5 \\ y = 1"
    assert verdicts(text) == [G, X, C]


def test_a_warning_does_not_join_the_system() -> None:
    # x = 2 loses a root; the final answer is still judged against x^2 - 5x + 6 = 0.
    text = r"x^2 - 5x + 6 = 0 \\ x - 2 = 0 \\ x = 2 \text{ or } x = 3"
    assert verdicts(text) == [G, W, C]


def test_an_extra_root_in_the_answer_is_a_warning() -> None:
    assert verdicts(r"x^2 - 5x + 6 = 0 \\ x = 2 \text{ or } x = 3 \text{ or } x = 4") == [G, W]


def test_an_identity_line_does_not_join_the_system() -> None:
    text = r"\sin^{2} x + \cos^{2} x = 1 \\ x^2 = 4 \\ x = \pm 2"
    assert verdicts(text) == [C, G, C]  # x^2 = 4 is still the first condition


def test_an_unparsed_line_breaks_the_chain() -> None:
    assert verdicts(r"(x + 1)^2 \\ \vec{v} \\ = x^2 + 2x + 1") == [G, U, U]


def test_the_report_names_the_first_error() -> None:
    report = check_latex(r"x^2 = 4 \\ x = 2 \\ x = 3 \\ x = 5", timeout=None)
    assert report.first_error is not None
    assert report.first_error.index == 2
    assert report.counts[Verdict.WRONG] == 2


def test_lines_keep_their_latex_and_kind() -> None:
    line = check_latex(r"(x + 1)^2 \\ = x^2 + 1", timeout=None).lines[1]
    assert (line.latex, line.kind.value) == ("= x^2 + 1", "continuation")


def test_the_worker_gives_the_same_verdicts_and_reasons() -> None:
    # Through the worker process, unevaluated lines such as 3·4 = 13 must survive the
    # trip (plain pickling would turn them into `False`).
    text = r"3 \cdot 4 = 13 \\ x^2 = 4 \\ x = 2 \\ (x + 1)^2 \\ = x^2 + 1"
    here, there = check_latex(text, timeout=None), check_latex(text, timeout=30)
    assert here == there
    assert "12 - 13 = -1" in there.lines[0].reason
