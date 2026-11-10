r"""Command line: ``stepcheck check solution.tex``.

Prints one line per solution line with its verdict and reason. The exit status is
1 if any line is wrong, so the command can gate a script or CI job.

    stepcheck check solution.tex           # a table
    stepcheck check solution.tex --json    # the CheckReport as JSON
    echo 'x^2 = 4 \\ x = 3' | stepcheck check -
"""

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from stepcheck.check import DEFAULT_TIMEOUT, check_latex
from stepcheck.models import CheckReport, Verdict


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    text = sys.stdin.read() if arguments.file == "-" else Path(arguments.file).read_text()
    report = check_latex(text, timeout=arguments.timeout)
    print(report.model_dump_json(indent=2) if arguments.json else _table(report))
    return 1 if report.first_error else 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="stepcheck", description="Check every step of a worked maths solution with SymPy."
    )
    commands = parser.add_subparsers(dest="command", required=True)
    check = commands.add_parser("check", help="check every line of a LaTeX solution")
    check.add_argument("file", help="a .tex file with one step per line, or - for stdin")
    check.add_argument("--json", action="store_true", help="print the report as JSON")
    check.add_argument(
        "--timeout",
        type=float,
        default=DEFAULT_TIMEOUT,
        help=f"seconds allowed per line (default {DEFAULT_TIMEOUT:g})",
    )
    return parser


def _table(report: CheckReport) -> str:
    width = min(max(len(line.latex) for line in report.lines), 40)
    rows = [
        f"{line.index + 1:>3}  {line.verdict.symbol}  {line.latex:<{width}}  {line.reason}"
        + (f"  (fails at {_format(line.witness)})" if line.witness else "")
        for line in report.lines
    ]
    counts = report.counts
    summary = ", ".join(f"{counts[v]} {v.value}" for v in Verdict if counts[v])
    first = report.first_error
    verdict_line = f"first error: line {first.index + 1}" if first else "no errors found"
    return "\n".join([*rows, "", f"{summary}; {verdict_line}"])


def _format(witness: dict[str, str]) -> str:
    return ", ".join(f"{name} = {value}" for name, value in witness.items())


if __name__ == "__main__":
    sys.exit(main())
