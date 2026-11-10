"""`stepcheck check`: the table, the JSON report and the exit status."""

import io
from pathlib import Path

import pytest

from stepcheck.cli import main
from stepcheck.models import CheckReport, Verdict


def write(tmp_path: Path, text: str) -> str:
    path = tmp_path / "solution.tex"
    path.write_text(text)
    return str(path)


def test_a_correct_solution_exits_0(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    file = write(tmp_path, "x^2 = 4 \\\\\nx = \\pm 2\n")
    assert main(["check", file, "--timeout", "30"]) == 0
    assert "no errors found" in capsys.readouterr().out


def test_a_wrong_line_exits_1(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    file = write(tmp_path, "% comment\n(x + 1)^2 \\\\\n= x^2 + 1\n")
    assert main(["check", file, "--timeout", "30"]) == 1
    output = capsys.readouterr().out
    assert "first error: line 2" in output
    assert "fails at x = 1" in output


def test_json_is_a_check_report(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    main(["check", write(tmp_path, "x^2 = 4 \\\\ x = 3"), "--json", "--timeout", "30"])
    report = CheckReport.model_validate_json(capsys.readouterr().out)
    assert [line.verdict for line in report.lines] == [Verdict.GIVEN, Verdict.WRONG]


def test_reads_stdin(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.setattr("sys.stdin", io.StringIO(r"3 \cdot 4 = 12"))
    assert main(["check", "-", "--timeout", "30"]) == 0
    assert "✓" in capsys.readouterr().out
