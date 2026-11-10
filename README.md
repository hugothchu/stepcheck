# stepcheck

Check every step of a worked maths solution with SymPy.

LLMs write fluent maths that is often wrong. stepcheck parses a solution written in
LaTeX, checks each line with a computer algebra system, and reports exactly where the
reasoning breaks.

> 🚧 Work in progress. The plan and design decisions are in [docs/plan.md](docs/plan.md).

## Usage

```bash
uv sync
uv run stepcheck check examples/quadratic.tex
```

```
  1  ·  x^2 - 5x + 6 = 0    the first condition: taken as given
  2  ✓  (x - 2)(x - 3) = 0  same solutions as the earlier conditions: x ∈ {2, 3}
  3  !  x - 2 = 0           consistent with the earlier conditions, but loses x ∈ {3}
  4  ✗  x = 4               no common solution: the earlier conditions give x ∈ {2, 3}; this line, x ∈ {4}  (fails at x = 2)

1 given, 1 correct, 1 warning, 1 wrong; first error: line 4
```

`·` given, `✓` correct, `!` warning (loses or adds solutions), `✗` wrong, `?` unknown.
Add `--json` for the full report. The exit status is 1 if any line is wrong.

From Python:

```python
from stepcheck.check import check_latex

report = check_latex(r"(x + 1)^2 \\ = x^2 + 1")
report.first_error.witness  # {'x': '1'}: (1 + 1)^2 = 4, but 1^2 + 1 = 2
```

## Development

```bash
uv sync
uv run pytest        # tests + doctests
HYPOTHESIS_PROFILE=thorough uv run pytest tests/test_properties.py tests/test_verify_properties.py   # deeper search
uv run ruff check .
uv run pyright
```
