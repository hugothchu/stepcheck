# stepcheck: plan and design decisions

> LLMs write fluent maths that is often wrong. stepcheck checks each step of a worked
> solution with SymPy, says exactly where the reasoning breaks, and feeds that back to
> the model so it can repair it. Then it measures how much this helps.

## Milestones (~25h over 2 weeks)

| # | Status | Deliverable |
|---|---|---|
| M1: Parsing and models | ✅ done | `parse.py` (latex2sympy2_extended + clean-up + misparse guard), Pydantic `Line` and `Solution`, tests |
| M2: The two checkers | next | Equivalence (`simplify`) and consistency (solver) checks; `stepcheck check solution.tex` with no LLM. Repo goes public. |
| M3: LLM solver + tools | | LLM writes LaTeX steps → `Solution` → per-line report. `simplify` and `solve` exposed to the LLM as **tools** for direct questions. |
| M4: Evals | | GSM8K + MATH-500 subsets, 2 models: accuracy, coverage (share of lines checkable), cost, latency |
| M5: Repair loop | | Feed the flagged line + SymPy evidence back; re-run evals |
| M6: MCP, demo, polish | | MCP server, Gradio app on HF Spaces, README with results table |
| Stretch | | `biggest_consistent_set` error localisation, measured on ProcessBench |

## Design decisions

**Input is LaTeX only**, parsed with `latex2sympy2_extended`, so the checker works on any
solution, from an LLM or a person. Unparseable lines become `LineKind.UNPARSED` and are
reported as "unknown"; they never crash the whole solution.

**Pydantic owns parsing and structure; plain functions own the maths.** A `Line` parses
its LaTeX in a validator; a `Solution` checks its own structure. Simplifying and solving
(slow, can time out) live in functions that take models and return typed result models.
Models live in `stepcheck/models/`, one module per concern (`solution.py` now; verdicts,
LLM tool I/O and eval records later), re-exported from `stepcheck.models`.

**Two checkers, no numeric fallback.**
- *Identity claims* (rewrites, typically continuation lines `= ...`, and identities such
  as `\sin^2 x + \cos^2 x = 1` or `\sin 2\theta = 2\sin\theta\cos\theta`) use
  `simplify(a - b) == 0`. They're refuted without sampling: if `a - b = 0` has only
  finitely many solutions, the identity is false. Anything else is "unknown", never "wrong".
- *Conditions* (equations being solved, including systems) use the solver: lines are
  consistent if they share at least one solution. If solution sets differ inside a
  consistent run, a separate "lost/extra roots" warning is raised (`x^2 = 4` → `x = 2`).
  A wrong rewrite must **not** go to the solver: `(x+1)^2 = x^2 + 1` has the solution
  `x = 0` and would pass.

**Error localisation (stretch): `biggest_consistent_set` over contiguous runs.** Find the
longest consistent run, recurse on the lines either side, then at each border between
runs that share symbols, flag the first line of the later run. Lines with no solution on
their own (`3 \cdot 4 = 13`) are flagged directly. Alternative to compare: greedy growth
from the left, which is provably minimal in the number of runs (consistency is
hereditary) and flags the first line that contradicts everything before it.

## Parser findings (SymPy 1.14.0, latex2sympy2_extended 1.11.0)

latex2sympy2_extended beats both built-in SymPy backends: `antlr` silently turns `\pm`
and `\boxed` into symbols; `lark` evaluates eagerly and returns an ambiguous parse tree
for `\sin^2 x + \cos^2 x`. Pinned with `antlr4-python3-runtime==4.11.0`.

What `parse.py` adds on top (each one has a test in `tests/test_parse.py`):

| Problem | Handling |
|---|---|
| Default `lowercase_symbols=True` turns `A + a` into `2a` | disabled |
| `x = 2, x = 3` and `x = 2 \text{ or } x = 3` both parse to the same set | split "and"/"or" ourselves; mixing both in one line is rejected as ambiguous |
| `x = \pm 2` → `Eq(x, {-2, 2})` | expanded to `Or(Eq(x, -2), Eq(x, 2))` |
| `1{,}000` / `1,000` → a set `{1, 0}` with no error | thousands separators removed first; any unexpected set is rejected |
| `50\%`, `\begin{cases}`, `&=` and `\implies` fail | rewritten or split before parsing |
| `\vec{v}`, `\boxed`, `\text{km}` become symbols | symbol guard: one letter or Greek (plus `\ell`, `\hbar`), optional `\bar`/`\hat`/`\tilde`/`\dot` accent, any subscript (`M_{1}`, `v_{\text{max}}`, `x_{i,j}`); multi-letter names are rejected |
| unit stripping: `5 \text{ km} = 5000 \text{ m}` → `5` | switched off: `\text{km}` becomes a symbol `km`, which the guard rejects (unit handling is out of scope) |
| `I` becomes the imaginary unit (`\mu_0 I`) | replaced by a plain symbol `I` (a current) |
| `f'(x)` and `v'` silently lose the prime | primes rejected |
| `\sin 30^{\circ}` is sin(30 radians) | degrees converted to radians |
| `\Delta x` is `Delta * x` | rewritten to one symbol `Delta_x` (a bare `\Delta` stays a symbol) |
| `E_{k}` is a syntax error (E is Euler's number) | rewritten to `\mathit{E}_{k}` |
| `\ln{\left(x \right)}^{2}` (SymPy's own printing of (ln x)²) is ln(x²) | braces around one bracketed argument are removed; found by Hypothesis |
| `a{,}b` | `{,}` is only removed between digits; otherwise the set guard rejects it |
| `$` stripping would eat `5 \$` | only unescaped `$` (math mode) is removed |
| decimals become floats | converted to exact rationals |
| unevaluated trees (`1*5*x`), wrong arithmetic could collapse to `False` | arithmetic evaluated; relations rebuilt unevaluated |
| parser can return `list` / `None` | rejected |

Trigonometry: `\sin 2\theta` correctly parses as `sin(2θ)`, and `simplify` proves both
example identities. Bracketless arguments are still ambiguous (`2 \sin x \cdot 3` →
`2 sin(3x)`, recorded as a strict xfail; `\sin x y` → `sin(xy)`), so **the LLM prompt
must require brackets around every function argument**, and units inside `\text{}`
(`s = 5 m` can't be told apart from `s = 5 \cdot m`). Heuristics can come later.

Not supported yet: set notation as a line's content (`\{1, 2, 3\}`), the imaginary unit
(`i` stays a symbol), chains with several `=` on one continuation line, vectors,
primes, ion concentrations (`[H^+]`), named quantities written as words
(`\mathrm{pH}`), and units. `\frac{\partial f}{\partial x}` treats `f` as a
constant, so it evaluates to 0 (a strict xfail). A lone `e` is Euler's number, which
clashes with the electron charge in physics.

## Testing

- `tests/test_parse.py`: the parser's traps, one test per rule in the table above.
- `tests/test_disciplines.py`: notation from physics, chemistry, logarithms,
  trigonometry and calculus, including what must be *rejected*.
- `tests/test_properties.py` (Hypothesis): random SymPy expressions → `sympy.latex` →
  parse must give the same maths; thousands separators, decimals, `\pm` and degrees
  for any number; and arbitrary LaTeX fragments must become a `Line` without raising.
  `HYPOTHESIS_PROFILE=thorough` runs 20× more examples.
