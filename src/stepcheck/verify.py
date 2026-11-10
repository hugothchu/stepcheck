r"""The two checkers, as plain functions from SymPy objects to a `Finding`.

* :func:`check_identity` (equivalence): does ``a = b`` hold for *every* value?
  Proved by ``simplify(a - b) == 0``; disproved by a counterexample, exact values
  at which ``a - b`` is not zero.
* :func:`check_statement` (consistency): is a condition consistent with the earlier
  conditions of a solution? Solved with SymPy. In one variable the solution sets are
  compared: the same set is correct, a set that loses or adds solutions is a warning,
  and no shared solution is wrong. With several variables, the line is correct if
  every earlier solution satisfies it, a warning if only some do, and wrong if none do.

Every symbol is taken to be a real number, as in school and undergraduate maths:
so ``\ln(e^x) = x`` is an identity, which it isn't for complex x.

There is no numeric fallback: every verdict is exact, or the answer is "unknown".
These functions can be slow; callers run them with a time limit (`timeout.py`).

    >>> from sympy import cos, sin, symbols
    >>> x = symbols("x")
    >>> check_identity(sin(x)**2 + cos(x)**2, 1).verdict
    <Verdict.CORRECT: 'correct'>
    >>> check_identity((x + 1)**2, x**2 + 1).witness
    {'x': '1'}
"""

from collections.abc import Sequence
from itertools import islice

from sympy import (
    Add,
    And,
    Basic,
    Complement,
    Eq,
    Expr,
    FiniteSet,
    Intersection,
    Or,
    Rational,
    S,
    Set,
    Symbol,
    Union,
    nan,
    oo,
    simplify,
    solve,
    solveset,
    zoo,
)
from sympy.core.relational import Relational
from sympy.logic.boolalg import Boolean, BooleanAtom, to_dnf
from sympy.sets.conditionset import ConditionSet

from stepcheck.models.verdicts import Finding, Method, Verdict

# Exact values tried, in order, when looking for a counterexample to an identity.
_TRIAL_VALUES: tuple[Expr, ...] = tuple(
    Rational(value) for value in ("1", "2", "-1", "3", "1/2", "-2", "5", "-1/3", "7", "0")
)


# ------------------------------------------------------------------- equivalence


def check_identity(left: Expr, right: Expr) -> Finding:
    """Is ``left = right`` true for every value of its symbols?

    Continuation lines (``= ...``) and identities such as ``\\sin 2x = 2\\sin x\\cos x``
    are claims of this kind. Derivatives, integrals and sums are evaluated first.
    """
    difference = _as_real((left - right).doit())
    if simplify(difference) == 0:
        return _finding(Verdict.CORRECT, Method.SIMPLIFY, f"simplify({_minus(left, right)}) = 0")

    if not difference.free_symbols:  # plain arithmetic: 3*4 = 13
        if simplify(difference).is_zero is False:
            reason = f"{_minus(left, right)} = {simplify(difference)}, not 0"
            return _finding(Verdict.WRONG, Method.SIMPLIFY, reason)
        return _finding(Verdict.UNKNOWN, Method.SIMPLIFY, "can't decide whether a constant is 0")

    if (counterexample := _counterexample(difference)) is not None:
        values, value = counterexample
        at = ", ".join(f"{symbol} = {number}" for symbol, number in values.items())
        reason = f"not an identity: at {at}, {_minus(left, right)} = {value}, not 0"
        witness = {str(symbol): str(number) for symbol, number in values.items()}
        return _finding(Verdict.WRONG, Method.SIMPLIFY, reason, witness)

    reason = f"simplify() can't reduce {_minus(left, right)} to 0, and finds no counterexample"
    return _finding(Verdict.UNKNOWN, Method.SIMPLIFY, reason)


def _as_real(expression: Expr) -> Expr:
    """The same expression with every symbol declared real (same names)."""
    real = {symbol: Symbol(symbol.name, real=True) for symbol in _symbols(expression)}
    result = expression.xreplace(real)
    assert isinstance(result, Expr)  # renaming symbols can't change what kind of object it is
    return result


def _minus(left: Expr, right: Expr) -> str:
    """Show ``left - right`` as written: ``(x + 1)**2 - (x**2 + 1)``, ``12 - 13``."""
    return f"{left} - ({right})" if isinstance(right, Add) else f"{left} - {right}"


def _counterexample(difference: Expr) -> tuple[dict[Symbol, Expr], Expr] | None:
    """Exact values at which `difference` is defined and provably not 0, if we find any.

    Symbols get different trial values in each round (x = 1, y = 2; then x = 2, y = -1;
    ...), so a coincidence such as x = y doesn't hide a counterexample.
    """
    symbols = sorted(_symbols(difference), key=str)
    for round_ in range(len(_TRIAL_VALUES)):
        values = {
            symbol: _TRIAL_VALUES[(round_ + position) % len(_TRIAL_VALUES)]
            for position, symbol in enumerate(symbols)
        }
        value = simplify(difference.subs(values.items()))
        if value.has(zoo, nan, oo, -oo):
            continue  # undefined there, e.g. 1/x at x = 0
        if value.is_zero is False:
            return values, value
    return None


# ------------------------------------------------------------------- consistency


def check_statement(statement: Boolean, earlier: Sequence[Boolean]) -> Finding:
    """Check a statement line against the earlier conditions of the solution.

    1. No symbols (``3 \\cdot 4 = 13``, ``2 < 5``): it is simply true or false.
    2. An equation that holds for every value (``\\sin^2 x + \\cos^2 x = 1``): correct.
    3. Otherwise it's a condition. Only earlier conditions that share symbols with
       it (directly or through each other) are relevant. With none, or if it
       introduces a new symbol, it's taken as given; else it's solved against them.
    """
    if not statement.free_symbols:
        return _check_constant(statement)
    if isinstance(statement, Eq):
        identity = check_identity(*_sides(statement))
        if identity.verdict == Verdict.CORRECT:
            return identity

    relevant = _connected(statement, earlier)
    known = set().union(*map(_symbols, relevant))
    if new := _symbols(statement) - known:
        names = ", ".join(sorted(map(str, new)))
        what = (
            "the first condition"
            if not earlier
            else f"introduces {names}"
            if relevant
            else "shares no symbol with the earlier conditions"
        )
        return _finding(Verdict.GIVEN, Method.NONE, f"{what}: taken as given")

    symbols = sorted(_symbols(statement) | known, key=str)
    if len(symbols) == 1:
        return _check_with_solution_sets(statement, relevant, symbols[0])
    if all(_is_equations(condition) for condition in (statement, *relevant)):
        return _check_with_solve(statement, relevant, symbols)
    reason = "inequalities in more than one variable aren't supported yet"
    return _finding(Verdict.UNKNOWN, Method.SOLVE, reason)


def _check_constant(statement: Boolean) -> Finding:
    if isinstance(statement, Eq):
        return check_identity(*_sides(statement))
    value = simplify(statement)
    if isinstance(value, BooleanAtom):
        verdict = Verdict.CORRECT if value is S.true else Verdict.WRONG
        return _finding(verdict, Method.SIMPLIFY, f"{statement} is {value}")
    return _finding(Verdict.UNKNOWN, Method.SIMPLIFY, f"can't decide whether {statement} holds")


def _connected(statement: Boolean, earlier: Sequence[Boolean]) -> list[Boolean]:
    """The earlier conditions linked to `statement` through shared symbols."""
    symbols = _symbols(statement)
    relevant: list[Boolean] = []
    remaining = list(earlier)
    while linked := [c for c in remaining if _symbols(c) & symbols]:
        for condition in linked:
            relevant.append(condition)
            remaining.remove(condition)
            symbols |= _symbols(condition)
    return [condition for condition in earlier if condition in relevant]  # keep line order


# One variable: compare solution sets over the reals. Handles inequalities too.


def _check_with_solution_sets(
    statement: Boolean, earlier: Sequence[Boolean], symbol: Symbol
) -> Finding:
    before = Intersection(*(_solution_set(condition, symbol) for condition in earlier))
    after = _solution_set(statement, symbol)
    if before.has(ConditionSet) or after.has(ConditionSet):
        return _finding(Verdict.UNKNOWN, Method.SOLVE, "SymPy can't solve these conditions")

    both = Intersection(before, after)
    if both.is_empty:
        reason = f"the earlier conditions give {symbol} ∈ {before}; this line, {symbol} ∈ {after}"
        witness = _first_element(before, symbol)
        return _finding(Verdict.WRONG, Method.SOLVE, f"no common solution: {reason}", witness)
    if both.is_empty is None:
        return _finding(Verdict.UNKNOWN, Method.SOLVE, "can't compare the solution sets")

    # Consistent. A step should keep exactly the same solutions: losing some (x² = 4 →
    # x = 2) or adding some (x = 2 → x² = 4, or a final answer with an extra root) is
    # a warning, not an error.
    lost, extra = Complement(before, after), Complement(after, before)
    if lost.is_empty and extra.is_empty:
        reason = f"same solutions as the earlier conditions: {symbol} ∈ {before}"
        return _finding(Verdict.CORRECT, Method.SOLVE, reason)
    changes = [
        f"{label} {symbol} ∈ {difference}"
        for label, difference in (("loses", lost), ("adds", extra))
        if difference.is_empty is False
    ]
    if not changes:
        return _finding(Verdict.UNKNOWN, Method.SOLVE, "can't compare the solution sets")
    reason = f"consistent with the earlier conditions, but {' and '.join(changes)}"
    return _finding(Verdict.WARNING, Method.SOLVE, reason)


def _solution_set(condition: Boolean, symbol: Symbol) -> Set:
    if isinstance(condition, And):
        return Intersection(*(_solution_set(part, symbol) for part in condition.args))
    if isinstance(condition, Or):
        return Union(*(_solution_set(part, symbol) for part in condition.args))
    return solveset(condition, symbol, domain=S.Reals)


def _first_element(solutions: Set, symbol: Symbol) -> dict[str, str] | None:
    """One earlier solution, as evidence: it satisfies the earlier lines, not this one."""
    if isinstance(solutions, FiniteSet):
        return {str(symbol): str(next(iter(solutions)))}
    return None


# Several variables: equations only. Solve the earlier system, then substitute.


def _check_with_solve(
    statement: Boolean, earlier: Sequence[Boolean], symbols: list[Symbol]
) -> Finding:
    solutions = [
        solution
        for branch in _branches(earlier)
        for solution in solve(branch, symbols, dict=True)
        if all(value.is_real is not False for value in solution.values())
    ]
    if not solutions:
        reason = "SymPy finds no real solution to the earlier conditions"
        return _finding(Verdict.UNKNOWN, Method.SOLVE, reason)

    holds = [_truth(statement.subs(solution.items())) for solution in solutions]
    if all(value is True for value in holds):
        reason = f"follows from the earlier conditions, whose solutions are {_show(solutions)}"
        return _finding(Verdict.CORRECT, Method.SOLVE, reason)
    if any(value is True for value in holds):
        lost = [solution for solution, value in zip(solutions, holds, strict=True) if not value]
        reason = f"consistent with the earlier conditions, but loses {_show(lost)}"
        return _finding(Verdict.WARNING, Method.SOLVE, reason)
    if all(value is False for value in holds):
        reason = f"no common solution: the earlier conditions give {_show(solutions)}"
        witness = {str(symbol): str(value) for symbol, value in solutions[0].items()}
        return _finding(Verdict.WRONG, Method.SOLVE, reason, witness)

    # Solutions with free parameters (x + y = 3 gives x = 3 - y): solve everything.
    together = [[*branch, statement] for branch in _branches(earlier)]
    if not any(solve(branch, symbols, dict=True) for branch in together):
        return _finding(Verdict.WRONG, Method.SOLVE, "no common solution with the earlier lines")
    reason = "consistent with the earlier conditions, but doesn't follow from them"
    return _finding(Verdict.WARNING, Method.SOLVE, reason)


def _branches(conditions: Sequence[Boolean]) -> list[list[Boolean]]:
    """``x = 2 or x = 3``, ``y = 1`` → [[x = 2, y = 1], [x = 3, y = 1]] (disjunctive form)."""
    combined = to_dnf(And(*conditions))
    return [list(And.make_args(term)) for term in Or.make_args(combined)]


def _is_equations(condition: Boolean) -> bool:
    """Equations joined by and/or only: no inequalities."""
    return all(isinstance(relation, Eq) for relation in condition.atoms(Relational))


def _truth(statement: Basic) -> bool | None:
    value = simplify(statement)
    return True if value is S.true else False if value is S.false else None


def _show(solutions: list[dict[Symbol, Expr]], limit: int = 4) -> str:
    shown = [
        "{" + ", ".join(f"{symbol} = {value}" for symbol, value in solution.items()) + "}"
        for solution in islice(solutions, limit)
    ]
    return " or ".join(shown) + (" or ..." if len(solutions) > limit else "")


# SymPy's type stubs say `.lhs` and `.free_symbols` are any `Basic`; these two
# helpers narrow them, checking at runtime rather than silencing the type checker.


def _sides(equation: Eq) -> tuple[Expr, Expr]:
    left, right = equation.lhs, equation.rhs
    if not (isinstance(left, Expr) and isinstance(right, Expr)):
        raise TypeError(f"both sides of an equation should be expressions: {equation}")
    return left, right


def _symbols(expression: Basic) -> set[Symbol]:
    return {symbol for symbol in expression.free_symbols if isinstance(symbol, Symbol)}


def _finding(
    verdict: Verdict, method: Method, reason: str, witness: dict[str, str] | None = None
) -> Finding:
    return Finding(verdict=verdict, method=method, reason=reason, witness=witness)
