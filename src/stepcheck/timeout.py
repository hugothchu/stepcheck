"""Run a slow SymPy call with a time limit.

`simplify()` and `solve()` can run for minutes on unlucky input, and a Python thread
can't be stopped from outside. So each call runs in a worker process, which is killed
and replaced when it runs over. The worker is started once and reused: starting a
process and importing SymPy takes about a second, a call takes milliseconds.

This works from any thread (Gradio and MCP servers call handlers from threads),
unlike `signal.alarm`, which only works in the main thread on Unix.

SymPy objects can't simply be pickled across: unpickling rebuilds them *evaluated*,
so a wrong line `Eq(12, 13)` (3·4 = 13) would arrive as `False`. They are sent, both
ways, as their expression tree instead and rebuilt with relations unevaluated.

    >>> run_with_timeout(pow, 2, 10, seconds=5.0)
    1024
"""

import multiprocessing
import threading
from collections.abc import Callable
from multiprocessing.connection import Connection
from multiprocessing.process import BaseProcess
from typing import Any

from sympy import Basic
from sympy.core.relational import Relational


class CheckAborted(Exception):
    """The call produced no result: it was stopped, or the worker process died."""


class CheckTimeout(CheckAborted):
    """The call didn't finish within its time limit."""


def run_with_timeout[T](function: Callable[..., T], *args: Any, seconds: float | None) -> T:
    """Call `function(*args)` in the worker process; `seconds=None` calls it here instead.

    `function` and its arguments must be picklable (module-level functions and SymPy
    objects are). Exceptions raised by `function` are re-raised here.
    """
    if seconds is None:
        return function(*args)
    return _WORKER.run(function, tuple(map(_freeze, args)), seconds)


def _serve(connection: Connection) -> None:
    """The worker's loop: receive a call, send back (succeeded, result or exception)."""
    while True:
        function, args = connection.recv()
        try:
            connection.send((True, _freeze(function(*map(_thaw, args)))))
        except Exception as error:
            connection.send((False, error))


# ------------------------------------------------------- sending SymPy objects


class _Tree:
    """A SymPy object as (class, children), so it can be rebuilt without evaluating."""

    def __init__(self, func: type, args: tuple[Any, ...]) -> None:
        self.func, self.args = func, args


def _freeze(value: Any) -> Any:
    if isinstance(value, Basic) and value.args:
        return _Tree(value.func, tuple(map(_freeze, value.args)))
    if isinstance(value, list | tuple):
        return type(value)(map(_freeze, value))
    return value  # atoms (symbols, numbers) and plain Python values pickle as they are


def _thaw(value: Any) -> Any:
    if isinstance(value, _Tree):
        args = map(_thaw, value.args)
        if issubclass(value.func, Relational):
            return value.func(*args, evaluate=False)  # keep both sides of 3·4 = 13
        return value.func(*args)  # already canonical: evaluating again changes nothing
    if isinstance(value, list | tuple):
        return type(value)(map(_thaw, value))
    return value


# ---------------------------------------------------------------- the worker


class _Worker:
    """One reusable worker process; restarted after a timeout."""

    def __init__(self) -> None:
        self._lock = threading.Lock()  # one call at a time: calls share one pipe
        self._process: BaseProcess | None = None
        self._connection: Connection | None = None

    def run(self, function: Callable[..., Any], args: tuple[Any, ...], seconds: float) -> Any:
        with self._lock:
            connection = self._start()
            connection.send((function, args))
            if not connection.poll(seconds):
                self._stop()  # the only way to interrupt it
                raise CheckTimeout(f"no answer within {seconds:g} s")
            try:
                succeeded, result = connection.recv()
            except (EOFError, OSError) as error:  # it crashed, e.g. ran out of memory
                self._stop()
                raise CheckAborted("the worker process stopped unexpectedly") from error
        if not succeeded:
            raise result
        return _thaw(result)

    def _start(self) -> Connection:
        if self._process is None or not self._process.is_alive() or self._connection is None:
            # "spawn" behaves the same on macOS, Linux and Windows, and is safe to use
            # from a process with threads (fork isn't).
            context = multiprocessing.get_context("spawn")
            ours, theirs = context.Pipe()
            self._process = context.Process(target=_serve, args=(theirs,), daemon=True)
            self._process.start()
            self._connection = ours
        return self._connection

    def _stop(self) -> None:
        if self._process is not None:
            self._process.kill()
            self._process.join()
        self._process = self._connection = None


_WORKER = _Worker()
