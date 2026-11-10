"""Pydantic models, one module per concern.

* :mod:`stepcheck.models.solution`: a worked solution and its lines (parsed LaTeX).
* :mod:`stepcheck.models.verdicts`: what the checkers conclude about each line.

Later milestones add their own modules here (LLM tool inputs and outputs, eval
records) and re-export the public models below.
"""

from stepcheck.models.solution import Line, Solution
from stepcheck.models.verdicts import CheckReport, Finding, LineCheck, Method, Verdict

__all__ = ["CheckReport", "Finding", "Line", "LineCheck", "Method", "Solution", "Verdict"]
