"""Pydantic models, one module per concern.

* :mod:`stepcheck.models.solution`: a worked solution and its lines (parsed LaTeX).

Later milestones add their own modules here (checker verdicts, LLM tool inputs and
outputs, eval records) and re-export the public models below.
"""

from stepcheck.models.solution import Line, Solution

__all__ = ["Line", "Solution"]
