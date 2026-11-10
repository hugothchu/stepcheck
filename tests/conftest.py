"""Shared test configuration.

Hypothesis profiles: the default keeps `pytest` quick; run a deeper search with
`HYPOTHESIS_PROFILE=thorough uv run pytest tests/test_properties.py`.
"""

import os

from hypothesis import HealthCheck, settings

# Parsing takes milliseconds, but simplify() on a random expression can be slow.
_COMMON = {"deadline": None, "suppress_health_check": [HealthCheck.too_slow]}
settings.register_profile("default", max_examples=150, **_COMMON)
settings.register_profile("thorough", max_examples=3000, **_COMMON)
settings.load_profile(os.getenv("HYPOTHESIS_PROFILE", "default"))
