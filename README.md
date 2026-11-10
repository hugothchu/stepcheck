# stepcheck

Check every step of a worked maths solution with SymPy.

LLMs write fluent maths that is often wrong. stepcheck parses a solution written in
LaTeX, checks each line with a computer algebra system, and reports exactly where the
reasoning breaks.

> 🚧 Work in progress. The plan and design decisions are in [docs/plan.md](docs/plan.md).

## Development

```bash
uv sync
uv run pytest        # tests + doctests
HYPOTHESIS_PROFILE=thorough uv run pytest tests/test_properties.py   # deeper search
uv run ruff check .
uv run pyright
```
