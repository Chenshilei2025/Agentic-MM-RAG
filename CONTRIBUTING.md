# Contributing

Thanks for improving Agentic MM-RAG. Keep changes small, tested, and aligned
with the public tool contracts.

## Development Setup

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -e ".[dev]"
```

Optional integrations are installed separately:

```bash
python -m pip install -e ".[openai]"
```

Document and video ingestion require heavier upstream dependencies and model
artifacts. See `data/README.md` before enabling those paths.

## Checks

Run these before submitting changes:

```bash
python -m pytest
python -m py_compile $(find . -path './data/vendor' -prune -o -name '*.py' -print)
```

## Design Constraints

- Keep the public tool names stable unless the change is explicitly breaking.
- Prefer provider and backend interfaces over hard-coded service clients.
- Do not commit processed corpora, model checkpoints, API keys, or private data.
- Preserve source attribution for third-party code.
