# Contributing to Athyr Python SDK

## Development Setup

```bash
# Clone the repository
git clone https://github.com/athyr-tech/athyr-sdk-python.git
cd athyr-sdk-python

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install with dev dependencies
pip install -e ".[dev]"
```

## Code Quality

Run checks before submitting:

```bash
# Format and lint
ruff check --fix .
ruff format .

# Type checking
mypy athyr

# Tests
pytest
```

## Project Structure

```
athyr/
├── __init__.py          # Public API
├── client.py            # Agent client
├── models.py            # Pydantic models
├── exceptions.py        # Error types
├── tools.py             # Tool decorator
├── kv.py                # KV storage
├── middleware.py        # Middleware system
├── _internal/           # Internal modules
│   ├── connection.py    # Connection management
│   ├── llm.py           # LLM operations
│   ├── messaging.py     # Pub/sub operations
│   └── sessions.py      # Session operations
└── orchestration/       # Multi-agent patterns
```

## Submitting Changes

1. Fork the repository
2. Create a feature branch
3. Make changes with tests
4. Run code quality checks
5. Submit a pull request

## Generating gRPC Stubs

If proto files change:

```bash
python -m grpc_tools.protoc \
    -I../athyr/api/v1 \
    --python_out=athyr/_grpc \
    --pyi_out=athyr/_grpc \
    --grpc_python_out=athyr/_grpc \
    ../athyr/api/v1/athyr.proto
```

## Questions?

Open an issue on GitHub.
