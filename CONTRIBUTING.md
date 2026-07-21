# Contributing to doc-search-lite

Thank you for considering contributing! This document outlines the guidelines.

## Code of Conduct

Please be respectful and constructive in all interactions.

## Getting Started

```bash
# Fork and clone
git clone https://github.com/your-username/doc-search-lite.git
cd doc-search-lite

# Set up virtual environment
python -m venv .venv
.venv\Scripts\pip install -e ".[dev]"

# Copy env config
copy .env.example .env
# Edit .env, set GLM_API_KEY=your-key
```

## Development Workflow

1. Create a branch: `git checkout -b feature/your-feature`
2. Make your changes
3. Run lint and tests:

```bash
.venv\Scripts\ruff check src/ tests/
.venv\Scripts\python -m pytest tests/ -q --tb=short
```

4. Commit and push, then open a Pull Request

## Code Style

- Follow existing patterns in the codebase
- Use **Result-based error handling**: `ConvertResult(success, errors)` and `ToolResult.ok()/.fail()`
- Use **lazy imports** for optional dependencies (`_get_X()` pattern)
- Use **fail-safe design**: exceptions should not block the caller
- Use `from src.xxx` imports (package name is `src`)
- Keep functions focused and small

## Testing

- Add tests for new functionality
- Existing test patterns to follow:
  - `mock_config` fixture from `tests/conftest.py` for Config mocking
  - `MagicMock(spec=...)` for type-safe mocks
  - `tmp_path` for temporary files
  - Class-based test grouping (`class TestXxx:`)

## Documentation

- Update `AGENTS.md` if changing module behavior or adding new features
- Update `README.md` and `README.zh.md` for user-facing changes
- Add docstrings to public API functions

## Pull Request Guidelines

- Keep PRs focused on a single change
- Write clear commit messages
- Reference any related issues
- Ensure CI passes (ruff + pytest)

## Questions?

Open an issue for discussion before starting significant work.
