# Claude Code Instructions

## Before every commit

Always run the linter and formatter before committing:

```bash
ruff format src/ tests/ && ruff check --fix src/ tests/
```

This project enforces `ruff` formatting and lint rules (including SIM, UP, B, etc.) in CI. Commits that fail these checks will not pass.
