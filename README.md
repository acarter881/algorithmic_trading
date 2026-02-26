# Kalshi Autotrader

Automated trading system for Kalshi prediction markets, focused on AI model leaderboard contracts (KXTOPMODEL, KXLLM1).

## Overview

This system monitors the [LMSYS Chatbot Arena](https://arena.ai) leaderboard and trades Kalshi prediction market contracts based on detected price dislocations. The primary edge comes from reacting faster and more accurately than manual traders when leaderboard data changes.

**Warning:** Automated trading carries real financial risk. This system defaults to demo/paper trading mode. Do not connect to production with real capital without reviewing performance data.

## Quick Start

```bash
# Install dependencies
pip install -e ".[dev]"

# Validate configuration
autotrader validate-config --config-dir config

# Initialize database
autotrader init-db-cmd --config-dir config

# Calculate fees for a trade
autotrader calc-fee 50 10

# Run (paper trading)
autotrader run --config-dir config
```

## Configuration

Configuration is loaded in layers:
1. `config/base.yaml` — defaults
2. `config/paper.yaml` or `config/live.yaml` — environment overrides
3. `config/strategies/*.yaml` — strategy parameters
4. `config/risk.yaml` — risk limits
5. Environment variables (`AUTOTRADER__SECTION__KEY`)

Copy `.env.example` to `.env` and fill in your Kalshi API credentials.

## Target Markets

| Series | Description | Resolution Metric |
|--------|-------------|-------------------|
| KXTOPMODEL | Top AI model by Rank (UB) | Per-model contracts |
| KXLLM1 | Best AI org by Rank (UB) | Per-organization contracts |

## Development

```bash
# Run tests
pytest tests/unit/ -v

# Lint
ruff check src/ tests/

# Type check
mypy src/autotrader/
```

## Project Structure

```
src/autotrader/       # Main package
  config/             # Pydantic config models and YAML loader
  api/                # Kalshi API client wrapper
  signals/            # Signal source plugins (Arena monitor, etc.)
  strategies/         # Trading strategies
  risk/               # Risk management
  execution/          # Order execution engine
  state/              # Database models and persistence
  monitoring/         # Logging and Discord alerts
  utils/              # Fee calculator, fuzzy matching
tests/                # Unit and integration tests
config/               # YAML configuration files
scripts/              # Utility scripts
```
