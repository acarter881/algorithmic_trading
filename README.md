# Kalshi Autotrader

Automated trading system for Kalshi prediction markets, focused on AI model leaderboard contracts (KXTOPMODEL, KXLLM1).

## Overview

This system monitors the [LMSYS Chatbot Arena](https://arena.ai) leaderboard and trades Kalshi prediction market contracts based on detected price dislocations. The primary edge comes from reacting faster and more accurately than manual traders when leaderboard data changes.

**Warning:** Automated trading carries real financial risk. This system defaults to demo/paper trading mode. Do not connect to production with real capital without reviewing performance data.

## Data Source

The autotrader monitors the **LMSYS Chatbot Arena** leaderboard using the **Text, Overall, No Style Control** ranking category. The `rank_stylectrl` column corresponds to `Rank (UB)` — the resolution metric for the KXTOPMODEL contract on Kalshi.

| Source | URL | Type |
|--------|-----|------|
| Primary | `https://raw.githubusercontent.com/fboulnois/llm-leaderboard-csv/main/csv/lmarena_text.csv` | CSV (updated regularly) |
| Fallback 1 | `https://arena.ai/leaderboard/text/overall-no-style-control` | HTML |
| Fallback 2 | `https://lmarena.ai/leaderboard/text/overall-no-style-control` | HTML |

The primary source is the [fboulnois/llm-leaderboard-csv](https://github.com/fboulnois/llm-leaderboard-csv) repository, which provides clean CSV data. The HTML fallbacks are used if the CSV source is unavailable.

Polling interval is **30 seconds** by default (configurable in `config/signal_sources/arena_monitor.yaml`).

## Quick Start (Docker)

Docker is the recommended way to run the autotrader.

### Prerequisites

- [Docker Desktop](https://www.docker.com/products/docker-desktop/) installed
- Kalshi API key ID and RSA private key (`.pem` file)
- (Optional) Discord webhook URL for trade alerts

### Setup

```bash
# 1. Copy .env.example and fill in your credentials
cp .env.example .env

# 2. Place your Kalshi private key in the project root
cp /path/to/your/private_key.pem ./kalshi_private_key.pem

# 3. Build and start (paper trading by default)
docker compose up -d --build
```

### Monitoring

```powershell
# Check container status
docker compose ps

# View recent logs
docker compose logs --tail 20

# Follow logs in real time
docker compose logs -f

# Stop the autotrader
docker compose down

# Restart after a config change
docker compose up -d --build
```

The Docker container runs in the background — you do not need to keep Docker Desktop open. Discord webhook notifications will alert you to trades and errors in real time.

### What the logs mean

| Log message | Meaning |
|-------------|---------|
| `arena_leaderboard_fetched` | Successfully pulled latest leaderboard data |
| `parsed_html_table` | Parsed leaderboard into structured data |
| `tick_no_signals` | Evaluated data, no actionable trade this tick |
| `signal_generated` | A trading signal was detected |

## Quick Start (Local)

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

### Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `AUTOTRADER__KALSHI__API_KEY_ID` | Kalshi API key ID | Yes |
| `AUTOTRADER__KALSHI__PRIVATE_KEY_PATH` | Path to RSA private key `.pem` file | Yes |
| `AUTOTRADER__KALSHI__ENVIRONMENT` | `demo` (paper) or `production` (live) | No (default: `demo`) |
| `AUTOTRADER__DISCORD__WEBHOOK_URL` | Discord webhook for alerts | No |
| `AUTOTRADER__DISCORD__ENABLED` | Enable Discord notifications | No |
| `AUTOTRADER__DATABASE__URL` | SQLite database path | No |
| `AUTOTRADER__LOGGING__LEVEL` | Log level (`DEBUG`, `INFO`, etc.) | No |

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
