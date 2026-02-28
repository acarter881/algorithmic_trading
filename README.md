# Kalshi Autotrader

Automated trading system for Kalshi prediction markets, focused on AI model leaderboard contracts (KXTOPMODEL, KXLLM1).

## Overview

This system monitors the [LMSYS Chatbot Arena](https://arena.ai) leaderboard and trades Kalshi prediction market contracts based on detected price dislocations. The primary edge comes from reacting faster and more accurately than manual traders when leaderboard data changes.

**Warning:** Automated trading carries real financial risk. This system defaults to demo/paper trading mode. Do not connect to production with real capital without reviewing performance data.

## Data Source

The autotrader monitors the **LMSYS Chatbot Arena** live leaderboard, specifically the **Text, Overall, No Style Control** category:

> **https://arena.ai/leaderboard/text/overall-no-style-control**

`Rank (UB)` from this page is the primary resolution input for KXTOPMODEL, with deterministic tie-break rules mirrored in `resolve_top_model`.

| Source | URL |
|--------|-----|
| Primary | `https://arena.ai/leaderboard/text/overall-no-style-control` |
| Fallback | `https://lmarena.ai/leaderboard/text/overall-no-style-control` |

Both URLs point to the same live leaderboard data. The fallback is used automatically if the primary is unreachable.

Polling interval is **30 seconds** by default (configurable in `config/signal_sources/arena_monitor.yaml`).

### Resolution Rules

Winner selection is implemented in `src/autotrader/signals/settlement.py::resolve_top_model` and follows this exact sort cascade:

1. `rank_ub` ascending (lower is better)
2. `score` descending (higher is better)
3. `votes` descending (higher is better)
4. `release_date` ascending (earlier date wins)
5. `model_name` ascending (lexicographic final tie-break)

Notes:
- Invalid/non-positive `rank_ub` values are treated as very large (sorted to the bottom).
- Missing `release_date` values are treated as very late dates (sorted to the bottom within ties).

This exact winner logic directly affects:
- `new_leader` generation in `src/autotrader/signals/arena_monitor.py` (leader-change detection uses `resolve_top_model` on previous vs current snapshots).
- Fair-value biasing in `src/autotrader/strategies/leaderboard_alpha.py::estimate_fair_value` (a model currently winning this cascade gets a small probability uplift when tie-break inputs are complete).

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

### Switching from demo to production (exact procedure)

> **Safety first:** production mode places real-money orders. Keep demo mode enabled until you have reviewed strategy behavior and risk limits.

1. **Confirm required production credentials are present in `.env`:**
   - `AUTOTRADER__KALSHI__API_KEY_ID` (a valid production API key ID)
   - `AUTOTRADER__KALSHI__PRIVATE_KEY_PATH` (or Docker-mounted `/keys/kalshi_private_key.pem`)
   - `AUTOTRADER__KALSHI__ENVIRONMENT=production`
2. **Verify private key mount and path alignment:**
   - In Docker Compose, the key is mounted as `./kalshi_private_key.pem:/keys/kalshi_private_key.pem:ro`.
   - Ensure `AUTOTRADER__KALSHI__PRIVATE_KEY_PATH=/keys/kalshi_private_key.pem`.
3. **Run a preflight sanity check before starting trading:**
   ```bash
   docker compose config
   docker compose run --rm autotrader validate-config --config-dir config
   ```
4. **Start/restart the service in production mode:**
   ```bash
   docker compose up -d --build
   ```
5. **Confirm production mode in startup logs (must match exactly):**
   ```bash
   docker compose logs --tail 100 autotrader
   ```
   Required confirmation log fields:
   - `runtime_mode_resolved` with `mode=production` and `api_base_url=https://api.elections.kalshi.com/trade-api/v2`
   - `kalshi_client_connected` with `environment=production` and `base_url=https://api.elections.kalshi.com/trade-api/v2`

If either log line still reports `demo` or the demo API host, stop immediately with `docker compose down` and fix `.env` before resuming.

### What the logs mean

| Log message | Meaning |
|-------------|---------|
| `arena_leaderboard_fetched` | Successfully pulled latest leaderboard data |
| `parsed_html_table` | Parsed leaderboard into structured data |
| `tick_no_signals` | Evaluated data, no actionable trade this tick |
| `signal_generated` | A trading signal was detected |

## Reviewing Trades

All orders, fills, positions, and daily P&L are recorded to a local SQLite database (`autotrader.db`). This is created automatically on first run — no database setup is needed in `.env`.

Your primary view of trade activity is the **Kalshi platform itself**:

- **Demo**: Log in at [demo.kalshi.com](https://demo.kalshi.com) to see paper-trade positions, fills, and P&L.
- **Production**: Log in at [kalshi.com](https://kalshi.com) to see live positions, fills, and P&L.

The two relevant event pages on Kalshi:
- **KXTOPMODEL** — per-model contracts on which AI model will be ranked #1
- **KXLLM1** — per-organization contracts on which AI org will lead

If Discord alerts are configured, you will also receive real-time notifications for trades, signals, and errors.

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
| KXTOPMODEL | Top AI model using the full `resolve_top_model` tie-break cascade | Per-model contracts |
| KXLLM1 | Top AI organization derived from the resolved top model winner | Per-organization contracts |

## Paper-Trading Hardening Checklist

Use this checklist before promoting from demo/paper trading to production:

- **Data quality**
  - Arena fetch success rate is stable across primary/fallback URLs.
  - Parsed leaderboard fields (`rank_ub`, `score`, `votes`, `release_date`) are consistently populated and sane.
  - Snapshot cadence and staleness monitoring are in place (no silent multi-poll data gaps).
- **Signal quality**
  - `new_leader`, `ranking_change`, and `score_shift` signals are manually spot-checked against raw leaderboard moves.
  - `new_leader` events match settlement tie-break outcomes from `resolve_top_model`.
  - False-positive/duplicate signal rates are reviewed from logs over multi-day runs.
- **Risk limits**
  - Position sizing, max exposure, and per-market caps are validated under paper fills.
  - Fee-aware edge checks remain positive after realistic spread/slippage assumptions.
  - Kill-switch / safe-disable procedures are tested and documented.
- **Drift checks**
  - Fair-value vs market-price residuals are tracked to detect model edge decay.
  - Pairwise and rank-based probability assumptions are reviewed on a recurring schedule.
  - Strategy behavior is revalidated after parser, signal, or tie-break logic changes.

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
