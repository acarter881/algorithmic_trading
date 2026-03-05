# Kalshi Autotrader

Automated trading system for Kalshi prediction markets, focused on AI model leaderboard contracts (KXTOPMODEL, KXLLM1).

## Overview

This system monitors the [LMSYS Chatbot Arena](https://arena.ai) leaderboard and trades Kalshi prediction market contracts based on detected price dislocations. The primary edge comes from reacting faster and more accurately than manual traders when leaderboard data changes.

**Warning:** Automated trading carries real financial risk. This system defaults to paper execution mode (simulated fills, no real orders). Do not switch to live execution without reviewing performance data from an extended paper trading run.

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
- Persisted `leaderboard_snapshots.snapshot_data.entries` include tie-break/audit fields (`votes`, `release_date`) for settlement replay and verification.

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
```

**Required `.env` settings for paper trading:**
```
AUTOTRADER__KALSHI__API_KEY_ID=<your API key ID>
AUTOTRADER__KALSHI__PRIVATE_KEY_PATH=/keys/kalshi_private_key.pem
AUTOTRADER__KALSHI__EXECUTION_MODE=paper
```

> Paper trading uses the production Kalshi API (real market data, real prices) but simulates fills locally — no orders are sent to the exchange.

```bash
# 3. Build and start (paper trading)
docker compose up -d --build
```

### Development Loop (Docker-only)

To iterate faster (lint, preflight, and fast tests) use:

```bash
./scripts/docker_dev_loop.sh --iterations 3
```

Defaults:
- runs Docker image build once
- runs `ruff format --check` + `ruff check`
- runs `preflight` (including a target-series market-availability check)
- runs fast tests (`pytest -m "not integration"`)
- writes logs to `reports/docker-dev-loop-<timestamp>.log`

Optional flags:

```bash
# Include full test suite after fast tests
./scripts/docker_dev_loop.sh --iterations 1 --full-tests

# Use alternate config directory
./scripts/docker_dev_loop.sh --config-dir config --iterations 2
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

By default in Docker, state is persisted under `/data` in the container:

- Paper execution mode DB: `sqlite:////data/autotrader_paper.db`
- Live execution mode DB: `sqlite:////data/autotrader_live.db`

### Paper trading startup checklist

Before running 24/7, walk through these steps once:

1. **Verify `.env` credentials and mode:**
   ```
   AUTOTRADER__KALSHI__API_KEY_ID=<your key>
   AUTOTRADER__KALSHI__PRIVATE_KEY_PATH=/keys/kalshi_private_key.pem
   AUTOTRADER__KALSHI__EXECUTION_MODE=paper
   ```
2. **Verify private key mount:**
   - Docker Compose mounts `./kalshi_private_key.pem:/keys/kalshi_private_key.pem:ro`.
   - The `PRIVATE_KEY_PATH` in `.env` must match the container path (`/keys/kalshi_private_key.pem`).
3. **Run preflight to validate all connections:**
   ```bash
   docker compose run --rm autotrader preflight --config-dir config
   ```
   This checks: config validity, DB init, Kalshi API authentication, Arena leaderboard reachability, required target-series market availability, and Discord webhook (if configured). All checks should pass before continuing.
4. **Start the service:**
   ```bash
   docker compose up -d --build
   ```
5. **Confirm startup logs show the correct mode:**
   ```bash
   docker compose logs --tail 100 autotrader
   ```
   Required confirmation:
   - `runtime_mode_resolved` with `execution_mode=paper` and `api_base_url=https://api.elections.kalshi.com/trade-api/v2`
   - `kalshi_client_connected` with `base_url=https://api.elections.kalshi.com/trade-api/v2`
   - execution mode = `paper`
   - database URL = `sqlite:////data/autotrader_paper.db`

   If any field is unexpected, stop immediately with `docker compose down` and fix `.env`.

6. **Monitor the first few hours** — you should see:
   - `arena_leaderboard_fetched` every ~30 seconds (healthy polling)
   - `tick_no_signals` on quiet ticks (no leaderboard changes — this is normal)
   - On any actual leaderboard change: `signal_generated` → `proposals_generated` → `paper_order_filled` or `order_rejected`

7. **Run paper trading for days to weeks** before considering live execution. Use the hardening checklist below to track readiness.

### Switching from paper to live execution

> **Safety first:** only change `EXECUTION_MODE` after an extended paper trading run. Consider starting with reduced position limits.

1. **Review paper trading results:**
   ```bash
   docker compose exec autotrader autotrader pnl --config-dir config --days 30
   ```
   Cross-reference fills in the local DB against trades shown on [kalshi.com](https://kalshi.com) (your production account).

2. **Optionally reduce risk limits** in `config/risk.yaml` for the initial live period:
   ```yaml
   per_strategy:
     leaderboard_alpha:
       max_position_per_contract: 10   # Start small
       max_position_per_event: 25
   ```

3. **Switch execution mode in `.env`:**
   ```
   AUTOTRADER__KALSHI__EXECUTION_MODE=live
   ```

4. **Restart and verify:**
   ```bash
   docker compose up -d --build
   docker compose logs --tail 100 autotrader
   ```
   Confirm: execution mode = `live`, database URL = `sqlite:////data/autotrader_live.db`.

5. **Monitor intensely for the first day.** Scale limits back up gradually as confidence builds.

### What the logs mean

| Log message | Meaning |
|-------------|---------|
| `arena_leaderboard_fetched` | Successfully pulled latest leaderboard data |
| `parsed_html_table` | Parsed leaderboard into structured data |
| `tick_no_signals` | Evaluated data, no actionable trade this tick |
| `signal_generated` | A trading signal was detected |

## Reviewing Trades

All orders, fills, positions, and daily P&L are recorded to SQLite. The default DB URL depends on how you run the service:

- **Docker paper mode:** `sqlite:////data/autotrader_paper.db`
- **Docker live mode:** `sqlite:////data/autotrader_live.db`
- **Local non-Docker CLI runs:** from YAML defaults (`config/base.yaml`, overlaid by `config/paper.yaml` or `config/live.yaml`):
  - base: `sqlite:///autotrader.db`
  - paper overlay: `sqlite:///autotrader_paper.db`
  - live overlay: `sqlite:///autotrader_live.db`

**Paper mode:** Paper fills are simulated locally — no orders are sent to Kalshi, so paper trades will **not** appear on the Kalshi website. Review paper trades via the local SQLite database or the `autotrader pnl` CLI command.

**Live mode:** Real orders are sent to Kalshi. Log in at [kalshi.com](https://kalshi.com) to see live positions, fills, and P&L. The two relevant event pages:
- **KXTOPMODEL** — per-model contracts on which AI model will be ranked #1
- **KXLLM1** — per-organization contracts on which AI org will lead

If Discord alerts are configured, you will also receive real-time notifications for trades, signals, and errors in both modes.

### Where is my DB?

Use these commands to locate and inspect persisted data in Docker:

```bash
# List compose volumes (look for this project's data volume)
docker volume ls

# Inspect the resolved volume mount path for the autotrader service
docker compose config

# Open a shell in the running container and inspect /data
docker compose exec autotrader sh -lc 'ls -lah /data && pwd'

# Optional: inspect the SQLite DB directly
docker compose exec autotrader sh -lc 'sqlite3 /data/autotrader_paper.db ".tables"'
```

If you run locally (non-Docker), the SQLite files are created relative to your current working directory unless you set an absolute URL.

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

Configuration is loaded in layers (later values override earlier):
1. `config/base.yaml` — defaults
2. `config/paper.yaml` or `config/live.yaml` — execution mode overlay (`paper` → `paper.yaml`, `live` → `live.yaml`)
3. `config/strategies/*.yaml` — strategy parameters
4. `config/risk.yaml` — risk limits
5. `config/signal_sources/*.yaml` — signal source parameters
6. Environment variables (`AUTOTRADER__SECTION__KEY`) — **highest precedence, always wins**

Copy `.env.example` to `.env` and fill in your Kalshi API credentials.

### Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `AUTOTRADER__KALSHI__API_KEY_ID` | Kalshi API key ID | Yes |
| `AUTOTRADER__KALSHI__PRIVATE_KEY_PATH` | Path to RSA private key `.pem` file | Yes |
| `AUTOTRADER__KALSHI__EXECUTION_MODE` | `paper` (simulated fills) or `live` (real orders) | No (default: `paper`) |
| `AUTOTRADER__KALSHI__WEBSOCKET_ENABLED` | Enable real-time WebSocket market data instead of REST polling | No (default: `false`) |
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

Use this checklist before switching from paper execution to live execution:

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

## CLI Commands

| Command | Description |
|---------|-------------|
| `autotrader run` | Start the trading loop (paper or live, based on config) |
| `autotrader preflight` | Run connectivity checks (API, DB, Arena, Discord) without trading |
| `autotrader validate-config` | Validate configuration files and print resolved settings |
| `autotrader init-db-cmd` | Create database tables without starting the loop |
| `autotrader pnl` | Show P&L report from the trading database (`--days 7`, `--csv`) |
| `autotrader calc-fee <price> <qty>` | Calculate taker/maker fees for a trade |
| `autotrader replay <signals.json>` | Replay historical signals through the strategy for backtesting |

Most commands accept `--config-dir` (default: `config`). `run` and `preflight` also accept `--execution-mode paper|live`. `calc-fee` is standalone and does not require configuration.

## Backtesting

The replay engine runs historical signals through the full pipeline (strategy → risk → paper execution) without connecting to any external service:

```bash
autotrader replay signals.json --market-data market_data.json --config-dir config
```

`signals.json` format — array of signal objects:
```json
[
  {
    "source": "arena_monitor",
    "event_type": "new_leader",
    "timestamp": "2026-02-15T12:00:00",
    "urgency": "high",
    "data": {"new_leader": "model-name", "previous_leader": "old-model"},
    "relevant_series": ["KXTOPMODEL"]
  }
]
```

Output includes: signals processed, proposals generated, risk approvals/rejections, fills, fees, and open positions.

Paper fills are simulated at the proposed price (no slippage or partial fill modeling). Use replay for signal quality and risk-limit validation, not P&L prediction.

## Risk Management

Every proposed order passes through 6 independent checks before reaching the execution engine. If **any** check fails, the order is rejected and a `RiskEvent` is persisted to the database.

| Check | Default Limit | Description |
|-------|---------------|-------------|
| Kill switch | off | Global circuit breaker — blocks all orders when active |
| Price sanity | 1–99c | Rejects prices outside valid Kalshi range |
| Position per contract | 100 contracts | Max position in a single contract ticker |
| Position per event | 250 contracts | Max aggregate position across an event's contracts |
| Daily loss | $200 | Max combined realized + unrealized loss per strategy per day |
| Portfolio exposure | 60% of balance | Max gross exposure as percentage of account balance |

Limits are configured in `config/risk.yaml`. The kill switch can be activated programmatically or via config (`kill_switch_enabled: true`).

In live mode, positions are reconciled against the exchange every 300 seconds to detect drift.

## WebSocket Streaming

By default the trading loop polls the Kalshi REST API for market data on every tick. Enabling WebSocket streaming switches to a persistent connection that pushes price updates in real time, which lowers latency and reduces REST API usage.

**WebSocket is disabled by default.** Enable it with an environment variable:

```
AUTOTRADER__KALSHI__WEBSOCKET_ENABLED=true
```

Or in `config/base.yaml` / `config/live.yaml`:

```yaml
kalshi:
  websocket_enabled: true
```

When the WebSocket is connected:
- REST polling for market data is skipped (the ticker channel provides updates).
- Mispricing proposals generated by strategies in response to real-time price updates are buffered internally and drained on the next tick, where they pass through the full risk-check and execution pipeline just like REST-sourced proposals.
- If the WebSocket disconnects, the loop automatically falls back to REST polling until the connection is restored.

### When does this matter?

1. **Fast-moving leaderboard shake-up.** A new Arena benchmark run causes model rankings to shift and several KXTOPMODEL contract prices move within seconds. With WebSocket enabled, the strategy sees each price update as it arrives and can generate mispricing proposals immediately — those proposals are queued and executed on the very next tick. Without WebSocket, the loop only sees a single snapshot per REST poll and may miss intermediate price dislocations entirely.

2. **Stale-price arbitrage after a leader change.** A `new_leader` signal fires and the strategy re-estimates fair values. One contract's market price hasn't adjusted yet. Via REST polling the loop might not see the stale price until the next 30-second poll, by which time the market has corrected. With WebSocket, the real-time ticker stream delivers the stale price instantly, the strategy flags the mispricing, and the proposal enters the risk pipeline within the same tick cycle.

## Development

```bash
# Run tests
pytest tests/unit/ -v

# Format and lint (required before committing)
ruff format src/ tests/ && ruff check --fix src/ tests/

# Type check
mypy src/autotrader/
```

## Project Structure

```
src/autotrader/       # Main package
  main.py             # CLI entry point (run, preflight, pnl, replay, etc.)
  config/             # Pydantic config models and YAML loader
  core/               # Trading loop orchestration
  api/                # Kalshi API client + WebSocket streaming
  signals/            # Signal source plugins (Arena monitor, parser, settlement)
  strategies/         # Trading strategies (leaderboard_alpha)
  risk/               # Risk management gate (6 independent checks)
  execution/          # Order execution engine (paper + live modes)
  backtest/           # Replay engine for historical signal backtesting
  state/              # SQLAlchemy models, database, repository
  monitoring/         # Structured logging, Discord alerts, metrics server
  utils/              # Fee calculator, fuzzy matching
tests/                # Unit and integration tests (~536 tests)
config/               # YAML configuration files (layered)
scripts/              # Utility scripts
```
