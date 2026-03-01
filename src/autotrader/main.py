"""CLI entry point for the Kalshi autotrader."""

from __future__ import annotations

import click

from autotrader import __version__


@click.group()
@click.version_option(version=__version__, prog_name="kalshi-autotrader")
def cli() -> None:
    """Kalshi Autotrader — Automated trading for AI prediction markets."""


@cli.command()
@click.option("--config-dir", default="config", help="Path to configuration directory.")
@click.option(
    "--environment",
    type=click.Choice(["demo", "production"]),
    default=None,
    help="Override environment (default: from config).",
)
def run(config_dir: str, environment: str | None) -> None:
    """Start the autotrader."""
    import asyncio
    import signal as signal_mod

    from autotrader.config.loader import load_config
    from autotrader.core.loop import TradingLoop
    from autotrader.monitoring.logging import setup_logging
    from autotrader.state.database import create_db_engine, get_session_factory, init_db

    config = load_config(config_dir=config_dir, environment=environment)
    setup_logging(
        level=config.logging.level,
        json_output=config.logging.json_output,
        log_dir=config.logging.log_dir,
    )

    from autotrader.monitoring.logging import get_logger

    log = get_logger("autotrader.main")
    effective_environment = config.kalshi.environment.value
    effective_base_url = config.kalshi.base_url

    log.info(
        "autotrader_starting",
        version=__version__,
        environment=effective_environment,
    )
    log.info(
        "runtime_mode_resolved",
        mode=effective_environment,
        api_base_url=effective_base_url,
    )

    # Initialize database
    db_engine = create_db_engine(url=config.database.url, echo=config.database.echo)
    init_db(db_engine)
    session_factory = get_session_factory(db_engine)
    log.info("database_initialized", url=config.database.url)

    click.echo(f"Kalshi Autotrader v{__version__}")
    click.echo(f"Environment: {effective_environment}")
    click.echo(f"Kalshi API base URL: {effective_base_url}")
    click.echo(f"Database: {config.database.url}")

    async def _run() -> None:
        trading_loop = TradingLoop(config)
        await trading_loop.initialize(session_factory=session_factory)

        # Graceful shutdown on SIGINT / SIGTERM
        loop = asyncio.get_running_loop()
        for sig in (signal_mod.SIGINT, signal_mod.SIGTERM):
            loop.add_signal_handler(sig, trading_loop.stop)

        click.echo("Autotrader running. Press Ctrl+C to stop.")
        try:
            await trading_loop.run()
        finally:
            await trading_loop.shutdown()
            click.echo("Autotrader stopped.")

    asyncio.run(_run())


@cli.command()
@click.option("--config-dir", default="config", help="Path to configuration directory.")
def validate_config(config_dir: str) -> None:
    """Validate configuration files without starting the system."""
    from autotrader.config.loader import load_config

    try:
        config = load_config(config_dir=config_dir)
        click.echo("Configuration is valid.")
        click.echo(f"  Environment: {config.kalshi.environment.value}")
        click.echo(f"  Database: {config.database.url}")
        click.echo(f"  Arena poll interval: {config.arena_monitor.poll_interval_seconds}s")
        click.echo(f"  Risk max portfolio exposure: {config.risk.global_config.max_portfolio_exposure_pct:.0%}")
        click.echo(f"  Strategy target series: {config.leaderboard_alpha.target_series}")
    except Exception as e:
        click.echo(f"Configuration error: {e}", err=True)
        raise SystemExit(1) from e


@cli.command()
@click.option("--config-dir", default="config", help="Path to configuration directory.")
def init_db_cmd(config_dir: str) -> None:
    """Initialize the database (create tables)."""
    from autotrader.config.loader import load_config
    from autotrader.state.database import create_db_engine, init_db

    config = load_config(config_dir=config_dir)
    engine = create_db_engine(url=config.database.url, echo=config.database.echo)
    init_db(engine)
    click.echo(f"Database initialized at {config.database.url}")


@cli.command()
@click.argument("price_cents", type=int)
@click.argument("quantity", type=int)
@click.option("--maker", is_flag=True, help="Calculate maker fees instead of taker.")
def calc_fee(price_cents: int, quantity: int, maker: bool) -> None:
    """Calculate trading fees for a given price and quantity."""
    from autotrader.utils.fees import FeeCalculator

    calc = FeeCalculator()
    if maker:
        result = calc.maker_fee(price_cents, quantity)
        fee_type = "Maker"
    else:
        result = calc.taker_fee(price_cents, quantity)
        fee_type = "Taker"

    click.echo(f"{fee_type} fee calculation:")
    click.echo(f"  Price: {price_cents}¢ × {quantity} contracts")
    click.echo(f"  Fee per contract: {result.fee_per_contract_cents}¢")
    click.echo(f"  Total fee: {result.total_fee_cents}¢")
    click.echo(f"  Fee as % of price: {result.fee_as_pct_of_price}%")
    click.echo(f"  Effective cost/contract: {result.effective_cost_cents}¢")


@cli.command()
@click.option("--config-dir", default="config", help="Path to configuration directory.")
@click.option("--days", default=7, show_default=True, help="Number of days to include.")
@click.option("--strategy", default=None, help="Filter by strategy name.")
@click.option("--csv", "as_csv", is_flag=True, help="Output as CSV.")
def pnl(config_dir: str, days: int, strategy: str | None, as_csv: bool) -> None:
    """Show P&L report from the trading database."""
    from autotrader.config.loader import load_config
    from autotrader.state.database import create_db_engine, get_session_factory
    from autotrader.state.repository import TradingRepository

    config = load_config(config_dir=config_dir)
    engine = create_db_engine(url=config.database.url, echo=False)
    session_factory = get_session_factory(engine)
    repo = TradingRepository(session_factory)

    rows = repo.get_pnl_history(strategy=strategy, days=days)

    if not rows:
        click.echo("No P&L data found.")
        return

    if as_csv:
        click.echo("date,strategy,realized_pnl_cents,unrealized_pnl_cents,fees_cents,trades,is_paper")
        for row in rows:
            date_str = row.date.strftime("%Y-%m-%d") if row.date else ""
            click.echo(
                f"{date_str},{row.strategy},{row.realized_pnl_cents},"
                f"{row.unrealized_pnl_cents},{row.total_fees_cents},"
                f"{row.trade_count},{row.is_paper}"
            )
        return

    # Table output
    click.echo(f"P&L Report (last {days} days)")
    if strategy:
        click.echo(f"Strategy: {strategy}")
    click.echo()

    header = f"{'Date':<12} {'Strategy':<20} {'Realized':>10} {'Unrealized':>10} {'Fees':>8} {'Trades':>7} {'Mode':<6}"
    click.echo(header)
    click.echo("-" * len(header))

    total_realized = 0
    total_unrealized = 0
    total_fees = 0
    total_trades = 0

    for row in rows:
        date_str = row.date.strftime("%Y-%m-%d") if row.date else "N/A"
        mode = "paper" if row.is_paper else "live"
        realized = row.realized_pnl_cents
        unrealized = row.unrealized_pnl_cents
        fees = row.total_fees_cents

        click.echo(
            f"{date_str:<12} {row.strategy:<20} {realized:>9}¢ {unrealized:>9}¢ {fees:>7}¢ {row.trade_count:>7} {mode:<6}"
        )

        total_realized += realized
        total_unrealized += unrealized
        total_fees += fees
        total_trades += row.trade_count

    click.echo("-" * len(header))
    click.echo(
        f"{'TOTAL':<12} {'':<20} {total_realized:>9}¢ {total_unrealized:>9}¢ "
        f"{total_fees:>7}¢ {total_trades:>7}"
    )


@cli.command()
@click.option("--config-dir", default="config", help="Path to configuration directory.")
@click.option(
    "--environment",
    type=click.Choice(["demo", "production"]),
    default=None,
    help="Override environment.",
)
def preflight(config_dir: str, environment: str | None) -> None:
    """Run connectivity checks without starting the trading loop."""
    import os

    from autotrader.config.loader import load_config

    checks_passed = 0
    checks_failed = 0

    def _pass(name: str, detail: str = "") -> None:
        nonlocal checks_passed
        checks_passed += 1
        msg = f"  PASS  {name}"
        if detail:
            msg += f" — {detail}"
        click.echo(msg)

    def _fail(name: str, detail: str = "") -> None:
        nonlocal checks_failed
        checks_failed += 1
        msg = f"  FAIL  {name}"
        if detail:
            msg += f" — {detail}"
        click.echo(msg)

    click.echo("Preflight checks")
    click.echo()

    # 1. Configuration
    try:
        config = load_config(config_dir=config_dir, environment=environment)
        _pass("Configuration", f"environment={config.kalshi.environment.value}")
    except Exception as e:
        _fail("Configuration", str(e))
        click.echo(f"\n{checks_passed} passed, {checks_failed} failed")
        raise SystemExit(1) from e

    # 2. Database
    try:
        from sqlalchemy import text

        from autotrader.state.database import create_db_engine, init_db

        db_engine = create_db_engine(url=config.database.url, echo=False)
        init_db(db_engine)
        with db_engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        _pass("Database", config.database.url)
    except Exception as e:
        _fail("Database", str(e))

    # 3. Kalshi API
    try:
        from autotrader.api.client import KalshiAPIClient

        client = KalshiAPIClient(config.kalshi)
        private_key_pem = os.environ.get("KALSHI_PRIVATE_KEY_PEM")
        client.connect(private_key_pem=private_key_pem)
        status = client.get_exchange_status()
        _pass("Kalshi API", f"exchange_active={status.exchange_active}")
    except Exception as e:
        _fail("Kalshi API", str(e))

    # 4. Arena monitor
    try:
        import httpx

        resp = httpx.get(
            config.arena_monitor.primary_url,
            timeout=config.arena_monitor.request_timeout_seconds,
            follow_redirects=True,
        )
        resp.raise_for_status()
        _pass("Arena monitor", f"status={resp.status_code}")
    except Exception as e:
        _fail("Arena monitor", str(e))

    # 5. Discord webhook (if configured)
    if config.discord.enabled and config.discord.webhook_url:
        try:
            import httpx

            # Send a GET to validate the webhook URL (Discord returns webhook info)
            resp = httpx.get(config.discord.webhook_url, timeout=10)
            if resp.status_code < 400:
                _pass("Discord webhook", "reachable")
            else:
                _fail("Discord webhook", f"status={resp.status_code}")
        except Exception as e:
            _fail("Discord webhook", str(e))
    else:
        _pass("Discord webhook", "not configured (skipped)")

    click.echo()
    click.echo(f"{checks_passed} passed, {checks_failed} failed")
    if checks_failed > 0:
        raise SystemExit(1)


@cli.command()
@click.option("--config-dir", default="config", help="Path to configuration directory.")
@click.argument("signals_file", type=click.Path(exists=True))
@click.option("--market-data", type=click.Path(exists=True), default=None, help="JSON file with market data.")
def replay(config_dir: str, signals_file: str, market_data: str | None) -> None:
    """Replay historical signals through the strategy for backtesting."""
    import asyncio
    import json as json_mod

    from autotrader.backtest.replay import ReplayEngine
    from autotrader.config.loader import load_config

    config = load_config(config_dir=config_dir)
    engine = ReplayEngine(config)

    md: dict | None = None
    if market_data:
        with open(market_data) as f:
            md = json_mod.load(f)

    result = asyncio.run(engine.run(signals_path=signals_file, market_data=md))

    click.echo("Replay Results")
    click.echo(f"  Signals processed:  {result.total_signals}")
    click.echo(f"  Proposals generated: {result.total_proposals}")
    click.echo(f"  Risk approved:      {result.total_approved}")
    click.echo(f"  Risk rejected:      {result.total_rejected}")
    click.echo(f"  Fills:              {result.total_fills}")
    click.echo(f"  Total fees:         {result.total_fees_cents}¢")
    click.echo(f"  Realized P&L:       {result.realized_pnl_cents}¢")
    if result.positions:
        click.echo("  Open positions:")
        for ticker, qty in result.positions.items():
            click.echo(f"    {ticker}: {qty}")


if __name__ == "__main__":
    cli()
