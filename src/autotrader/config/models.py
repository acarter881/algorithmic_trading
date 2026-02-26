"""Pydantic configuration models for the autotrader system."""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field, field_validator


class Environment(StrEnum):
    """Trading environment."""

    DEMO = "demo"
    PRODUCTION = "production"


class KalshiConfig(BaseModel):
    """Kalshi API connection settings."""

    environment: Environment = Environment.DEMO
    api_key_id: str = ""
    private_key_path: str = ""
    demo_base_url: str = "https://demo-api.kalshi.co/trade-api/v2"
    production_base_url: str = "https://api.elections.kalshi.com/trade-api/v2"
    websocket_url: str = "wss://api.elections.kalshi.com/trade-api/ws/v2"
    request_timeout_seconds: float = 30.0
    max_retries: int = 3
    rate_limit_buffer_pct: float = 0.1

    @property
    def base_url(self) -> str:
        if self.environment == Environment.DEMO:
            return self.demo_base_url
        return self.production_base_url


class ArenaMonitorConfig(BaseModel):
    """Arena leaderboard monitor settings."""

    primary_url: str = (
        "https://raw.githubusercontent.com/fboulnois/llm-leaderboard-csv/main/csv/lmarena_text.csv"
    )
    fallback_urls: list[str] = Field(
        default_factory=lambda: [
            "https://arena.ai/leaderboard/text/overall-no-style-control",
            "https://lmarena.ai/leaderboard/text/overall-no-style-control",
        ]
    )
    poll_interval_seconds: int = 30
    request_timeout_seconds: float = 15.0
    max_consecutive_failures: int = 5


class RiskGlobalConfig(BaseModel):
    """Global risk parameters."""

    max_portfolio_exposure_pct: float = Field(default=0.60, ge=0.0, le=1.0)
    max_daily_loss_pct: float = Field(default=0.05, ge=0.0, le=1.0)
    kill_switch_enabled: bool = False
    reconciliation_interval_seconds: int = 300


class RiskStrategyConfig(BaseModel):
    """Per-strategy risk parameters."""

    max_position_per_contract: float = 100.00
    max_position_per_event: float = 250.00
    max_strategy_loss: float = 200.00
    min_edge_multiplier: float = 2.5


class RiskConfig(BaseModel):
    """Risk management configuration."""

    global_config: RiskGlobalConfig = Field(default_factory=RiskGlobalConfig, alias="global")
    per_strategy: dict[str, RiskStrategyConfig] = Field(default_factory=dict)

    model_config = {"populate_by_name": True}


class LeaderboardAlphaConfig(BaseModel):
    """Leaderboard alpha strategy parameters."""

    min_edge_after_fees_cents: int = 3
    elo_shift_threshold: int = 3
    rank_spread_change_threshold: int = 2
    time_decay_acceleration_hours: int = 24
    max_position_per_contract: float = 100.00
    max_position_per_event: float = 250.00
    preliminary_model_discount: float = Field(default=0.3, ge=0.0, le=1.0)
    fuzzy_match_threshold: float = Field(default=0.8, ge=0.0, le=1.0)
    model_name_overrides: dict[str, str] = Field(default_factory=dict)
    target_series: list[str] = Field(default_factory=lambda: ["KXTOPMODEL", "KXLLM1"])


class DiscordConfig(BaseModel):
    """Discord alerting configuration."""

    webhook_url: str = ""
    enabled: bool = False
    alert_on_trades: bool = True
    alert_on_signals: bool = True
    alert_on_errors: bool = True
    large_trade_threshold: float = 50.00


class DatabaseConfig(BaseModel):
    """Database configuration."""

    url: str = "sqlite:///autotrader.db"
    echo: bool = False


class LoggingConfig(BaseModel):
    """Logging configuration."""

    level: str = "INFO"
    json_output: bool = True
    log_dir: str = "logs"
    rotate_days: int = 90

    @field_validator("level")
    @classmethod
    def validate_level(cls, v: str) -> str:
        valid = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        upper = v.upper()
        if upper not in valid:
            raise ValueError(f"Invalid log level: {v}. Must be one of {valid}")
        return upper


class AppConfig(BaseModel):
    """Top-level application configuration."""

    kalshi: KalshiConfig = Field(default_factory=KalshiConfig)
    arena_monitor: ArenaMonitorConfig = Field(default_factory=ArenaMonitorConfig)
    risk: RiskConfig = Field(default_factory=RiskConfig)
    leaderboard_alpha: LeaderboardAlphaConfig = Field(default_factory=LeaderboardAlphaConfig)
    discord: DiscordConfig = Field(default_factory=DiscordConfig)
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AppConfig:
        return cls.model_validate(data)
