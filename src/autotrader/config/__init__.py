"""Configuration management for the autotrader."""

from autotrader.config.loader import load_config
from autotrader.config.models import AppConfig

__all__ = ["AppConfig", "load_config"]
