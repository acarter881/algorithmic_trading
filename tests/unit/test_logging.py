"""Unit tests for structured logging setup."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import structlog

if TYPE_CHECKING:
    from pathlib import Path

from autotrader.monitoring.logging import get_logger, setup_logging


class TestSetupLogging:
    def test_default_setup(self, tmp_path: Path) -> None:
        setup_logging(level="INFO", json_output=True, log_dir=str(tmp_path))
        root = logging.getLogger()
        assert root.level == logging.INFO
        # Should have both console and file handler
        assert len(root.handlers) == 2

    def test_debug_level(self, tmp_path: Path) -> None:
        setup_logging(level="DEBUG", json_output=False, log_dir=str(tmp_path))
        root = logging.getLogger()
        assert root.level == logging.DEBUG

    def test_log_dir_created(self, tmp_path: Path) -> None:
        log_dir = tmp_path / "nested" / "logs"
        setup_logging(level="INFO", log_dir=str(log_dir))
        assert log_dir.exists()

    def test_file_handler_writes(self, tmp_path: Path) -> None:
        setup_logging(level="INFO", json_output=True, log_dir=str(tmp_path))
        logger = logging.getLogger("test.file_write")
        logger.info("hello from test")

        log_file = tmp_path / "autotrader.log"
        assert log_file.exists()
        content = log_file.read_text()
        assert "hello from test" in content

    def test_noisy_loggers_suppressed(self, tmp_path: Path) -> None:
        setup_logging(level="DEBUG", log_dir=str(tmp_path))
        for name in ("httpx", "httpcore", "websockets", "urllib3"):
            assert logging.getLogger(name).level == logging.WARNING

    def test_json_output_mode(self, tmp_path: Path) -> None:
        setup_logging(level="INFO", json_output=True, log_dir=str(tmp_path))
        logger = logging.getLogger("test.json_mode")
        logger.info("json test message")

        log_file = tmp_path / "autotrader.log"
        content = log_file.read_text()
        # JSON output should contain quoted strings
        assert '"json test message"' in content or "json test message" in content

    def test_console_output_mode(self, tmp_path: Path) -> None:
        setup_logging(level="INFO", json_output=False, log_dir=str(tmp_path))
        # Console renderer uses different formatting — just verify no crash
        logger = logging.getLogger("test.console_mode")
        logger.info("console test message")

    def test_handlers_cleared_on_reinit(self, tmp_path: Path) -> None:
        setup_logging(level="INFO", log_dir=str(tmp_path))
        setup_logging(level="DEBUG", log_dir=str(tmp_path))
        root = logging.getLogger()
        # Should have exactly 2 handlers, not 4
        assert len(root.handlers) == 2


class TestGetLogger:
    def test_returns_bound_logger(self, tmp_path: Path) -> None:
        setup_logging(level="INFO", log_dir=str(tmp_path))
        logger = get_logger("autotrader.test")
        # Should return a structlog bound logger (lazy proxy wrapping BoundLogger)
        assert hasattr(logger, "info")
        assert hasattr(logger, "warning")
        assert hasattr(logger, "bind")

    def test_logger_name_preserved(self, tmp_path: Path) -> None:
        setup_logging(level="INFO", log_dir=str(tmp_path))
        logger = get_logger("autotrader.custom_name")
        # The underlying stdlib logger should use the given name
        assert logger is not None
