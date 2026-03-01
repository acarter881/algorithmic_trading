"""In-process metrics for observability.

Tracks counters and gauges that are:
- Logged periodically via structlog
- Exposed as JSON on an optional HTTP endpoint (``/metrics``)

If ``prometheus_client`` is installed, the same metrics are also registered
as Prometheus gauges/counters and served on ``/prometheus`` in OpenMetrics
format.  This is entirely optional — the module works without it.
"""

from __future__ import annotations

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any

import structlog

logger = structlog.get_logger("autotrader.monitoring.metrics")

# Try to import prometheus_client for optional Prometheus exposition
try:
    from prometheus_client import Counter as PromCounter
    from prometheus_client import Gauge as PromGauge
    from prometheus_client import generate_latest

    _PROM_AVAILABLE = True
except ImportError:
    _PROM_AVAILABLE = False


class Metrics:
    """Thread-safe in-memory metrics store."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._counters: dict[str, int] = {}
        self._gauges: dict[str, float] = {}
        self._start_time = time.monotonic()

        # Optional Prometheus metrics
        self._prom_counters: dict[str, Any] = {}
        self._prom_gauges: dict[str, Any] = {}

    def increment(self, name: str, value: int = 1) -> None:
        """Increment a counter by *value*."""
        with self._lock:
            self._counters[name] = self._counters.get(name, 0) + value
        if _PROM_AVAILABLE:
            pc = self._prom_counters.get(name)
            if pc is None:
                pc = PromCounter(name, f"Counter: {name}")
                self._prom_counters[name] = pc
            pc.inc(value)

    def set_gauge(self, name: str, value: float) -> None:
        """Set a gauge to an absolute value."""
        with self._lock:
            self._gauges[name] = value
        if _PROM_AVAILABLE:
            pg = self._prom_gauges.get(name)
            if pg is None:
                pg = PromGauge(name, f"Gauge: {name}")
                self._prom_gauges[name] = pg
            pg.set(value)

    def snapshot(self) -> dict[str, Any]:
        """Return a point-in-time snapshot of all metrics."""
        with self._lock:
            return {
                "counters": dict(self._counters),
                "gauges": dict(self._gauges),
                "uptime_seconds": round(time.monotonic() - self._start_time, 1),
            }

    def log_summary(self) -> None:
        """Emit a structured log with all current metrics."""
        snap = self.snapshot()
        logger.info("metrics_summary", **snap)


# Singleton instance — import this from other modules.
metrics = Metrics()


# ── HTTP metrics endpoint ────────────────────────────────────────────────


class _MetricsHandler(BaseHTTPRequestHandler):
    """Serves /metrics (JSON) and /prometheus (OpenMetrics) endpoints."""

    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/metrics":
            body = json.dumps(metrics.snapshot(), indent=2).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif self.path == "/prometheus" and _PROM_AVAILABLE:
            body = generate_latest()
            self.send_response(200)
            self.send_header("Content-Type", "text/plain; version=0.0.4; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif self.path == "/health":
            body = b'{"status":"ok"}'
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
        """Suppress default HTTP server logging — use structlog instead."""


_server: HTTPServer | None = None
_server_thread: threading.Thread | None = None


def start_metrics_server(port: int = 9090) -> None:
    """Start the metrics HTTP server in a background daemon thread."""
    global _server, _server_thread  # noqa: PLW0603
    if _server is not None:
        return  # already running
    _server = HTTPServer(("0.0.0.0", port), _MetricsHandler)
    _server_thread = threading.Thread(target=_server.serve_forever, daemon=True)
    _server_thread.start()
    logger.info("metrics_server_started", port=port, prometheus=_PROM_AVAILABLE)


def stop_metrics_server() -> None:
    """Stop the metrics HTTP server."""
    global _server, _server_thread  # noqa: PLW0603
    if _server is not None:
        _server.shutdown()
        _server = None
        _server_thread = None
        logger.info("metrics_server_stopped")
