"""Unit tests for the metrics module."""

from __future__ import annotations

import json
import threading
import urllib.request

from autotrader.monitoring.metrics import (
    Metrics,
    start_metrics_server,
    stop_metrics_server,
)


class TestMetrics:
    def test_increment_counter(self) -> None:
        m = Metrics()
        m.increment("test_counter")
        m.increment("test_counter", 5)
        snap = m.snapshot()
        assert snap["counters"]["test_counter"] == 6

    def test_set_gauge(self) -> None:
        m = Metrics()
        m.set_gauge("test_gauge", 42.5)
        snap = m.snapshot()
        assert snap["gauges"]["test_gauge"] == 42.5

    def test_set_gauge_overwrites(self) -> None:
        m = Metrics()
        m.set_gauge("g", 1.0)
        m.set_gauge("g", 2.0)
        assert m.snapshot()["gauges"]["g"] == 2.0

    def test_snapshot_includes_uptime(self) -> None:
        m = Metrics()
        snap = m.snapshot()
        assert "uptime_seconds" in snap
        assert snap["uptime_seconds"] >= 0

    def test_empty_snapshot(self) -> None:
        m = Metrics()
        snap = m.snapshot()
        assert snap["counters"] == {}
        assert snap["gauges"] == {}

    def test_log_summary_does_not_crash(self) -> None:
        m = Metrics()
        m.increment("x")
        m.set_gauge("y", 1.0)
        m.log_summary()  # Should not raise

    def test_thread_safety(self) -> None:
        m = Metrics()
        errors: list[str] = []

        def worker() -> None:
            try:
                for _ in range(100):
                    m.increment("thread_counter")
                    m.set_gauge("thread_gauge", 1.0)
            except Exception as e:
                errors.append(str(e))

        threads = [threading.Thread(target=worker) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        assert m.snapshot()["counters"]["thread_counter"] == 400


class TestMetricsServer:
    def test_start_stop_server(self) -> None:
        start_metrics_server(port=19876)
        try:
            resp = urllib.request.urlopen("http://127.0.0.1:19876/metrics", timeout=2)
            data = json.loads(resp.read())
            assert "counters" in data
            assert "gauges" in data
        finally:
            stop_metrics_server()

    def test_health_endpoint(self) -> None:
        start_metrics_server(port=19877)
        try:
            resp = urllib.request.urlopen("http://127.0.0.1:19877/health", timeout=2)
            data = json.loads(resp.read())
            assert data["status"] == "ok"
        finally:
            stop_metrics_server()

    def test_404_on_unknown_path(self) -> None:
        start_metrics_server(port=19878)
        try:
            try:
                urllib.request.urlopen("http://127.0.0.1:19878/unknown", timeout=2)
                assert False, "Expected 404"  # noqa: B011
            except urllib.error.HTTPError as e:
                assert e.code == 404
        finally:
            stop_metrics_server()
