"""Prometheus request/error-rate instrumentation for the Text-Eval service.

Adds a /metrics endpoint (scraped by the AXL-DevOps generic-service Helm
chart's Prometheus ServiceMonitor) so internal API calls to this service are
visible, not just the traffic that passes through the Traefik gateway.

All tunable settings live in config/metrics.yaml, not in this file - see
that file for what can be changed without a code edit.
"""
import os

import yaml
from prometheus_fastapi_instrumentator import Instrumentator, metrics

_DEFAULT_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config", "metrics.yaml")

# Fallback values used only if config/metrics.yaml is missing or a key is
# omitted, so a bad/partial config can never crash startup or silently
# disable metrics.
_DEFAULTS = {
    "enabled": True,
    "endpoint": "/metrics",
    "include_in_schema": False,
    "group_status_codes": True,
    "excluded_handlers": ["/metrics", "/api/docs", "/api/openapi.json"],
    "metric_namespace": "all",
    "metric_subsystem": "text_eval",
}


def _load_config(config_path: str = _DEFAULT_CONFIG_PATH) -> dict:
    config = dict(_DEFAULTS)
    try:
        with open(config_path, "r") as f:
            loaded = yaml.safe_load(f) or {}
        config.update(loaded)
    except FileNotFoundError:
        pass
    return config


def setup_metrics(app, config_path: str = _DEFAULT_CONFIG_PATH) -> None:
    """Attach Prometheus request-count/latency/error-rate instrumentation
    and expose it on the configured endpoint. Call once, right after the
    FastAPI app is created.
    """
    config = _load_config(config_path)
    if not config.get("enabled", True):
        return

    instrumentator = Instrumentator(
        should_group_status_codes=config.get("group_status_codes", True),
        excluded_handlers=config.get("excluded_handlers", []),
    )
    instrumentator.add(
        metrics.default(
            metric_namespace=config.get("metric_namespace", ""),
            metric_subsystem=config.get("metric_subsystem", ""),
        )
    )
    instrumentator.instrument(app).expose(
        app,
        endpoint=config.get("endpoint", "/metrics"),
        include_in_schema=config.get("include_in_schema", False),
    )
