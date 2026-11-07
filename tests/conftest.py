"""Shared fixtures. GPU and network tests are skipped unless the machine can run them."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from intensity.config import Config, load_config

FIXTURES = Path(__file__).parent / "fixtures"


def _has_cuda() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    cuda = _has_cuda()
    net = os.environ.get("INTENSITY_NET") == "1"
    for item in items:
        if "gpu" in item.keywords and not cuda:
            item.add_marker(pytest.mark.skip(reason="needs an NVIDIA GPU"))
        if "net" in item.keywords and not net:
            item.add_marker(pytest.mark.skip(reason="needs network; set INTENSITY_NET=1"))


@pytest.fixture
def config() -> Config:
    """Default config with no overlays, the heuristic forecast and polish off: deterministic."""
    return load_config(overrides=["engine.forecast.source=heuristic", "engine.polish.enabled=false"])


@pytest.fixture
def envelope_fixtures() -> dict[str, dict]:
    return {p.stem: json.loads(p.read_text()) for p in sorted((FIXTURES / "envelopes").glob("*.json"))}
