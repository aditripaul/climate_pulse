"""Shared test setup: make `src/` importable and reset cached settings."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


@pytest.fixture
def fresh_settings(monkeypatch):
    """Yield a function that rebuilds settings after env changes, then restore them."""
    from config import get_settings

    def rebuild(**env):
        for key, value in env.items():
            monkeypatch.setenv(key, str(value))
        get_settings.cache_clear()
        return get_settings()

    yield rebuild
    monkeypatch.undo()
    get_settings.cache_clear()
