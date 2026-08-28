"""Shared fixtures.

`cli_args` builds argparse.Namespace objects through the REAL parser, so test
argument sets can never drift from the CLI's actual flags and defaults.
"""
from __future__ import annotations

import pytest

from cleancut import cache
from cleancut.config import Config


@pytest.fixture(autouse=True)
def isolated_cache(tmp_path_factory, monkeypatch):
    """Never let tests read or write the user's real ~/.cache/cleancut."""
    monkeypatch.setattr(cache, "CACHE_DIR", tmp_path_factory.mktemp("cleancut_cache"))
    yield


@pytest.fixture
def default_config() -> Config:
    return Config.load_defaults()


@pytest.fixture
def cli_args():
    """Parse real CLI argv into a Namespace: cli_args("scan", video, "--no-whisper")."""
    from cleancut.cli import build_parser

    def make(*argv: str, **overrides):
        ns = build_parser().parse_args([str(a) for a in argv])
        for k, v in overrides.items():
            setattr(ns, k, v)
        return ns

    return make
