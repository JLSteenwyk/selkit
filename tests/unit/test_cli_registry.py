from __future__ import annotations

import argparse

import pytest

from selkit.cli_registry import CLI_COMMANDS, CommandSpec


def test_registry_contains_expected_groups() -> None:
    names = {c.group for c in CLI_COMMANDS}
    assert "codeml" in names
    assert "validate" in names
    assert "rerun" in names


def test_command_spec_has_parser_builder() -> None:
    for cmd in CLI_COMMANDS:
        assert callable(cmd.build_parser)
        assert callable(cmd.handle)
