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


def test_codeml_site_has_beb_flags() -> None:
    from selkit.cli_registry import build_argparser
    p = build_argparser()
    ns = p.parse_args([
        "codeml", "site",
        "--alignment", "x.fa", "--tree", "y.nwk", "--output", "/tmp/out",
        "--no-beb", "--beb-grid", "25",
    ])
    assert ns.beb is False
    assert ns.beb_grid == 25


def test_codeml_branch_site_has_beb_flags() -> None:
    from selkit.cli_registry import build_argparser
    p = build_argparser()
    ns = p.parse_args([
        "codeml", "branch-site",
        "--alignment", "x.fa", "--tree", "y.nwk", "--output", "/tmp/out",
        "--foreground", "a,b",
        "--beb-grid", "15",
    ])
    assert ns.beb is True        # default True (no --no-beb)
    assert ns.beb_grid == 15


def test_codeml_branch_has_beb_flags_as_noop() -> None:
    """branch family has no BEB models; flags parse but are no-op."""
    import pytest
    pytest.importorskip("selkit.services.codeml.branch_models")
    from selkit.cli_registry import build_argparser
    p = build_argparser()
    ns = p.parse_args([
        "codeml", "branch",
        "--alignment", "x.fa", "--tree", "y.nwk", "--output", "/tmp/out",
        "--no-beb",
    ])
    assert ns.beb is False
