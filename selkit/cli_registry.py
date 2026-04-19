from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True)
class CommandSpec:
    group: str
    sub: str | None
    build_parser: Callable[[argparse.ArgumentParser], None]
    handle: Callable[[argparse.Namespace], int]


def _codeml_site_models_builder(p: argparse.ArgumentParser) -> None:
    p.add_argument("--alignment", required=False)
    p.add_argument("--alignment-dir", required=False)
    p.add_argument("--tree", required=True)
    p.add_argument("--output", required=True, dest="output_dir")
    p.add_argument("--threads", type=int, default=1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-starts", type=int, default=3)
    p.add_argument("--genetic-code", default="standard")
    p.add_argument("--foreground", default=None)
    p.add_argument("--foreground-tips", default=None)
    p.add_argument("--labels-file", default=None)
    p.add_argument("--tests", default=None)
    p.add_argument("--models", default=None)
    p.add_argument("--strip-stop-codons", action="store_true")
    p.add_argument("--no-strip-terminal-stop", action="store_true")
    p.add_argument("--allow-unconverged", action="store_true")
    p.add_argument("--config", default=None)


def _validate_builder(p: argparse.ArgumentParser) -> None:
    p.add_argument("--alignment", required=True)
    p.add_argument("--tree", required=True)
    p.add_argument("--genetic-code", default="standard")
    p.add_argument("--foreground", default=None)
    p.add_argument("--foreground-tips", default=None)
    p.add_argument("--labels-file", default=None)
    p.add_argument("--strip-stop-codons", action="store_true")
    p.add_argument("--no-strip-terminal-stop", action="store_true")


def _rerun_builder(p: argparse.ArgumentParser) -> None:
    p.add_argument("config", help="path to run.yaml")
    p.add_argument("--output", required=False, dest="output_dir")


def _codeml_site_models_handle(ns: argparse.Namespace) -> int:
    from selkit.cli import handle_codeml_site_models
    return handle_codeml_site_models(ns)


def _validate_handle(ns: argparse.Namespace) -> int:
    from selkit.cli import handle_validate
    return handle_validate(ns)


def _rerun_handle(ns: argparse.Namespace) -> int:
    from selkit.cli import handle_rerun
    return handle_rerun(ns)


CLI_COMMANDS: tuple[CommandSpec, ...] = (
    CommandSpec(
        group="codeml", sub="site-models",
        build_parser=_codeml_site_models_builder,
        handle=_codeml_site_models_handle,
    ),
    CommandSpec(
        group="validate", sub=None,
        build_parser=_validate_builder,
        handle=_validate_handle,
    ),
    CommandSpec(
        group="rerun", sub=None,
        build_parser=_rerun_builder,
        handle=_rerun_handle,
    ),
)


def build_argparser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(prog="selkit")
    subs = root.add_subparsers(dest="group", required=True)
    codeml = subs.add_parser("codeml")
    codeml_subs = codeml.add_subparsers(dest="sub", required=True)
    for cmd in CLI_COMMANDS:
        if cmd.group == "codeml":
            p = codeml_subs.add_parser(cmd.sub or "_default")
            cmd.build_parser(p)
            p.set_defaults(_cmd=cmd)
        else:
            p = subs.add_parser(cmd.group)
            cmd.build_parser(p)
            p.set_defaults(_cmd=cmd)
    return root
