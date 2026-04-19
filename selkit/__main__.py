from __future__ import annotations

import sys
from typing import Sequence


def main(argv: Sequence[str] | None = None) -> int:
    from selkit.cli_registry import build_argparser
    parser = build_argparser()
    args = parser.parse_args(argv)
    return args._cmd.handle(args)


if __name__ == "__main__":
    sys.exit(main())
