"""Maintainer entry point; T09 extends rl_engine.p3's existing runner."""

import sys

from rl_engine.p3.__main__ import main

if __name__ == "__main__":
    raise SystemExit(main(["check_p3", *sys.argv[1:]]))
