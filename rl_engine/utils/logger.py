# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

import logging
import sys
from functools import lru_cache
from types import MethodType
from typing import Any, cast

_DEFAULT_FORMAT = "%(levelname)s %(asctime)s [%(name)s]: %(message)s"
_DATE_FORMAT = "%m-%d %H:%M:%S"


@lru_cache(maxsize=None)
def _log_once_impl(logger: logging.Logger, level: int, msg: str, *args: Any) -> None:
    """Implementation for one-time logging using LRU cache."""
    logger.log(level, msg, *args, stacklevel=3)


def _info_once(self: logging.Logger, msg: str, *args: Any) -> None:
    """Log INFO message only once across the same logger instance."""
    _log_once_impl(self, logging.INFO, msg, *args)


def _warn_once(self: logging.Logger, msg: str, *args: Any) -> None:
    """Log WARNING message only once."""
    _log_once_impl(self, logging.WARNING, msg, *args)


def _info_on_rank(self: logging.Logger, msg: str, rank: int = 0, *args: Any) -> None:
    """
    Experimental: Log only on a specific distributed rank.
    Useful for multi-node RL training to avoid log flooding.
    """
    from rl_engine.platforms.device import device_ctx

    if getattr(device_ctx, "rank", 0) == rank:
        self.info(msg, *args)


class RLEngineLogger(logging.Logger):
    """
    Type-hinting stub for patched methods.
    Methods are actually injected at runtime via init_logger.
    """

    def info_once(self, msg: str, *args: Any) -> None: ...
    def warn_once(self, msg: str, *args: Any) -> None: ...
    def info_on_rank(self, msg: str, rank: int = 0, *args: Any) -> None: ...


def init_logger(name: str) -> RLEngineLogger:
    """
    Initializes logger and patches extended methods.
    Ensures consistent logging style across the entire engine.
    """
    logger = logging.getLogger(name)

    # Configure handler if not already set
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(_DEFAULT_FORMAT, datefmt=_DATE_FORMAT)
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.propagate = False

    logger.info_once = MethodType(_info_once, logger)  # type: ignore[attr-defined]
    logger.warn_once = MethodType(_warn_once, logger)  # type: ignore[attr-defined]
    logger.info_on_rank = MethodType(_info_on_rank, logger)  # type: ignore[attr-defined]

    return cast(RLEngineLogger, logger)


logger = init_logger("RL-Kernel")
