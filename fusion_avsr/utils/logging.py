"""Shared logging setup.

Defines the project's logging format in one place, so every module logs
consistently instead of each reaching for `print()` or configuring its
own `logging.basicConfig`. Every other module should get its logger via
`get_logger(__name__)`, the same way it would call
`logging.getLogger(__name__)` directly -- the only difference is this
also guarantees the format/level are configured the first time it's
called anywhere in the process, regardless of import order.
"""

from __future__ import annotations

import logging

_LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

_root_configured = False


def _configure_root_logger(level: int) -> None:
    """Configure the root logger's format/level exactly once per process."""
    global _root_configured
    if _root_configured:
        return
    logging.basicConfig(format=_LOG_FORMAT, datefmt=_DATE_FORMAT, level=level)
    _root_configured = True


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Return a logger configured with this project's standard format.

    Args:
        name: Logger name, conventionally `__name__` of the calling
            module.
        level: Logging level for the root logger, applied the first time
            this function is called anywhere in the process (later calls
            with a different `level` have no effect on the already-set
            root level).

    Returns:
        A standard library `logging.Logger` instance.
    """
    _configure_root_logger(level)
    return logging.getLogger(name)
