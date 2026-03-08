from __future__ import annotations

import logging
from dataclasses import dataclass


@dataclass(frozen=True)
class LoggingConfig:
    """Configuration for application logging."""

    name: str = "scalable_rqa_volatility"
    level: int = logging.INFO


def get_logger(cfg: LoggingConfig | None = None) -> logging.Logger:
    """Return a configured logger with a consistent formatter."""
    cfg = cfg or LoggingConfig()
    logger = logging.getLogger(cfg.name)
    logger.setLevel(cfg.level)

    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setLevel(cfg.level)
        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.propagate = False

    return logger