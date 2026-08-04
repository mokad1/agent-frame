"""结构化日志。"""

from __future__ import annotations

import logging
import sys

from agent_frame.config import config


def setup_logging(level: str | None = None) -> None:
    lvl = level or config.log_level
    root = logging.getLogger("agent_frame")
    root.setLevel(getattr(logging, lvl.upper(), logging.INFO))
    root.handlers.clear()
    h = logging.StreamHandler(sys.stdout)
    h.setFormatter(logging.Formatter("%(asctime)s [%(levelname)-7s] %(name)s: %(message)s"))
    root.addHandler(h)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(f"agent_frame.{name}")
