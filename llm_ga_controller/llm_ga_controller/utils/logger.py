"""
utils/logger.py
===============
Logging configuration for the LLM-GA Meta-Controller.

Sets up two handlers:
  1. Console (stdout)  — INFO level by default; coloured if colorlog is available.
  2. File (logs/run_<timestamp>.log) — DEBUG level, full detail.

Usage
-----
    from utils.logger import setup_logging
    setup_logging()          # call once at program start
    logger = logging.getLogger(__name__)
    logger.info("...")
"""

from __future__ import annotations

import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

from config.settings import LOG_DIR


# ── Colour support (optional dependency) ──────────────────────────────────────

try:
    import colorlog
    _HAS_COLORLOG = True
except ImportError:
    _HAS_COLORLOG = False


# ── Formatters ────────────────────────────────────────────────────────────────

CONSOLE_FMT  = "%(asctime)s  %(levelname)-8s  %(name)s  |  %(message)s"
FILE_FMT     = "%(asctime)s  %(levelname)-8s  %(name)s  |  %(message)s"
DATE_FMT     = "%H:%M:%S"

COLOR_MAP = {
    "DEBUG":    "cyan",
    "INFO":     "green",
    "WARNING":  "yellow",
    "ERROR":    "red",
    "CRITICAL": "bold_red",
}


def _make_console_handler(level: int) -> logging.Handler:
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)
    if _HAS_COLORLOG:
        fmt = colorlog.ColoredFormatter(
            "%(log_color)s" + CONSOLE_FMT,
            datefmt=DATE_FMT,
            log_colors=COLOR_MAP,
        )
    else:
        fmt = logging.Formatter(CONSOLE_FMT, datefmt=DATE_FMT)
    handler.setFormatter(fmt)
    return handler


def _make_file_handler(log_path: Path) -> logging.Handler:
    handler = logging.FileHandler(log_path, encoding="utf-8")
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(logging.Formatter(FILE_FMT, datefmt="%Y-%m-%d %H:%M:%S"))
    return handler


# ── Public setup function ─────────────────────────────────────────────────────

def setup_logging(
    console_level: int = logging.INFO,
    log_dir: str = LOG_DIR,
    run_id: str | None = None,
) -> Path:
    """Initialise root logger with console + file handlers.

    Parameters
    ----------
    console_level : Verbosity for the console handler (logging.INFO / DEBUG).
    log_dir       : Directory where log files are written.
    run_id        : Optional tag appended to the log filename.

    Returns
    -------
    Path of the log file created.
    """
    Path(log_dir).mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    suffix    = f"_{run_id}" if run_id else ""
    log_path  = Path(log_dir) / f"run_{timestamp}{suffix}.log"

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)   # file handler captures everything

    # Avoid adding duplicate handlers if called more than once
    if not root.handlers:
        root.addHandler(_make_console_handler(console_level))
        root.addHandler(_make_file_handler(log_path))

    # Silence noisy third-party loggers
    logging.getLogger("anthropic").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)

    logging.getLogger(__name__).info("Logging initialised → %s", log_path)
    return log_path


# ── Section printer (visual separator in console output) ─────────────────────

def print_section(title: str, width: int = 72) -> None:
    """Print a prominent section header to stdout."""
    bar = "─" * width
    print(f"\n{bar}")
    print(f"  {title}")
    print(bar)


def print_llm_decision(
    generation:      int,
    phase:           str,
    root_cause:      str,
    reasoning:       str,
    expected_outcome: str,
    confidence:      float,
    report_summary:  str,
) -> None:
    """Pretty-print the LLM's decision to the console."""
    print_section(f"LLM Meta-Controller  ·  Generation {generation}")
    print(f"  Phase assessment : {phase}")
    print(f"  Root cause       : {root_cause}")
    print(f"  Reasoning        : {reasoning}")
    print(f"  Expected outcome : {expected_outcome}")
    print(f"  Confidence       : {confidence:.0%}")
    print()
    print("  Parameter changes:")
    print(report_summary if report_summary.strip() else "    None")
    print()


def print_generation_summary(
    generation:   int,
    best_fitness: float,
    avg_fitness:  float,
    diversity:    float,
    mutation_rate: float,
    elapsed:      float,
) -> None:
    """One-line per-generation summary printed to console."""
    print(
        f"  Gen {generation:>4}  |  best={best_fitness:>14,.1f}  "
        f"avg={avg_fitness:>14,.1f}  "
        f"div={diversity:.3f}  μ={mutation_rate:.3f}  "
        f"({elapsed:.1f}s)"
    )
