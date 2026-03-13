from __future__ import annotations

import logging
from pathlib import Path


def configure_logging(logs_dir: Path, run_name: str, verbose: bool = True) -> logging.Logger:
    logs_dir.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("rag_sgu")
    logger.setLevel(logging.INFO)
    logger.propagate = False

    for handler in list(logger.handlers):
        logger.removeHandler(handler)

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    if verbose:
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

    log_file = logs_dir / f"rag_{run_name}.log"
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger
