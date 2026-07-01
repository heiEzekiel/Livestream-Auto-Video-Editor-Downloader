import logging
from pathlib import Path


def setup_logger(
    log_file: Path,
    level=logging.INFO
) -> logging.Logger:
    """
    Creates a logger that logs to both console and a file.
    """

    log_file.parent.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("sermon_pipeline")
    logger.setLevel(level)

    # Prevent duplicate handlers if called multiple times
    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # File handler
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(formatter)
    file_handler.setLevel(level)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    console_handler.setLevel(level)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger
