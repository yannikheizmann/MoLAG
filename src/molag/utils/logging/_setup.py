import logging
import sys


def setup_logging(level: int = logging.INFO) -> None:
    """Configure the root logger for a command-line workflow."""
    logging.basicConfig(
        level=level,
        format="[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout,
        force=True,
    )
