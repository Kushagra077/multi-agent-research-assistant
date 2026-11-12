# Logging setup. Call setup_logging() once, from the entry point (script or notebook).
# Library modules only create loggers: logger = logging.getLogger(__name__).

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

LOG_DIR = Path("logs")
LOG_FILE = LOG_DIR / "logfile.log"
LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"

MAX_LOG_BYTES = 1_000_000
BACKUP_COUNT = 3

# Libraries that log every request at INFO/DEBUG. Keep them at WARNING so a single
# run does not bury our own lines.
NOISY_LIBRARIES = ("httpx", "httpcore", "urllib3")


def setup_logging(level: int = logging.INFO) -> None:
    root = logging.getLogger()
    root.setLevel(level)

    # Calling this twice (for example from a notebook cell) must not stack handlers,
    # or every line would be written twice.
    for handler in root.handlers[:]:
        root.removeHandler(handler)
        handler.close()

    LOG_DIR.mkdir(exist_ok=True)
    formatter = logging.Formatter(LOG_FORMAT)

    file_handler = RotatingFileHandler(
        LOG_FILE,
        maxBytes=MAX_LOG_BYTES,
        backupCount=BACKUP_COUNT,
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    root.addHandler(file_handler)
    root.addHandler(console_handler)

    for name in NOISY_LIBRARIES:
        logging.getLogger(name).setLevel(logging.WARNING)
