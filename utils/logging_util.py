# logging_util.py
import os

from loguru import logger
import sys
import threading

# Get the absolute path of the current file
CURRENT_FILE_PATH = os.path.abspath(__file__)
# Get the directory containing the current file
BASE_DIR = os.path.dirname(CURRENT_FILE_PATH)

# Thread-local storage for context
_thread_local = threading.local()


def set_context(key: str, value: str):
    """Set a value in the logging context"""
    if not hasattr(_thread_local, 'context'):
        _thread_local.context = {}
    _thread_local.context[key] = value


def clear_context():
    """Clear all context values"""
    if hasattr(_thread_local, 'context'):
        del _thread_local.context


def get_context():
    """Get the current context dictionary"""
    return getattr(_thread_local, 'context', {})


class ContextFilter:
    def __call__(self, record):
        context = get_context()
        record["extra"].update(context)
        return True


def configure_logger(log_file="app.log", max_bytes=10 * 1024 * 1024, backup_count=5):
    """Configure logger with context support"""
    logger.remove()

    # File output format
    logger.add(
        sink=log_file,
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "{level.icon} {level.name:<8} | "
            "<blue>{thread.name}</blue> | "
            "<blue>{process}</blue> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
            "context:{extra} - "
            "{message}"
        ),
        filter=ContextFilter(),
        colorize=False,
        enqueue=True,
        rotation=max_bytes,
        retention=backup_count
    )

    # Console output format
    logger.add(
        sink=sys.stdout,
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level.icon} {level.name:<8}</level> | "
            "<blue>{thread.name}</blue> | "
            "<blue>{process}</blue> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
            "context:{extra} - "
            "<level>{message}</level>"
        ),
        filter=ContextFilter(),
        colorize=True,
        enqueue=True,
    )

    return logger


# Initialize the logger
logger = configure_logger(os.path.join(BASE_DIR, "../app.log"))
logger.level("DEBUG")

if __name__ == "__main__":
    logger.info("This is an info message.")
