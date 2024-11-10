# logging_util.py

from loguru import logger
import sys


def configure_logger(log_file="app.log", max_bytes=10 * 1024 * 1024, backup_count=5):
    """
    Configures the logger with custom format, colors, rolling, and max bytes.

    :param log_file: Path to the log file (default is "app.log").
    :param max_bytes: Maximum size of the log file before it rolls over (default is 10 MB).
    :param backup_count: Number of backup log files to keep (default is 5).
    """
    # Define a custom format string with explicit colorization
    custom_format = (
        "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
        "{level.icon} {level.name:<8} | "
        "<blue>{thread.name}</blue> | "
        "<blue>{process}</blue> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
        "{message}"
    )

    # Define a custom colorizer function
    def custom_colorizer(record):
        level_color = {
            "TRACE": "magenta",
            "DEBUG": "blue",
            "INFO": "green",
            "SUCCESS": "bold_green",
            "WARNING": "yellow",
            "ERROR": "red",
            "CRITICAL": "bold_red"
        }
        return level_color.get(record["level"].name, "")

    logger.remove()
    logger.add(sink=log_file, format=custom_format, colorize=False, enqueue=True, rotation=max_bytes,
               retention=backup_count)

    logger.add(
        sink=sys.stdout,
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level.icon} {level.name:<8}</level> | "
            "<blue>{thread.name}</blue> | "
            "<blue>{process}</blue> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
            "<level>{message}</level>"
        ),
        colorize=True,  # Enable colorization for console output
        enqueue=True,
    )

    return logger


# Initialize the logger
logger = configure_logger()
logger.level("DEBUG")

if __name__ == "__main__":
    logger.info("This is an info message.")
