# logging_util.py

from loguru import logger
import sys


def configure_logger(log_file="app.log"):
    """
    Configures the logger with custom format and colors.

    :param log_file: Path to the log file (default is "app.log").
    """
    # Define a custom format string with explicit colorization
    custom_format = (
        "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
        "{level.icon} {level.name:<8} | "
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

    # Configure logger with custom format and colors
    logger.configure(handlers=[
        {
            "sink": log_file,
            "format": custom_format,
            "colorize": False,  # Disable colorization for file output
            "enqueue": True
        },
        {
            "sink": sys.stdout,
            "format": (
                "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
                "<level>{level.icon} {level.name:<8}</level> | "
                "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
                "<level>{message}</level>"
            ),
            "colorize": True,  # Enable colorization for console output
            "enqueue": True,
        }
    ])

    return logger


# Initialize the logger
logger = configure_logger()
logger.level("DEBUG")


if __name__ == "__main__":
    logger.info("This is an info message.")