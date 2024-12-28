# logging_util.py
import os

from loguru import logger
import sys
import threading
from contextvars import ContextVar
from typing import Dict

# Get the absolute path of the current file
CURRENT_FILE_PATH = os.path.abspath(__file__)
# Get the directory containing the current file
BASE_DIR = os.path.dirname(CURRENT_FILE_PATH)

# Thread-local storage for context
_thread_local = threading.local()

# Update the context implementation
_request_context: ContextVar[Dict[str, str]] = ContextVar('request_context', default={})

def set_context(key: str, value: str):
    """Set a value in the logging context"""
    try:
        context = _request_context.get()
        new_context = context.copy()
        new_context[key] = value
        _request_context.set(new_context)
        logger.debug(f"Context set: {key}={value}")  # Debug log
    except Exception as e:
        logger.error(f"Error setting context: {str(e)}")
        _request_context.set({key: value})

def get_context() -> Dict[str, str]:
    """Get the current context dictionary"""
    try:
        return _request_context.get()
    except Exception as e:
        logger.error(f"Error getting context: {str(e)}")
        return {}
def clear_context():
    """lear all context values"""
    try:
       _request_context.set({})
       logger.debug("Context cleared")  # Debug log
    except Exception as e:
       logger.error(f"Error clearing context: {str(e)}")


class SafeContextFilter:
    def __call__(self, record):
        try:
            # Get current context
            context = get_context()
            
            # Update record's extra dict with context
            record["extra"].update(context)
            
            # Add context info to message if present
            # if context:
            #     context_str = ", ".join(f"{k}={v}" for k, v in context.items())
            #     record["message"] = f"[{context_str}] | {record['message']}"

            return True
        except Exception as e:
            logger.error(f"Error in context filter: {str(e)}")
            return True

def configure_logger(log_file="app.log", max_bytes=10 * 1024 * 1024, backup_count=5):
    """Configure logger with context support"""
    logger.remove()

    # Format string
    log_format = (
        "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
        "{level.icon} {level.name:<8} | "
        "<blue>{thread.name}</blue> | "
        "<blue>{process.id}</blue> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
        "{message}"
    )

    # Add file handler
    logger.add(
        sink=log_file,
        format=log_format,
        filter=SafeContextFilter(),
        colorize=False,
        enqueue=True,
        rotation=max_bytes,
        retention=backup_count,
        catch=True,
        level="DEBUG"  # Set to DEBUG temporarily for testing
    )

    # Console handler
    logger.add(
        sink=sys.stdout,
        format=log_format,
        filter=SafeContextFilter(),
        colorize=True,
        enqueue=True,
        catch=True,
        level="DEBUG"  # Set to DEBUG temporarily for testing
    )

    return logger


# Initialize the logger
logger = configure_logger(os.path.join(BASE_DIR, "../app.log"))
logger.level("INFO")

if __name__ == "__main__":
    logger.info("This is an info message.")
