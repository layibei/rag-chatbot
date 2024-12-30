# logging_util.py
import os
import sys
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

# Default logging configuration
DEFAULT_LOG_CONFIG = {
    'level': 'INFO',
    'format': '%(asctime)s | %(levelname)s | %(threadName)s | %(module)s:%(funcName)s:%(lineno)d | %(metadata)s | %(message)s',
    'date_format': '%Y-%m-%d %H:%M:%S'
}

def configure_logger(log_file_path: str = None, config: dict = None) -> logging.Logger:
    """Configure and return a logger instance"""
    log_config = config or DEFAULT_LOG_CONFIG
    
    # Create logger
    logger = logging.getLogger('app')
    logger.setLevel(log_config.get('level', 'INFO'))
    
    # Create formatters and handlers
    formatter = logging.Formatter(
        fmt=log_config.get('format', DEFAULT_LOG_CONFIG['format']),
        datefmt=log_config.get('date_format', DEFAULT_LOG_CONFIG['date_format'])
    )
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler (if path provided)
    if log_file_path:
        # Ensure directory exists
        log_dir = os.path.dirname(log_file_path)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
            
        file_handler = RotatingFileHandler(
            log_file_path,
            maxBytes=10*1024*1024,  # 10MB
            backupCount=5
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger

# Initialize default logger
BASE_DIR = str(Path(__file__).resolve().parent)
logger = configure_logger()
