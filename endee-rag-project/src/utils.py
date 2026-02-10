"""
Utility functions for logging and error handling.
"""

import logging
import sys

def setup_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """
    Set up a logger with console handler.
    
    Args:
        name: Logger name
        level: Logging level (default: INFO)
    
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Avoid duplicate handlers
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(level)
        
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    
    return logger


def handle_error(logger: logging.Logger, error: Exception, context: str):
    """
    Handle errors with consistent logging.
    
    Args:
        logger: Logger instance
        error: Exception that occurred
        context: Context description
    """
    logger.error(f"Error in {context}: {str(error)}", exc_info=True)
