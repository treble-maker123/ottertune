"""
This module provides a convenient method for building a logger with multiple
output streams.
"""
import logging
import sys


def build_logger(log_level: int = 10, output_path: str = None) -> logging.Logger:
    """
    Args:
        log_level: 10-DEBUG, 20-INFO, 30-WARN, 40-ERROR, 50-CRITICAL
        output_path: where to write the logs, not written to file if None
    """
    log_handlers = [logging.StreamHandler(sys.stdout)]

    if output_path is not None:
        file_handler = logging.FileHandler(filename=output_path)
        log_handlers.append(file_handler)

    logging.basicConfig(level=log_level,
                        handlers=log_handlers,
                        format='%(asctime)s [%(levelname)s] %(message)s')

    return logging.getLogger()
