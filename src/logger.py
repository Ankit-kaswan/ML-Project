import logging
import os
from datetime import datetime

class Logger:
    """Singleton Logger for consistent logging across the project."""

    _logger = None  # Static instance

    @staticmethod
    def get_logger(log_dir="logs", log_level=logging.INFO, logger_name='ml_project_logger'):
        """Returns a configured logger instance (singleton)."""

        if Logger._logger is None:
            # Create logs directory if not exists
            os.makedirs(log_dir, exist_ok=True)

            # Define log file with timestamp
            log_file = f"{datetime.now().strftime('%Y_%m_%d')}.log"
            log_file_path = os.path.join(log_dir, log_file)

            # Create logger
            logger = logging.getLogger(logger_name)
            logger.setLevel(log_level)

            # File handler
            file_handler = logging.FileHandler(log_file_path)
            file_handler.setFormatter(
                logging.Formatter("[%(asctime)s] %(lineno)d %(name)s - %(levelname)s - %(message)s")
            )

            # Console handler
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(
                logging.Formatter("[%(asctime)s] %(lineno)d %(name)s - %(levelname)s - %(message)s")
            )

            # Add handlers
            logger.addHandler(file_handler)
            logger.addHandler(console_handler)

            # Store singleton instance
            Logger._logger = logger

        return Logger._logger


# Testing
if __name__ == "__main__":
    logger = Logger.get_logger()
    logger.info("Logging system initialized.")
    logger.warning("Warning: Model accuracy below threshold.")
    logger.error("Error in data processing pipeline.")
