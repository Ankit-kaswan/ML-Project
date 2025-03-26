import sys
import traceback
import threading
from src.logger import Logger


# Singleton logger instance
logger = Logger.get_logger()


class CustomException(Exception):
    """Custom exception class that provides detailed error messages and automatic logging."""

    _local = threading.local()  # Thread-local storage for safe exception handling

    def __init__(self, error_message, cause=None):
        """
        Initializes the CustomException.

        Args:
            error_message (str): The error message.
            cause (Exception, optional): The original exception causing this error.
        """
        super().__init__(error_message)

        # Capture full stack trace
        self.error_message = self._get_detailed_error_message(error_message, cause)

        # Store in thread-local storage for debugging in multi-threaded apps
        self._local.last_error = self.error_message

        # Auto-log the error
        logger.error(self.error_message)

    def __str__(self):
        return self.error_message

    @staticmethod
    def _get_detailed_error_message(error_message, cause):
        """
        Generates a detailed error message including file name, line number, and stack trace.

        Args:
            error_message (str): Custom error message.
            cause (Exception, optional): Original cause of the exception.

        Returns:
            str: Formatted error message.
        """
        exc_type, exc_value, exc_tb = sys.exc_info()
        if exc_tb is not None:
            tb_frame = traceback.extract_tb(exc_tb)[-1]  # Get last traceback entry
            file_name = tb_frame.filename
            line_number = tb_frame.lineno
            error_type = exc_type.__name__
            detailed_message = f"[{error_type}] Error in {file_name}, line {line_number}: {error_message}"
        else:
            detailed_message = f"[{error_message}]"

        # Add original cause if provided
        if cause:
            cause_trace = "".join(traceback.format_exception(type(cause), cause, cause.__traceback__))
            detailed_message += f"\nCaused by:\n{cause_trace}"

        return detailed_message

    @classmethod
    def get_last_error(cls):
        """Returns the last recorded error for the current thread (useful in multi-threaded environments)."""
        return getattr(cls._local, "last_error", None)


# Testing
if __name__ == '__main__':
    def faulty_function():
        return 1 / 0  # This will raise ZeroDivisionError

    try:
        faulty_function()
    except Exception as e:
        raise CustomException("Division failed!", cause=e)
