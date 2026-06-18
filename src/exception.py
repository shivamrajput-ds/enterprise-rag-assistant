import sys

from src.logger import logger


class RagException(Exception):
    """
    Custom exception used throughout the RAG application.

    Captures:
    - Original error message
    - Source file name
    - Line number

    The formatted exception message is automatically logged
    for easier debugging and troubleshooting.
    """

    def __init__(self, message, error_detail):
        """
        Initialize the custom exception and extract traceback details.
        """
        super().__init__(message)

        self.message = str(message)

        _, _, exc_tb = error_detail.exc_info()

        if exc_tb:
            self.file_name = exc_tb.tb_frame.f_code.co_filename
            self.line_number = exc_tb.tb_lineno
        else:
            self.file_name = "Unknown"
            self.line_number = -1

        logger.error(str(self))

    def __str__(self) -> str:
        """
        Return a formatted exception message.
        """
        return (
            f"Error occurred in file: {self.file_name}\n"
            f"Line Number: {self.line_number}\n"
            f"Error Message: {self.message}"
        )


if __name__ == "__main__":
    try:
        result = 10 / 0
        print(result)

    except Exception as e:
        raise RagException(e, sys)