import logging
import os
from datetime import datetime


# Create logs directory if it does not exist.
os.makedirs("logs", exist_ok=True)


# Application logger.
logger = logging.getLogger("enterprise_rag")
logger.setLevel(logging.INFO)


# Log message format.
formatter = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)


# Create a timestamped log file.
LOG_FILE = (
    f"{datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}.log"
)


# Prevent duplicate handlers during reloads
# (especially useful in Streamlit applications).
if not logger.handlers:

    file_handler = logging.FileHandler(
        os.path.join("logs", LOG_FILE)
    )
    file_handler.setFormatter(formatter)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)


# Prevent propagation to the root logger.
logger.propagate = False