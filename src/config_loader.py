import os
import sys

import yaml

from src.logger import logger
from src.exception import RagException


try:
    config_path = os.path.join(os.getcwd(), "config.yaml")

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"config.yaml not found at: {config_path}")

    with open(config_path, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if not config:
        raise ValueError("config.yaml is empty or invalid.")

    if "paths" not in config:
        raise ValueError("config.yaml missing required section: paths")

    if "documents_dir" not in config["paths"]:
        raise ValueError("config.yaml missing required key: paths.documents_dir")

    logger.info("Config loaded successfully")

except Exception as e:
    logger.error(f"Failed to load config: {str(e)}")
    raise RagException(
        f"Failed to load config.yaml: {str(e)}",
        sys,
    )