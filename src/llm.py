import os
import sys

from dotenv import load_dotenv
from langchain_groq import ChatGroq

from src.config_loader import config
from src.exception import RagException
from src.logger import logger


load_dotenv()


def get_llm() -> ChatGroq:
    """
    Initialize and return the configured Groq chat model.
    """
    try:
        logger.info("Loading Groq LLM")

        groq_api_key = os.getenv("GROQ_API_KEY")

        if not groq_api_key:
            raise ValueError("GROQ_API_KEY not found in environment variables.")

        llm = ChatGroq(
            api_key=groq_api_key,
            model=config["llm"]["model_name"],
            temperature=config["llm"]["temperature"],
            max_tokens=config["llm"]["max_tokens"],
        )

        logger.info("Groq LLM loaded successfully")

        return llm

    except Exception as e:
        logger.error(f"LLM loading failed: {str(e)}")
        raise RagException(str(e), sys)