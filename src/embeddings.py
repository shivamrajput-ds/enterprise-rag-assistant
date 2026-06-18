import sys

from langchain_huggingface import HuggingFaceEmbeddings

from src.config_loader import config
from src.exception import RagException
from src.logger import logger


# Shared embedding model instance used across the application.
# Loading embedding models repeatedly is expensive, so the model
# is initialized once and reused.
_embedding_model = None


def get_embedding_model() -> HuggingFaceEmbeddings:
    """
    Load and return the configured embedding model.

    Returns:
        HuggingFaceEmbeddings: Initialized embedding model.
    """
    global _embedding_model

    try:
        if _embedding_model is not None:
            return _embedding_model

        logger.info("Starting embedding model loading process")

        model_name = config["embedding"]["model_name"]

        _embedding_model = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={
                "device": "cpu",
            },
            encode_kwargs={
                "normalize_embeddings": True,
            },
        )

        logger.info(
            f"Embedding model loaded successfully: {model_name}"
        )

        return _embedding_model

    except Exception as e:
        logger.error(
            f"Embedding model loading failed: {str(e)}"
        )
        raise RagException(str(e), sys)


if __name__ == "__main__":
    model = get_embedding_model()

    print("Embedding model loaded successfully.")
    print(model)