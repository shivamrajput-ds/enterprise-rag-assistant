from fastapi import FastAPI

from api.routes import router


app = FastAPI(
    title="Enterprise RAG Assistant API",
    description="FastAPI backend for document-based RAG assistant",
    version="1.0.0"
)

app.include_router(router)