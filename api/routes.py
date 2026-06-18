import time
from typing import List

from fastapi import APIRouter, HTTPException

from api.schema import AskRequest, AskResponse
from src.rag_chain import generate_answer


router = APIRouter(tags=["RAG"])


FALLBACK_ANSWER = "I don't know based on the provided documents."


def build_sources(docs) -> List[str]:
    source_map = {}

    for doc in docs:
        file_name = doc.metadata.get("file_name", "unknown")
        file_type = doc.metadata.get("file_type", "unknown")
        page = doc.metadata.get("page")
        row = doc.metadata.get("row")
        sheet = doc.metadata.get("sheet_name")

        key = file_name

        if key not in source_map:
            source_map[key] = {
                "file_type": file_type,
                "pages": set(),
                "rows": set(),
                "sheets": set()
            }

        if page is not None:
            source_map[key]["pages"].add(page)

        if row is not None:
            source_map[key]["rows"].add(row)

        if sheet is not None:
            source_map[key]["sheets"].add(sheet)

    sources = []

    for file_name, values in source_map.items():
        parts = [f"📄 {file_name}", f"Type: {values['file_type']}"]

        if values["sheets"]:
            sheet_text = ", ".join(sorted(values["sheets"]))
            parts.append(f"Sheet: {sheet_text}")

        if values["pages"]:
            page_text = ", ".join(str(p) for p in sorted(values["pages"]))
            parts.append(f"Pages: {page_text}")

        if values["rows"]:
            row_text = ", ".join(str(r) for r in sorted(values["rows"]))
            parts.append(f"Rows: {row_text}")

        sources.append(" | ".join(parts))

    return sources


@router.get("/")
def root():
    return {
        "status": "success",
        "message": "Enterprise RAG API is running"
    }


@router.get("/health")
def health_check():
    return {
        "status": "healthy"
    }


@router.post("/ask", response_model=AskResponse)
def ask_question(request: AskRequest):
    try:
        start_time = time.time()

        result = generate_answer(request.query)

        response_time = round(time.time() - start_time, 2)

        answer = result["answer"].strip()

        if answer == FALLBACK_ANSWER:
            sources = []
        else:
            sources = build_sources(result["sources"][:1])

        return AskResponse(
            answer=answer,
            retrieved_chunks=result["retrieved_chunks"],
            response_time=response_time,
            sources=sources
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )