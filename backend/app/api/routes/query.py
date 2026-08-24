import logging
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from app.api.dependencies import get_retrieval_service
from app.services.retrieval_service import RetrievalService

logger = logging.getLogger(__name__)

router = APIRouter(tags=["retrieval"])


class QueryRequest(BaseModel):
    query: str = Field(min_length=1, description="The natural-language search query.")
    top_k: int = Field(
        default=10, gt=0, le=100, description="Maximum number of results to return."
    )
    document_id: str | None = Field(
        default=None,
        min_length=1,
        description="Optional filter to scope results to a single document.",
    )


class QueryResultItem(BaseModel):
    chunk_id: str
    document_id: str
    content: str
    score: float
    metadata: dict[str, Any] = Field(default_factory=dict)


class QueryResponse(BaseModel):
    query: str
    result_count: int
    results: list[QueryResultItem]


@router.post("/query", response_model=QueryResponse, status_code=200)
def query_documents(
    request: QueryRequest,
    retrieval_service: RetrievalService = Depends(get_retrieval_service),
) -> QueryResponse:
    """Runs a hybrid (lexical + vector, RRF-fused) search over indexed
    chunks and returns the ranked results. This route only handles HTTP
    concerns (request validation, status-code mapping, response
    shaping) -- query embedding, lexical search, vector search, and RRF
    fusion all remain RetrievalService's responsibility. No reranking or
    LLM generation happens here; results are returned as retrieved.
    """
    try:
        results = retrieval_service.search(
            request.query, top_k=request.top_k, document_id=request.document_id
        )
    except ValueError as exc:
        # RetrievalService/RetrievalRequest validation failures (e.g. a
        # whitespace-only query slipping past the min_length check above)
        raise HTTPException(status_code=400, detail="Invalid query request.") from exc
    except Exception as exc:
        # Deliberately no str(exc) in the response -- see app/api/routes/
        # upload.py for the same convention (full detail logged
        # server-side only, never sent to the client).
        logger.exception("Unexpected error during query %r", request.query)
        raise HTTPException(status_code=500, detail="Unexpected error during retrieval.") from exc

    return QueryResponse(
        query=request.query,
        result_count=len(results),
        results=[
            QueryResultItem(
                chunk_id=r.chunk_id,
                document_id=r.document_id,
                content=r.content,
                score=r.score,
                metadata=r.metadata,
            )
            for r in results
        ],
    )
