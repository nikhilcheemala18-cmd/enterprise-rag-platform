import logging
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from app.api.dependencies import get_rag_service
from app.services.rag_service import (
    GenerationFailedError,
    GenerationTemporarilyUnavailableError,
    InvalidQuestionError,
    RAGService,
    RetrievalFailedError,
)

logger = logging.getLogger(__name__)

router = APIRouter(tags=["rag"])


class AskRequest(BaseModel):
    question: str = Field(min_length=1, description="The natural-language question to answer.")
    top_k: int = Field(
        default=10,
        gt=0,
        le=100,
        description="Maximum number of retrieved chunks to use as grounding context.",
    )
    document_id: str | None = Field(
        default=None,
        min_length=1,
        description="Optional filter to scope the answer to a single document.",
    )


class AskSourceItem(BaseModel):
    chunk_id: str
    document_id: str
    content: str
    score: float
    metadata: dict[str, Any] = Field(default_factory=dict)


class AskResponse(BaseModel):
    question: str
    answer: str
    retrieved_count: int
    sources: list[AskSourceItem]


@router.post("/ask", response_model=AskResponse, status_code=200)
def ask_question(
    request: AskRequest,
    rag_service: RAGService = Depends(get_rag_service),
) -> AskResponse:
    """Answers a question using the existing RAGService. This route only
    handles HTTP concerns (request validation, status-code mapping, and
    response shaping) -- retrieval, prompt construction, and generation
    all remain RAGService's responsibility.
    """
    try:
        result = rag_service.answer(
            request.question,
            top_k=request.top_k,
            document_id=request.document_id,
        )
    except InvalidQuestionError as exc:
        raise HTTPException(status_code=400, detail="Invalid ask request.") from exc
    except RetrievalFailedError as exc:
        logger.exception("Retrieval failed while answering question %r", request.question)
        raise HTTPException(status_code=500, detail="Retrieval failed.") from exc
    except GenerationTemporarilyUnavailableError as exc:
        logger.exception("AI service temporarily unavailable while answering question %r", request.question)
        raise HTTPException(
            status_code=503,
            detail=(
                "The AI service is temporarily unavailable due to high demand. "
                "Please try again shortly."
            ),
        ) from exc
    except GenerationFailedError as exc:
        logger.exception("Generation failed while answering question %r", request.question)
        raise HTTPException(status_code=500, detail="Answer generation failed.") from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="Invalid ask request.") from exc
    except Exception as exc:
        logger.exception("Unexpected error while answering question %r", request.question)
        raise HTTPException(status_code=500, detail="Unexpected error during answer generation.") from exc

    return AskResponse(
        question=result.question,
        answer=result.answer,
        retrieved_count=result.retrieved_count,
        sources=[
            AskSourceItem(
                chunk_id=s.chunk_id,
                document_id=s.document_id,
                content=s.content,
                score=s.score,
                metadata=s.metadata,
            )
            for s in result.sources
        ],
    )
