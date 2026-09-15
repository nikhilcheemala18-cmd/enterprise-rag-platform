from typing import Any

from pydantic import BaseModel, Field

from app.rag.llm import LLMProvider, LLMServiceUnavailableError
from app.rag.prompt import SYSTEM_INSTRUCTION, build_user_prompt
from app.services.retrieval_service import RetrievalService


class RAGError(Exception):
    """Base class for failures raised by RAGService.answer()."""


class InvalidQuestionError(RAGError, ValueError):
    """The question was invalid (empty/whitespace)."""


class RetrievalFailedError(RAGError, RuntimeError):
    """RetrievalService.search() failed."""


class GenerationFailedError(RAGError, RuntimeError):
    """The injected LLMProvider failed to generate an answer."""


class GenerationTemporarilyUnavailableError(GenerationFailedError):
    """The LLM provider is temporarily unavailable."""


class RAGSource(BaseModel):
    """One retrieved chunk backing the answer. Built directly from a
    RetrievalResult -- never from anything the LLM produced -- so
    attribution stays deterministic.
    """

    chunk_id: str
    document_id: str
    content: str
    score: float
    metadata: dict[str, Any] = Field(default_factory=dict)


class RAGAnswer(BaseModel):
    question: str
    answer: str
    sources: list[RAGSource]
    retrieved_count: int


class RAGService:
    """Application-level entry point for the question -> grounded-answer
    flow:

        question
          -> RetrievalService.search(question, top_k, document_id)   (existing)
          -> list[RetrievalResult]
          -> build_user_prompt(question, results)                     (existing)
          -> LLMProvider.generate(prompt, system_instruction=...)      (injected)
          -> RAGAnswer(answer, sources, retrieved_count)

    Does not know about FastAPI, does not reimplement retrieval, prompt
    building, or generation -- it only sequences the existing pieces,
    mirroring IngestionService's and RetrievalService's own composition
    style. `sources` always come from the retrieval results themselves,
    never from LLM output, so citations remain deterministic even if the
    model's answer text is imperfect.
    """

    NO_CONTEXT_ANSWER = (
        "I couldn't find any relevant information in the indexed documents "
        "to answer this question."
    )

    def __init__(self, retrieval_service: RetrievalService, llm_provider: LLMProvider):
        self.retrieval_service = retrieval_service
        self.llm_provider = llm_provider

    def answer(
        self,
        question: str,
        top_k: int = 10,
        document_id: str | None = None,
    ) -> RAGAnswer:
        if not question or not question.strip():
            raise InvalidQuestionError("question must not be empty")

        try:
            results = self.retrieval_service.search(
                question, top_k=top_k, document_id=document_id
            )
        except Exception as exc:
            raise RetrievalFailedError(
                f"Retrieval failed for the given question ({type(exc).__name__})"
            ) from exc

        if not results:
            return RAGAnswer(
                question=question,
                answer=self.NO_CONTEXT_ANSWER,
                sources=[],
                retrieved_count=0,
            )

        prompt = build_user_prompt(question, results)

        try:
            answer_text = self.llm_provider.generate(
                prompt, system_instruction=SYSTEM_INSTRUCTION
            )
        except LLMServiceUnavailableError as exc:
            raise GenerationTemporarilyUnavailableError(
                f"Answer generation temporarily unavailable ({type(exc).__name__})"
            ) from exc
        except Exception as exc:
            raise GenerationFailedError(
                f"Answer generation failed ({type(exc).__name__})"
            ) from exc

        return RAGAnswer(
            question=question,
            answer=answer_text,
            sources=[
                RAGSource(
                    chunk_id=r.chunk_id,
                    document_id=r.document_id,
                    content=r.content,
                    score=r.score,
                    metadata=r.metadata,
                )
                for r in results
            ],
            retrieved_count=len(results),
        )
