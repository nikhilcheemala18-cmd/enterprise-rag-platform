from abc import ABC, abstractmethod
from typing import Any, Protocol

from app.retrieval.models import RetrievalRequest, RetrievalResult


class SearchBackend(Protocol):
    """Structural contract a retriever wraps -- matches the shape already
    exposed by app.indexing.lexical.LexicalIndexer.search() and
    app.indexing.vector.VectorIndexer.search() (query-or-embedding,
    limit) -> list of objects with chunk_id/document_id/content/score/
    metadata. Real indexer instances satisfy this without any change;
    tests can pass a plain fake instead. This is the retrieval layer's
    only coupling to "how search actually happens" -- it never issues a
    query itself.
    """

    def search(self, query: Any, limit: int = 10) -> list[Any]: ...


class BaseRetriever(ABC):
    """Shared contract so lexical and vector retrieval expose the same
    search interface to HybridSearchService (and to each other's tests).
    """

    @abstractmethod
    def search(self, request: RetrievalRequest) -> list[RetrievalResult]:
        raise NotImplementedError


def to_retrieval_result(item: Any) -> RetrievalResult:
    """Adapt a backend search hit (e.g. app.indexing.models.SearchResult,
    or an equivalent fake in tests) into the retrieval layer's own
    RetrievalResult. Reads fields structurally rather than importing
    SearchResult as a hard type dependency.
    """
    return RetrievalResult(
        chunk_id=item.chunk_id,
        document_id=item.document_id,
        content=item.content,
        score=item.score,
        metadata=item.metadata,
    )


def apply_document_filter(
    results: list[RetrievalResult], document_id: str | None
) -> list[RetrievalResult]:
    """Post-filter results by document_id in plain Python. Backends are
    not required to support this filter server-side in this phase (no
    live PostgreSQL queries are implemented yet), so filtering happens
    here instead of being pushed down into indexing/SQL.
    """
    if document_id is None:
        return results
    return [r for r in results if r.document_id == document_id]
