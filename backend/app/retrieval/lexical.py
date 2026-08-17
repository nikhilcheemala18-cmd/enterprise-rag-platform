from app.retrieval.base import BaseRetriever, SearchBackend, apply_document_filter, to_retrieval_result
from app.retrieval.models import RetrievalRequest, RetrievalResult


class LexicalRetriever(BaseRetriever):
    """Retrieval-layer boundary for lexical (Postgres full-text) search.

    Wraps a search backend (in production, an app.indexing.lexical.
    LexicalIndexer; in tests, a fake) and adapts its results into
    RetrievalResult. Does NOT execute PostgreSQL queries itself -- it
    only defines the boundary and delegates.
    """

    def __init__(self, backend: SearchBackend):
        self.backend = backend

    def search(self, request: RetrievalRequest) -> list[RetrievalResult]:
        if not request.query:
            raise ValueError("LexicalRetriever requires request.query")
        raw_results = self.backend.search(request.query, limit=request.top_k)
        results = [to_retrieval_result(r) for r in raw_results]
        return apply_document_filter(results, request.document_id)
