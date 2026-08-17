from app.retrieval.base import BaseRetriever, SearchBackend, apply_document_filter, to_retrieval_result
from app.retrieval.models import RetrievalRequest, RetrievalResult


class VectorRetriever(BaseRetriever):
    """Retrieval-layer boundary for semantic (pgvector) search.

    Wraps a search backend (in production, an app.indexing.vector.
    VectorIndexer; in tests, a fake) and adapts its results into
    RetrievalResult. Accepts only a precomputed query embedding -- it
    does NOT generate embeddings and does NOT execute PostgreSQL queries
    itself; it only defines the boundary and delegates.
    """

    def __init__(self, backend: SearchBackend):
        self.backend = backend

    def search(self, request: RetrievalRequest) -> list[RetrievalResult]:
        if not request.query_embedding:
            raise ValueError("VectorRetriever requires request.query_embedding")
        raw_results = self.backend.search(request.query_embedding, limit=request.top_k)
        results = [to_retrieval_result(r) for r in raw_results]
        return apply_document_filter(results, request.document_id)
