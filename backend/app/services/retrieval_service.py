from app.embeddings.base import EmbeddingProvider
from app.embeddings.query import embed_query
from app.retrieval.hybrid import HybridSearchService
from app.retrieval.models import RetrievalRequest, RetrievalResult


class RetrievalService:
    """Application-level entry point for the query -> ranked-results flow:

        query
          -> embed_query(embedding_provider, query)   (existing)
          -> RetrievalRequest                          (existing)
          -> HybridSearchService.search(request)       (existing: lexical +
             vector retrieval, then reciprocal_rank_fusion)
          -> list[RetrievalResult]                     (existing)

    Every stage is an already-existing, independently tested component;
    this class only sequences them -- exactly mirroring how
    IngestionService sequences loader/chunking/embedding/indexing on the
    write side. It does not reimplement lexical search, vector search,
    hybrid fusion, RRF, or embedding generation.

    Always supplies both `query` and `query_embedding` to
    RetrievalRequest, so HybridSearchService runs true hybrid mode (both
    retrievers + RRF) rather than single-mode -- matching this milestone's
    stated flow (query -> ... -> lexical + vector retrieval -> RRF).
    """

    def __init__(
        self,
        embedding_provider: EmbeddingProvider,
        hybrid_search_service: HybridSearchService,
    ):
        self.embedding_provider = embedding_provider
        self.hybrid_search_service = hybrid_search_service

    def search(
        self,
        query: str,
        top_k: int = 10,
        document_id: str | None = None,
    ) -> list[RetrievalResult]:
        if not query or not query.strip():
            raise ValueError("query must not be empty")

        query_embedding = embed_query(self.embedding_provider, query)

        # top_k/document_id validation is intentionally not duplicated
        # here -- RetrievalRequest already validates both (top_k > 0,
        # document_id non-empty if given) and raises on construction.
        request = RetrievalRequest(
            query=query,
            query_embedding=query_embedding,
            top_k=top_k,
            document_id=document_id,
        )

        return self.hybrid_search_service.search(request)
