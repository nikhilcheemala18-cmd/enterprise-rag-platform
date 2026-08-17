from app.retrieval.base import BaseRetriever
from app.retrieval.models import RetrievalRequest, RetrievalResult
from app.retrieval.rrf import RRFConfig, reciprocal_rank_fusion


class HybridSearchService:
    """Coordinates lexical + vector retrieval and fuses their results
    with Reciprocal Rank Fusion.

    Only calls the sub-retriever(s) relevant to what the request actually
    carries: a request with just `query` runs lexical-only, just
    `query_embedding` runs vector-only, both runs true hybrid fusion.
    RetrievalRequest's own validation guarantees at least one is present,
    so this never calls RRF with zero result lists.

    Stops at fused ranked results -- no reranking, no LLM generation.
    """

    def __init__(
        self,
        lexical_retriever: BaseRetriever,
        vector_retriever: BaseRetriever,
        rrf_config: RRFConfig | None = None,
    ):
        self.lexical_retriever = lexical_retriever
        self.vector_retriever = vector_retriever
        self.rrf_config = rrf_config or RRFConfig()

    def search(self, request: RetrievalRequest) -> list[RetrievalResult]:
        result_lists: list[list[RetrievalResult]] = []

        if request.query is not None:
            result_lists.append(self.lexical_retriever.search(request))

        if request.query_embedding is not None:
            result_lists.append(self.vector_retriever.search(request))

        return reciprocal_rank_fusion(result_lists, self.rrf_config)
