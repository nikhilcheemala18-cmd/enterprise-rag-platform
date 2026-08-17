from app.embeddings.base import EmbeddingProvider


def embed_query(provider: EmbeddingProvider, query: str) -> list[float]:
    """Embed a user's query for semantic retrieval.

    This is a thin, deliberately visible entry point rather than callers
    reaching for provider.embed_text() directly: it documents the single
    architectural rule that matters here -- query embeddings MUST be
    produced by the same EmbeddingProvider (same model, same config) used
    for document-side chunk embeddings, or vector search results are
    meaningless. Passing the same `provider` instance on both sides is
    what guarantees that; this function does not do anything provider-
    specific itself.

    The resulting vector is intended for
    RetrievalRequest(query_embedding=...) -> VectorRetriever ->
    HybridSearchService. This module does not import the retrieval
    layer -- constructing the request is left to the caller.
    """
    return provider.embed_text(query)
