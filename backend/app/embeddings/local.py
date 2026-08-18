import hashlib

from app.embeddings.base import EmbeddingProvider


class DeterministicTestEmbeddingProvider(EmbeddingProvider):
    """A deterministic, dependency-free embedding provider for TESTS AND
    LOCAL DEVELOPMENT ONLY.

    It hashes each input text (SHA-256) and expands the digest bytes into
    a fixed-length vector, so the same text always produces the same
    vector and different texts produce different vectors. That is enough
    to exercise the embedding contract, ChunkEmbeddingService, and
    VectorIndexer wiring end-to-end with no network access and no API
    credentials.

    IMPORTANT: this is NOT a semantic embedding model. Distances between
    its vectors do not reflect meaning, similarity, or relevance in any
    way -- two texts about completely unrelated topics are just as
    likely to land close together as two texts about the same topic. It
    must never be used to serve real retrieval quality, only to validate
    that data flows correctly through the pipeline.
    """

    def _embed_texts(self, texts: list[str], is_query: bool) -> list[list[float]]:
        # no asymmetric query/document convention -- is_query is ignored.
        return [self._hash_to_vector(text) for text in texts]

    def _hash_to_vector(self, text: str) -> list[float]:
        dimension = self.config.dimension
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        repeats = (dimension // len(digest)) + 1
        raw_bytes = (digest * repeats)[:dimension]
        return [(byte / 127.5) - 1.0 for byte in raw_bytes]
