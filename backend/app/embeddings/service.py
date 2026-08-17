from app.embeddings.base import EmbeddingProvider
from app.models.chunk import Chunk


class ChunkEmbeddingService:
    """Embeds Chunk.content using a configured EmbeddingProvider.

    Returns a {chunk_id: vector} mapping -- exactly the shape
    IndexingService.index_chunks(chunks, embeddings=...) already expects
    (see app/indexing/service.py), so its output can be passed straight
    through without any adapter. Chunk order is preserved by construction
    (one batched embed_texts() call, zipped back against the input list);
    document_id is preserved implicitly since the Chunk objects
    themselves are never touched -- only chunk_id is extracted as the
    dict key.

    Does not mutate the Chunk model and does not store vectors anywhere;
    storage remains the indexing layer's responsibility.
    """

    def __init__(self, provider: EmbeddingProvider):
        self.provider = provider

    def embed_chunks(self, chunks: list[Chunk]) -> dict[str, list[float]]:
        if not chunks:
            return {}
        vectors = self.provider.embed_texts([chunk.content for chunk in chunks])
        return {chunk.chunk_id: vector for chunk, vector in zip(chunks, vectors)}
