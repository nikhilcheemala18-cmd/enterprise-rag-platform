from abc import ABC, abstractmethod
from typing import Any

from app.models.chunk import Chunk


class BaseIndexer(ABC):
    """Shared contract for a swappable search-index backend.

    LexicalIndexer and VectorIndexer implement this today; a future real
    BM25Index (e.g. backed by Elasticsearch/Tantivy) or an alternative
    vector store would implement it too, without IndexingService or
    ChunkRepository changing. Different backends need different extra
    data at index time (a vector needs an embedding; lexical search does
    not), so index() takes **kwargs rather than forcing one identical
    signature across fundamentally different backends.
    """

    @abstractmethod
    def index(self, chunk: Chunk, **kwargs: Any) -> None:
        raise NotImplementedError
