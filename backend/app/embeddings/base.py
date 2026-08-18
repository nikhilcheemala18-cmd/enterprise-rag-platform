from abc import ABC, abstractmethod

from app.embeddings.config import EmbeddingConfig


class EmbeddingProvider(ABC):
    """Converts text into vectors. Not coupled to any specific vendor
    (OpenAI, Hugging Face, Ollama, ...) -- concrete subclasses own that.

    embed_text()/embed_texts() are concrete and shared by every provider:
    they call the subclass's _embed_texts() and then validate, in one
    place, that (a) the number of vectors matches the number of input
    texts and (b) every vector's length matches self.config.dimension.
    This is what makes it structurally impossible for a provider to
    silently return vectors of the wrong shape -- the check happens on
    every call, not just in tests.

    `is_query` signals whether the given text(s) are search queries as
    opposed to documents/passages. It exists because some real models
    (e.g. BGE) have a documented asymmetric convention -- an instruction
    prefix on the query side only -- while others have none. A provider
    without such a convention (e.g. DeterministicTestEmbeddingProvider)
    simply ignores the flag. Document-side callers (ChunkEmbeddingService)
    never pass is_query=True; only embed_query() does. This keeps the
    asymmetry a provider-internal detail rather than leaking a
    model-specific instruction string into generic caller code.
    """

    def __init__(self, config: EmbeddingConfig):
        self.config = config

    def embed_text(self, text: str, is_query: bool = False) -> list[float]:
        return self.embed_texts([text], is_query=is_query)[0]

    def embed_texts(self, texts: list[str], is_query: bool = False) -> list[list[float]]:
        vectors = self._embed_texts(texts, is_query=is_query)

        if len(vectors) != len(texts):
            raise ValueError(
                f"{type(self).__name__} returned {len(vectors)} vectors "
                f"for {len(texts)} input texts"
            )

        for vector in vectors:
            if len(vector) != self.config.dimension:
                raise ValueError(
                    f"{type(self).__name__} returned a {len(vector)}-dimensional "
                    f"vector, expected {self.config.dimension} per EmbeddingConfig "
                    f"(model={self.config.model_name!r})"
                )

        return vectors

    @abstractmethod
    def _embed_texts(self, texts: list[str], is_query: bool) -> list[list[float]]:
        """Subclasses implement the actual embedding call here. May
        assume texts is a list (possibly empty); must return one vector
        per input text, in the same order. `is_query` is provided for
        providers whose model has an asymmetric query/document
        convention; providers without one may ignore it.
        """
        raise NotImplementedError
