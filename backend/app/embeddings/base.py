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
    """

    def __init__(self, config: EmbeddingConfig):
        self.config = config

    def embed_text(self, text: str) -> list[float]:
        return self.embed_texts([text])[0]

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        vectors = self._embed_texts(texts)

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
    def _embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Subclasses implement the actual embedding call here. May
        assume texts is a list (possibly empty); must return one vector
        per input text, in the same order.
        """
        raise NotImplementedError
