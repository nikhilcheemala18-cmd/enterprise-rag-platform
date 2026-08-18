from sentence_transformers import SentenceTransformer

from app.embeddings.base import EmbeddingProvider
from app.embeddings.config import EmbeddingConfig

BGE_MODEL_NAME = "BAAI/bge-base-en-v1.5"
BGE_DIMENSION = 768
BGE_MAX_TOKENS = 512
BGE_QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "


def default_bge_config() -> EmbeddingConfig:
    return EmbeddingConfig(
        provider_name="huggingface", model_name=BGE_MODEL_NAME, dimension=BGE_DIMENSION
    )


def _to_float_matrix(vectors) -> list[list[float]]:
    """Convert a numpy array (or any 2D array-like of numeric rows) into
    plain Python list[list[float]]. The provider contract promises plain
    floats to every caller outside this module, never numpy scalar
    types -- kept as a standalone function so it's unit-testable with a
    plain numpy array, without loading the actual BGE model.
    """
    return [[float(value) for value in row] for row in vectors]


class BGEEmbeddingProvider(EmbeddingProvider):
    """Real local semantic embedding provider: BAAI/bge-base-en-v1.5 via
    sentence-transformers. Runs fully offline at inference time -- no
    external API calls, no API key. The underlying model weights are
    fetched through the standard Hugging Face cache the first time this
    class is instantiated for a given model (see "Model download/
    caching" below); nothing is bundled with this repository.

    Usage choices below come directly from the model's Hugging Face model
    card (https://huggingface.co/BAAI/bge-base-en-v1.5), not invented:

    - normalize_embeddings=True is used for BOTH document and query
      embeddings -- the model card's own usage examples call
      model.encode(..., normalize_embeddings=True) on both `p_embeddings`
      (passages) and `q_embeddings` (queries). Normalizing both sides
      consistently is also what the task requires and what makes cosine
      similarity (and pgvector's cosine distance) well-defined here.
    - The documented query instruction, "Represent this sentence for
      searching relevant passages: ", is prepended ONLY when is_query is
      True. The card is explicit that "the instruction is not needed for
      passages." For v1.5 the card notes the instruction is optional
      (a small quality difference without it), but since applying it
      costs nothing, this provider follows the documented recommendation
      rather than skip it.

    Model loading: the SentenceTransformer model is loaded exactly once,
    in __init__, and reused for every embed_text()/embed_texts() call --
    it is never reloaded per call. Batch calls go through
    SentenceTransformer.encode()'s native batching, not a Python loop
    over single-text calls.

    Model download/caching: on first use, sentence-transformers downloads
    the model via huggingface_hub into the standard HF cache (typically
    ~/.cache/huggingface on Linux/macOS, %USERPROFILE%\\.cache\\
    huggingface on Windows), controllable via the standard HF_HOME /
    SENTENCE_TRANSFORMERS_HOME environment variables, or by passing an
    explicit `cache_folder` to this constructor. Subsequent loads reuse
    the cache and do not re-download. No model files are committed to
    this repository.

    CPU/memory: bge-base-en-v1.5 is a ~110M-parameter BERT-sized model
    (~420MB of weights on disk). Encoding runs on CPU by default in this
    environment (the installed torch build has no CUDA support here);
    CPU inference is noticeably slower than GPU but entirely workable for
    the batch sizes this pipeline currently produces (single documents/
    chunks, not high-QPS serving). `device` can be set explicitly (e.g.
    "cuda") if a GPU-enabled torch build is available later.
    """

    def __init__(
        self,
        config: EmbeddingConfig | None = None,
        cache_folder: str | None = None,
        device: str | None = None,
    ):
        config = config or default_bge_config()
        if config.model_name != BGE_MODEL_NAME:
            raise ValueError(
                f"BGEEmbeddingProvider only supports model_name={BGE_MODEL_NAME!r}, "
                f"got {config.model_name!r}"
            )
        if config.dimension != BGE_DIMENSION:
            raise ValueError(
                f"{BGE_MODEL_NAME} produces {BGE_DIMENSION}-dimensional vectors, "
                f"but EmbeddingConfig.dimension={config.dimension}"
            )

        super().__init__(config)
        self._model = SentenceTransformer(
            config.model_name, cache_folder=cache_folder, device=device
        )

    def _embed_texts(self, texts: list[str], is_query: bool) -> list[list[float]]:
        if not texts:
            return []

        inputs = [BGE_QUERY_INSTRUCTION + text for text in texts] if is_query else texts

        vectors = self._model.encode(
            inputs,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return _to_float_matrix(vectors)
