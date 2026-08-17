from pydantic import BaseModel, Field

from app.retrieval.models import RetrievalResult


class RRFConfig(BaseModel):
    k: int = Field(default=60, gt=0)
    top_k: int = Field(default=10, gt=0)


def reciprocal_rank_fusion(
    result_lists: list[list[RetrievalResult]],
    config: RRFConfig | None = None,
) -> list[RetrievalResult]:
    """Combine one or more ranked result lists via Reciprocal Rank Fusion.

        RRF(d) = sum over lists containing d of  1 / (k + rank(d))

    rank(d) is d's 1-indexed position WITHIN EACH list, not a raw score
    -- this is what lets RRF combine scoring systems that aren't directly
    comparable (Postgres ts_rank vs. pgvector cosine similarity). A
    chunk_id present in multiple lists gets a contribution from each and
    those contributions are summed; a chunk_id present in only one list
    still receives that single contribution and remains eligible for the
    final ranking. The returned RetrievalResult.score is the fused RRF
    score, replacing whatever score the chunk carried in its source
    list(s); all other fields are taken from the first list (in
    result_lists order) the chunk_id was seen in.

    Ties in the final score are broken deterministically by chunk_id
    (ascending), so ordering never depends on dict/set iteration order
    or which list happened to be processed first.
    """
    config = config or RRFConfig()

    scores: dict[str, float] = {}
    representative: dict[str, RetrievalResult] = {}

    for results in result_lists:
        for rank, result in enumerate(results, start=1):
            scores[result.chunk_id] = scores.get(result.chunk_id, 0.0) + 1.0 / (
                config.k + rank
            )
            representative.setdefault(result.chunk_id, result)

    fused = [
        representative[chunk_id].model_copy(update={"score": score})
        for chunk_id, score in scores.items()
    ]
    fused.sort(key=lambda r: (-r.score, r.chunk_id))

    return fused[: config.top_k]
