from collections.abc import Iterable
from dataclasses import dataclass

from app.evaluation.dataset import EvaluationCase
from app.retrieval.models import RetrievalResult
from app.services.rag_service import RAGSource


@dataclass(frozen=True)
class RetrievalMetrics:
    recall_at_k: dict[int, float | None]
    precision_at_k: dict[int, float | None]
    mrr: float | None


@dataclass(frozen=True)
class SourceGroundingMetrics:
    expected_document_found: bool | None
    expected_evidence_found: bool | None
    grounded: bool | None


def has_retrieval_ground_truth(case: EvaluationCase) -> bool:
    return bool(case.expected_chunk_ids or case.expected_evidence)


def result_matches_expected(result: RetrievalResult | RAGSource, case: EvaluationCase) -> bool:
    if result.chunk_id in set(case.expected_chunk_ids):
        return True

    content = result.content.lower()
    return any(evidence.lower() in content for evidence in case.expected_evidence)


def recall_at_k(
    results: list[RetrievalResult],
    case: EvaluationCase,
    k: int,
) -> float | None:
    expected_items = _expected_retrieval_items(case)
    if not expected_items:
        return None

    top_results = results[:k]
    found = sum(1 for item in expected_items if _expected_item_found(item, top_results))
    return found / len(expected_items)


def precision_at_k(
    results: list[RetrievalResult],
    case: EvaluationCase,
    k: int,
) -> float | None:
    if not has_retrieval_ground_truth(case):
        return None

    top_results = results[:k]
    if not top_results:
        return 0.0

    relevant_count = sum(1 for result in top_results if result_matches_expected(result, case))
    return relevant_count / k


def mean_reciprocal_rank(
    ranked_results: list[list[RetrievalResult]],
    cases: list[EvaluationCase],
) -> float | None:
    reciprocal_ranks: list[float] = []

    for results, case in zip(ranked_results, cases, strict=True):
        if not has_retrieval_ground_truth(case):
            continue
        reciprocal_ranks.append(_reciprocal_rank(results, case))

    if not reciprocal_ranks:
        return None
    return sum(reciprocal_ranks) / len(reciprocal_ranks)


def evaluate_retrieval(
    results: list[RetrievalResult],
    case: EvaluationCase,
    k_values: Iterable[int] = (5, 10),
) -> RetrievalMetrics:
    k_list = list(k_values)
    return RetrievalMetrics(
        recall_at_k={k: recall_at_k(results, case, k) for k in k_list},
        precision_at_k={k: precision_at_k(results, case, k) for k in k_list},
        mrr=_reciprocal_rank(results, case) if has_retrieval_ground_truth(case) else None,
    )


def evaluate_source_grounding(
    sources: list[RAGSource] | list[RetrievalResult],
    case: EvaluationCase,
) -> SourceGroundingMetrics:
    expected_document_found = None
    if case.expected_document_id is not None:
        expected_document_found = any(
            source.document_id == case.expected_document_id for source in sources
        )

    expected_evidence_found = None
    if case.expected_evidence:
        source_text = "\n".join(source.content for source in sources).lower()
        expected_evidence_found = all(
            evidence.lower() in source_text for evidence in case.expected_evidence
        )

    checks = [
        value
        for value in (expected_document_found, expected_evidence_found)
        if value is not None
    ]
    grounded = all(checks) if checks else None

    return SourceGroundingMetrics(
        expected_document_found=expected_document_found,
        expected_evidence_found=expected_evidence_found,
        grounded=grounded,
    )


def _reciprocal_rank(results: list[RetrievalResult], case: EvaluationCase) -> float:
    for rank, result in enumerate(results, start=1):
        if result_matches_expected(result, case):
            return 1.0 / rank
    return 0.0


def _expected_retrieval_items(case: EvaluationCase) -> list[tuple[str, str]]:
    items = [("chunk_id", chunk_id) for chunk_id in case.expected_chunk_ids]
    items.extend(("evidence", evidence) for evidence in case.expected_evidence)
    return items


def _expected_item_found(
    item: tuple[str, str],
    results: list[RetrievalResult],
) -> bool:
    kind, expected = item
    if kind == "chunk_id":
        return any(result.chunk_id == expected for result in results)

    expected_lower = expected.lower()
    return any(expected_lower in result.content.lower() for result in results)
