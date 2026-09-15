import argparse
from dataclasses import dataclass

from app.api.dependencies import (
    get_embedding_provider,
    get_hybrid_search_service,
    get_llm_provider,
)
from app.embeddings.query import embed_query
from app.evaluation.dataset import EvaluationCase, load_evaluation_dataset
from app.evaluation.metrics import (
    RetrievalMetrics,
    SourceGroundingMetrics,
    evaluate_retrieval,
    evaluate_source_grounding,
    mean_reciprocal_rank,
)
from app.evaluation.timing import LatencyAggregator, LatencyBreakdown, Timer
from app.rag.prompt import SYSTEM_INSTRUCTION, build_user_prompt
from app.retrieval.models import RetrievalRequest, RetrievalResult
from app.retrieval.rrf import RRFConfig, reciprocal_rank_fusion
from app.services.rag_service import RAGSource


@dataclass(frozen=True)
class CaseEvaluationResult:
    case: EvaluationCase
    results: list[RetrievalResult]
    retrieval_metrics: RetrievalMetrics
    source_grounding: SourceGroundingMetrics
    latency: LatencyBreakdown
    generated_answer: str | None = None


class RAGEvaluationRunner:
    def __init__(
        self,
        embedding_provider,
        lexical_retriever,
        vector_retriever,
        llm_provider=None,
        rrf_config: RRFConfig | None = None,
    ):
        self.embedding_provider = embedding_provider
        self.lexical_retriever = lexical_retriever
        self.vector_retriever = vector_retriever
        self.llm_provider = llm_provider
        self.rrf_config = rrf_config or RRFConfig()

    def evaluate_case(
        self,
        case: EvaluationCase,
        top_k: int = 10,
        document_id: str | None = None,
        generate_answer: bool = True,
    ) -> CaseEvaluationResult:
        scoped_document_id = document_id or case.expected_document_id

        with Timer() as total_timer:
            with Timer() as embedding_timer:
                query_embedding = embed_query(self.embedding_provider, case.question)

            request = RetrievalRequest(
                query=case.question,
                query_embedding=query_embedding,
                top_k=top_k,
                document_id=scoped_document_id,
            )

            with Timer() as hybrid_timer:
                with Timer() as lexical_timer:
                    lexical_results = self.lexical_retriever.search(request)
                with Timer() as vector_timer:
                    vector_results = self.vector_retriever.search(request)
                with Timer() as rrf_timer:
                    results = reciprocal_rank_fusion(
                        [lexical_results, vector_results],
                        self._rrf_config_for(top_k),
                    )

            answer = None
            generation_ms = None
            sources_for_grounding: list[RAGSource] | list[RetrievalResult] = results
            if generate_answer and results and self.llm_provider is not None:
                prompt = build_user_prompt(case.question, results)
                with Timer() as generation_timer:
                    answer = self.llm_provider.generate(
                        prompt,
                        system_instruction=SYSTEM_INSTRUCTION,
                    )
                generation_ms = generation_timer.elapsed_ms
                sources_for_grounding = [
                    RAGSource(
                        chunk_id=result.chunk_id,
                        document_id=result.document_id,
                        content=result.content,
                        score=result.score,
                        metadata=result.metadata,
                    )
                    for result in results
                ]

        latency = LatencyBreakdown(
            query_embedding_ms=embedding_timer.elapsed_ms,
            lexical_retrieval_ms=lexical_timer.elapsed_ms,
            vector_retrieval_ms=vector_timer.elapsed_ms,
            rrf_ms=rrf_timer.elapsed_ms,
            hybrid_retrieval_ms=hybrid_timer.elapsed_ms,
            generation_ms=generation_ms,
            total_ask_ms=total_timer.elapsed_ms,
        )

        return CaseEvaluationResult(
            case=case,
            results=results,
            retrieval_metrics=evaluate_retrieval(results, case, k_values=(5, 10)),
            source_grounding=evaluate_source_grounding(sources_for_grounding, case),
            latency=latency,
            generated_answer=answer,
        )

    def _rrf_config_for(self, top_k: int):
        return self.rrf_config.model_copy(update={"top_k": top_k})


def run_evaluation(
    cases: list[EvaluationCase],
    top_k: int = 10,
    document_id: str | None = None,
    generate_answers: bool = True,
) -> list[CaseEvaluationResult]:
    hybrid_service = get_hybrid_search_service()
    runner = RAGEvaluationRunner(
        embedding_provider=get_embedding_provider(),
        lexical_retriever=hybrid_service.lexical_retriever,
        vector_retriever=hybrid_service.vector_retriever,
        llm_provider=get_llm_provider() if generate_answers else None,
        rrf_config=hybrid_service.rrf_config,
    )
    return [
        runner.evaluate_case(
            case,
            top_k=top_k,
            document_id=document_id,
            generate_answer=generate_answers,
        )
        for case in cases
    ]


def format_report(results: list[CaseEvaluationResult]) -> str:
    lines = [f"Questions evaluated: {len(results)}"]

    if not results:
        lines.append("Answer accuracy: not automatically evaluated")
        return "\n".join(lines)

    for k in (5, 10):
        recall = _average_metric(
            result.retrieval_metrics.recall_at_k.get(k) for result in results
        )
        precision = _average_metric(
            result.retrieval_metrics.precision_at_k.get(k) for result in results
        )
        if recall is not None:
            lines.append(f"Recall@{k}: {_format_percent(recall)}")
        if precision is not None:
            lines.append(f"Precision@{k}: {_format_percent(precision)}")

    mrr = mean_reciprocal_rank(
        [result.results for result in results],
        [result.case for result in results],
    )
    if mrr is not None:
        lines.append(f"MRR: {mrr:.2f}")

    source_grounding = _average_metric(
        float(result.source_grounding.grounded)
        for result in results
        if result.source_grounding.grounded is not None
    )
    if source_grounding is not None:
        lines.append(f"Source grounding: {_format_percent(source_grounding)}")

    lines.append("Answer accuracy: not automatically evaluated")

    latency = LatencyAggregator()
    for result in results:
        latency.add(result.latency)
    averages = latency.averages()

    _append_latency(lines, averages, "query_embedding_ms", "Average query embedding latency")
    _append_latency(lines, averages, "lexical_retrieval_ms", "Average lexical retrieval latency")
    _append_latency(lines, averages, "vector_retrieval_ms", "Average vector retrieval latency")
    _append_latency(lines, averages, "rrf_ms", "Average RRF latency")
    _append_latency(lines, averages, "hybrid_retrieval_ms", "Average hybrid/RRF retrieval latency")
    _append_latency(lines, averages, "generation_ms", "Average generation latency", seconds=True)
    _append_latency(lines, averages, "total_ask_ms", "Average total /ask latency", seconds=True)

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the native RAG evaluation benchmark.")
    parser.add_argument("--dataset", default=None, help="Path to evaluation dataset JSON.")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--document-id", default=None)
    parser.add_argument(
        "--skip-generation",
        action="store_true",
        help="Measure retrieval only and avoid Gemini answer generation.",
    )
    args = parser.parse_args()

    cases = load_evaluation_dataset(args.dataset)
    results = run_evaluation(
        cases,
        top_k=args.top_k,
        document_id=args.document_id,
        generate_answers=not args.skip_generation,
    )
    print(format_report(results))


def _average_metric(values) -> float | None:
    measured = [value for value in values if value is not None]
    if not measured:
        return None
    return sum(measured) / len(measured)


def _format_percent(value: float) -> str:
    return f"{value * 100:.0f}%"


def _append_latency(
    lines: list[str],
    averages: dict[str, float],
    key: str,
    label: str,
    seconds: bool = False,
) -> None:
    if key not in averages:
        return
    value = averages[key]
    if seconds:
        lines.append(f"{label}: {value / 1000:.2f} s")
    else:
        lines.append(f"{label}: {value:.1f} ms")


if __name__ == "__main__":
    main()
