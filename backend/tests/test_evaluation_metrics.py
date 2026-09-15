import unittest

from app.evaluation.dataset import EvaluationCase
from app.evaluation.metrics import (
    evaluate_retrieval,
    evaluate_source_grounding,
    mean_reciprocal_rank,
    precision_at_k,
    recall_at_k,
)
from app.evaluation.runner import CaseEvaluationResult, RAGEvaluationRunner, format_report
from app.evaluation.timing import LatencyAggregator, LatencyBreakdown, average_optional
from app.embeddings.base import EmbeddingProvider
from app.embeddings.config import EmbeddingConfig
from app.retrieval.models import RetrievalResult


def result(
    chunk_id: str,
    content: str | None = None,
    document_id: str = "doc-1",
) -> RetrievalResult:
    return RetrievalResult(
        chunk_id=chunk_id,
        document_id=document_id,
        content=content or f"content for {chunk_id}",
        score=1.0,
        metadata={},
    )


class FakeEmbeddingProvider(EmbeddingProvider):
    def __init__(self):
        super().__init__(
            EmbeddingConfig(provider_name="fake", model_name="fake", dimension=2)
        )
        self.calls: list[tuple[str, bool]] = []

    def _embed_texts(self, texts: list[str], is_query: bool) -> list[list[float]]:
        self.calls.extend((text, is_query) for text in texts)
        return [[1.0, 0.0] for _ in texts]


class FakeRetriever:
    def __init__(self, results: list[RetrievalResult]):
        self.results = results
        self.calls = 0

    def search(self, request):
        self.calls += 1
        return self.results


class FakeLLMProvider:
    def __init__(self):
        self.calls = 0

    def generate(self, prompt: str, system_instruction: str | None = None) -> str:
        self.calls += 1
        return "The answer is grounded in the retrieved source."


class TestRetrievalEvaluationMetrics(unittest.TestCase):
    def test_recall_at_k_uses_expected_chunk_ids(self):
        case = EvaluationCase(
            id="q1",
            question="question?",
            expected_chunk_ids=["c2", "c4"],
        )
        results = [result("c1"), result("c2"), result("c3"), result("c4")]

        self.assertEqual(recall_at_k(results, case, 2), 0.5)
        self.assertEqual(recall_at_k(results, case, 4), 1.0)

    def test_recall_at_k_uses_expected_evidence(self):
        case = EvaluationCase(
            id="q1",
            question="question?",
            expected_evidence=["Revenue grew 18%", "Client A"],
        )
        results = [result("c1", "Revenue grew 18% in July for Client A.")]

        self.assertEqual(recall_at_k(results, case, 5), 1.0)

    def test_precision_at_k_is_none_without_ground_truth(self):
        case = EvaluationCase(id="q1", question="question?")

        self.assertIsNone(precision_at_k([result("c1")], case, 5))

    def test_precision_at_k_counts_relevant_top_k_results(self):
        case = EvaluationCase(id="q1", question="question?", expected_chunk_ids=["c2"])
        results = [result("c1"), result("c2"), result("c3")]

        self.assertEqual(precision_at_k(results, case, 2), 0.5)

    def test_mrr_uses_first_relevant_rank(self):
        case = EvaluationCase(id="q1", question="question?", expected_chunk_ids=["c3"])
        results = [result("c1"), result("c2"), result("c3")]

        metrics = evaluate_retrieval(results, case)

        self.assertAlmostEqual(metrics.mrr, 1 / 3)

    def test_mean_reciprocal_rank_skips_missing_ground_truth(self):
        cases = [
            EvaluationCase(id="q1", question="question?", expected_chunk_ids=["c2"]),
            EvaluationCase(id="q2", question="question?"),
        ]
        ranked_results = [[result("c1"), result("c2")], [result("c9")]]

        self.assertAlmostEqual(mean_reciprocal_rank(ranked_results, cases), 0.5)


class TestSourceGroundingEvaluation(unittest.TestCase):
    def test_source_grounding_checks_expected_document_and_evidence(self):
        case = EvaluationCase(
            id="q1",
            question="question?",
            expected_document_id="doc-7",
            expected_evidence=["Invoice INV-2026-01847"],
        )
        sources = [result("c1", "Invoice INV-2026-01847 was paid.", document_id="doc-7")]

        grounding = evaluate_source_grounding(sources, case)

        self.assertTrue(grounding.expected_document_found)
        self.assertTrue(grounding.expected_evidence_found)
        self.assertTrue(grounding.grounded)

    def test_source_grounding_reports_missing_ground_truth_as_none(self):
        case = EvaluationCase(id="q1", question="question?")

        grounding = evaluate_source_grounding([result("c1")], case)

        self.assertIsNone(grounding.expected_document_found)
        self.assertIsNone(grounding.expected_evidence_found)
        self.assertIsNone(grounding.grounded)


class TestTimingAggregation(unittest.TestCase):
    def test_latency_aggregator_averages_only_measured_values(self):
        aggregate = LatencyAggregator()
        aggregate.add(LatencyBreakdown(query_embedding_ms=10, generation_ms=None))
        aggregate.add(LatencyBreakdown(query_embedding_ms=30, generation_ms=2000))

        averages = aggregate.averages()

        self.assertEqual(averages["query_embedding_ms"], 20)
        self.assertEqual(averages["generation_ms"], 2000)

    def test_average_optional_returns_none_for_no_values(self):
        self.assertIsNone(average_optional([None, None]))
        self.assertEqual(average_optional([None, 5.0, 15.0]), 10.0)


class TestEvaluationReport(unittest.TestCase):
    def test_empty_dataset_report(self):
        self.assertEqual(
            format_report([]),
            "Questions evaluated: 0\nAnswer accuracy: not automatically evaluated",
        )

    def test_missing_ground_truth_report_does_not_print_fake_retrieval_metrics(self):
        case = EvaluationCase(id="q1", question="question?")
        evaluation = CaseEvaluationResult(
            case=case,
            results=[result("c1")],
            retrieval_metrics=evaluate_retrieval([result("c1")], case),
            source_grounding=evaluate_source_grounding([result("c1")], case),
            latency=LatencyBreakdown(query_embedding_ms=1.0, total_ask_ms=5.0),
        )

        report = format_report([evaluation])

        self.assertIn("Questions evaluated: 1", report)
        self.assertIn("Answer accuracy: not automatically evaluated", report)
        self.assertNotIn("Recall@", report)
        self.assertNotIn("MRR:", report)


class TestEvaluationRunner(unittest.TestCase):
    def test_runner_uses_fakes_and_records_stage_timings(self):
        case = EvaluationCase(
            id="q1",
            question="What revenue was listed?",
            expected_evidence=["Acme,120,45000.50"],
        )
        lexical = FakeRetriever([result("c1", "Acme,120,45000.50")])
        vector = FakeRetriever([result("c1", "Acme,120,45000.50")])
        llm = FakeLLMProvider()
        runner = RAGEvaluationRunner(
            embedding_provider=FakeEmbeddingProvider(),
            lexical_retriever=lexical,
            vector_retriever=vector,
            llm_provider=llm,
        )

        evaluation = runner.evaluate_case(case, top_k=5)

        self.assertEqual(lexical.calls, 1)
        self.assertEqual(vector.calls, 1)
        self.assertEqual(llm.calls, 1)
        self.assertEqual(evaluation.generated_answer, "The answer is grounded in the retrieved source.")
        self.assertEqual(evaluation.retrieval_metrics.recall_at_k[5], 1.0)
        self.assertTrue(evaluation.source_grounding.grounded)
        self.assertIsNotNone(evaluation.latency.query_embedding_ms)
        self.assertIsNotNone(evaluation.latency.lexical_retrieval_ms)
        self.assertIsNotNone(evaluation.latency.vector_retrieval_ms)
        self.assertIsNotNone(evaluation.latency.rrf_ms)
        self.assertIsNotNone(evaluation.latency.hybrid_retrieval_ms)
        self.assertIsNotNone(evaluation.latency.generation_ms)
        self.assertIsNotNone(evaluation.latency.total_ask_ms)


if __name__ == "__main__":
    unittest.main()
