import unittest

from app.retrieval.models import RetrievalResult
from app.retrieval.rrf import RRFConfig, reciprocal_rank_fusion


def result(chunk_id: str, score: float = 1.0, document_id: str = "doc-1") -> RetrievalResult:
    return RetrievalResult(
        chunk_id=chunk_id,
        document_id=document_id,
        content=f"content for {chunk_id}",
        score=score,
        metadata={},
    )


class TestReciprocalRankFusionTwoLists(unittest.TestCase):
    def test_two_result_lists_are_merged(self):
        lexical = [result("c1"), result("c2"), result("c3")]
        vector = [result("c2"), result("c4")]
        fused = reciprocal_rank_fusion([lexical, vector])
        chunk_ids = {r.chunk_id for r in fused}
        self.assertEqual(chunk_ids, {"c1", "c2", "c3", "c4"})

    def test_duplicate_chunk_ids_are_merged_not_repeated(self):
        lexical = [result("c1"), result("c2")]
        vector = [result("c2"), result("c1")]
        fused = reciprocal_rank_fusion([lexical, vector])
        chunk_ids = [r.chunk_id for r in fused]
        self.assertEqual(len(chunk_ids), len(set(chunk_ids)))

    def test_result_in_both_lists_receives_both_contributions(self):
        k = 60
        lexical = [result("shared"), result("only_lexical")]  # shared rank=1
        vector = [result("only_vector"), result("shared")]  # shared rank=2
        fused = reciprocal_rank_fusion([lexical, vector], RRFConfig(k=k, top_k=10))
        shared = next(r for r in fused if r.chunk_id == "shared")
        expected_score = 1 / (k + 1) + 1 / (k + 2)
        self.assertAlmostEqual(shared.score, expected_score)

    def test_result_in_only_one_list_remains_eligible(self):
        lexical = [result("only_lexical")]
        vector = [result("only_vector")]
        fused = reciprocal_rank_fusion([lexical, vector])
        chunk_ids = {r.chunk_id for r in fused}
        self.assertIn("only_lexical", chunk_ids)
        self.assertIn("only_vector", chunk_ids)

    def test_result_in_only_one_list_gets_single_contribution(self):
        k = 60
        lexical = [result("a"), result("only_lexical")]  # only_lexical rank=2
        vector: list[RetrievalResult] = []
        fused = reciprocal_rank_fusion([lexical, vector], RRFConfig(k=k, top_k=10))
        only = next(r for r in fused if r.chunk_id == "only_lexical")
        self.assertAlmostEqual(only.score, 1 / (k + 2))

    def test_results_appearing_in_both_lists_outrank_single_list_results(self):
        lexical = [result("shared"), result("solo_lex")]
        vector = [result("shared"), result("solo_vec")]
        fused = reciprocal_rank_fusion([lexical, vector])
        self.assertEqual(fused[0].chunk_id, "shared")

    def test_sorted_by_descending_score(self):
        lexical = [result("c1"), result("c2"), result("c3")]
        vector = [result("c3"), result("c1")]
        fused = reciprocal_rank_fusion([lexical, vector])
        scores = [r.score for r in fused]
        self.assertEqual(scores, sorted(scores, reverse=True))


class TestReciprocalRankFusionConfig(unittest.TestCase):
    def test_configurable_k_changes_scores(self):
        lexical = [result("c1")]
        low_k = reciprocal_rank_fusion([lexical], RRFConfig(k=1, top_k=10))
        high_k = reciprocal_rank_fusion([lexical], RRFConfig(k=1000, top_k=10))
        self.assertGreater(low_k[0].score, high_k[0].score)

    def test_default_k_is_60(self):
        self.assertEqual(RRFConfig().k, 60)

    def test_configurable_top_k_limits_results(self):
        lexical = [result(f"c{i}") for i in range(20)]
        fused = reciprocal_rank_fusion([lexical], RRFConfig(k=60, top_k=5))
        self.assertEqual(len(fused), 5)

    def test_top_k_keeps_highest_scoring_results(self):
        lexical = [result(f"c{i}") for i in range(10)]
        full = reciprocal_rank_fusion([lexical], RRFConfig(k=60, top_k=100))
        limited = reciprocal_rank_fusion([lexical], RRFConfig(k=60, top_k=3))
        self.assertEqual([r.chunk_id for r in limited], [r.chunk_id for r in full[:3]])

    def test_default_top_k_is_10(self):
        self.assertEqual(RRFConfig().top_k, 10)

    def test_rejects_non_positive_k(self):
        from pydantic import ValidationError

        with self.assertRaises(ValidationError):
            RRFConfig(k=0)

    def test_rejects_non_positive_top_k(self):
        from pydantic import ValidationError

        with self.assertRaises(ValidationError):
            RRFConfig(top_k=0)


class TestReciprocalRankFusionDeterminism(unittest.TestCase):
    def test_exact_tie_breaks_by_chunk_id_ascending(self):
        # both chunks appear at rank 1 in their own single-item list ->
        # identical RRF scores, so the tiebreak must be deterministic.
        fused = reciprocal_rank_fusion([[result("zebra")], [result("alpha")]])
        self.assertEqual(fused[0].chunk_id, "alpha")
        self.assertEqual(fused[1].chunk_id, "zebra")

    def test_repeated_calls_produce_identical_ordering(self):
        lexical = [result("c1"), result("c2"), result("c3")]
        vector = [result("c3"), result("c2"), result("c1")]
        first = [r.chunk_id for r in reciprocal_rank_fusion([lexical, vector])]
        second = [r.chunk_id for r in reciprocal_rank_fusion([lexical, vector])]
        self.assertEqual(first, second)


class TestReciprocalRankFusionFieldPreservation(unittest.TestCase):
    def test_document_id_and_content_and_metadata_preserved(self):
        original = RetrievalResult(
            chunk_id="c1",
            document_id="doc-7",
            content="Invoice INV-2026-01847 was issued in July.",
            score=0.5,
            metadata={"element_type": "text", "page_number": 3},
        )
        fused = reciprocal_rank_fusion([[original]])
        self.assertEqual(fused[0].document_id, "doc-7")
        self.assertEqual(fused[0].content, "Invoice INV-2026-01847 was issued in July.")
        self.assertEqual(fused[0].metadata, {"element_type": "text", "page_number": 3})

    def test_empty_result_lists_produce_empty_output(self):
        self.assertEqual(reciprocal_rank_fusion([]), [])
        self.assertEqual(reciprocal_rank_fusion([[], []]), [])


if __name__ == "__main__":
    unittest.main()
