import json
from pathlib import Path

from pydantic import BaseModel, Field


class EvaluationCase(BaseModel):
    """One deterministic RAG benchmark question.

    expected_answer is descriptive benchmark context for humans. The
    initial evaluator does not grade answer correctness with an LLM.
    """

    id: str = Field(min_length=1)
    question: str = Field(min_length=1)
    expected_answer: str | None = None
    expected_document_id: str | None = Field(default=None, min_length=1)
    expected_chunk_ids: list[str] = Field(default_factory=list)
    expected_evidence: list[str] = Field(default_factory=list)


def default_dataset_path() -> Path:
    return Path(__file__).parent / "datasets" / "sample_csv.json"


def load_evaluation_dataset(path: str | Path | None = None) -> list[EvaluationCase]:
    dataset_path = Path(path) if path is not None else default_dataset_path()
    payload = json.loads(dataset_path.read_text(encoding="utf-8"))

    if isinstance(payload, dict):
        raw_cases = payload.get("cases", [])
    else:
        raw_cases = payload

    return [EvaluationCase.model_validate(item) for item in raw_cases]
