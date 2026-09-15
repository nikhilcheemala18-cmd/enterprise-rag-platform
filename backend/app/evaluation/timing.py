from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from time import perf_counter


@dataclass(frozen=True)
class LatencyBreakdown:
    query_embedding_ms: float | None = None
    lexical_retrieval_ms: float | None = None
    vector_retrieval_ms: float | None = None
    rrf_ms: float | None = None
    hybrid_retrieval_ms: float | None = None
    generation_ms: float | None = None
    total_ask_ms: float | None = None


class Timer:
    def __enter__(self) -> "Timer":
        self._started_at = perf_counter()
        self.elapsed_ms = 0.0
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.elapsed_ms = (perf_counter() - self._started_at) * 1000


class LatencyAggregator:
    def __init__(self) -> None:
        self._values: dict[str, list[float]] = defaultdict(list)

    def add(self, breakdown: LatencyBreakdown) -> None:
        for name, value in breakdown.__dict__.items():
            if value is not None:
                self._values[name].append(value)

    def averages(self) -> dict[str, float]:
        return {
            name: sum(values) / len(values)
            for name, values in self._values.items()
            if values
        }


def average_optional(values: Iterable[float | None]) -> float | None:
    measured = [value for value in values if value is not None]
    if not measured:
        return None
    return sum(measured) / len(measured)
