"""Retrieval metrics. Each takes a ranked list of document ids (best first) and
the set of ids that are actually relevant."""

from __future__ import annotations

from typing import Collection, Sequence


def recall_at_k(ranked: Sequence[str], relevant: Collection[str], k: int) -> float:
    """Fraction of the relevant documents that appear in the top `k` results.

    With one relevant document per query (as in the bundled data) this is simply
    1 if it is in the top k and 0 if not (a "hit rate").
    """
    if not relevant:
        raise ValueError("a query needs at least one relevant document")
    if k < 1:
        raise ValueError("k must be at least 1")
    return len(set(ranked[:k]) & set(relevant)) / len(set(relevant))


def reciprocal_rank(ranked: Sequence[str], relevant: Collection[str]) -> float:
    """1 / rank of the first relevant result (1.0 if first, 0.5 if second ...),
    or 0 if no relevant document was retrieved at all."""
    for rank, doc_id in enumerate(ranked, start=1):
        if doc_id in relevant:
            return 1.0 / rank
    return 0.0


def mean(values: Sequence[float]) -> float:
    """Average of `values`; MRR is just `mean` of `reciprocal_rank` over queries."""
    return sum(values) / len(values) if values else 0.0
