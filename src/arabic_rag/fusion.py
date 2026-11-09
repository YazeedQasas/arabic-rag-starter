"""Reciprocal-rank fusion (RRF): merge several ranked lists into one.

Each ranker gives a document the score 1 / (k + rank), where rank starts at 1,
and a document's fused score is the sum over rankers:

    BM25 rank 1, n-gram rank 3  ->  1/61 + 1/63

Why this works: it only looks at ranks, never at raw scores. BM25 scores and
cosine similarities live on completely different scales, so adding them would
need calibration; ranks are already comparable. A document that BOTH rankers
like beats one that only a single ranker puts first. k=60 (from the original
paper) flattens the difference between rank 1 and rank 2 so one ranker can't
dominate.
"""

from __future__ import annotations

from typing import Sequence

from arabic_rag.ranking import Hit

DEFAULT_K = 60


def rrf_score(rank: int, k: int = DEFAULT_K) -> float:
    """One ranker's contribution for a document at 1-based `rank`."""
    return 1.0 / (k + rank)


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[str]], k: int = DEFAULT_K
) -> list[Hit]:
    """Fuse any number of ranked lists of document ids (best first).

    Ties break by first appearance: the document seen earlier, scanning the
    lists in the order given, wins. That keeps results deterministic.
    """
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + rrf_score(rank, k)
    ordered = sorted(scores, key=lambda doc_id: -scores[doc_id])  # stable sort
    return [Hit(doc_id, scores[doc_id]) for doc_id in ordered]
