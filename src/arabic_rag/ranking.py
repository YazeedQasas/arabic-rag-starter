"""Shared helper: turn a score per document into a ranked top-k list."""

from __future__ import annotations

from typing import NamedTuple, Sequence


class Hit(NamedTuple):
    doc_id: str
    score: float


def top_k_hits(doc_ids: Sequence[str], scores: Sequence[float], top_k: int) -> list[Hit]:
    """Rank documents by score, best first.

    Two deliberate choices:
    - Documents with a score of 0 are dropped. A ranker should only return
      documents it actually matched; otherwise a query with no matching terms
      would "rank" every document in arbitrary order and that noise would leak
      into the fused ranking.
    - Ties break by corpus order (the sort is stable), so results are
      deterministic.
    """
    order = sorted(range(len(doc_ids)), key=lambda i: -scores[i])
    return [Hit(doc_ids[i], float(scores[i])) for i in order[:top_k] if scores[i] > 0]
