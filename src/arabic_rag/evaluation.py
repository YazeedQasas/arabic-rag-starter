"""Run a retriever over labelled queries and aggregate the metrics.

Also defines the standard configurations the evaluation compares, so the script
and the tests use exactly the same ones.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from arabic_rag.data import Query
from arabic_rag.metrics import mean, recall_at_k, reciprocal_rank
from arabic_rag.normalize import DEFAULT_CONFIG, RULES, NormalizeConfig
from arabic_rag.pipeline import HybridRetriever, RetrieverConfig

# Each query is judged on its top 10 results, so reciprocal rank is really
# "MRR@10": a relevant document ranked 11th or lower counts as not found.
DEFAULT_DEPTH = 10
KS = (1, 3, 5)


@dataclass(frozen=True)
class QueryResult:
    query: Query
    ranked_ids: tuple[str, ...]

    @property
    def found(self) -> bool:
        """True if any relevant document made it into the results at all."""
        return any(doc_id in self.query.relevant_ids for doc_id in self.ranked_ids)


@dataclass(frozen=True)
class Scores:
    n: int
    recall: dict[int, float]  # k -> recall@k
    mrr: float


def run(
    retriever: HybridRetriever, queries: Sequence[Query], depth: int = DEFAULT_DEPTH
) -> list[QueryResult]:
    """Search every query and keep the ranked document ids."""
    return [
        QueryResult(q, tuple(h.doc_id for h in retriever.search(q.text, depth)))
        for q in queries
    ]


def score(results: Sequence[QueryResult], ks: Sequence[int] = KS) -> Scores:
    """recall@k for each k, and MRR, averaged over the queries."""
    return Scores(
        n=len(results),
        recall={
            k: mean([recall_at_k(r.ranked_ids, r.query.relevant_ids, k) for r in results])
            for k in ks
        },
        mrr=mean([reciprocal_rank(r.ranked_ids, r.query.relevant_ids) for r in results]),
    )


def base_kind(kind: str) -> str:
    """'diacritics+alef' counts towards 'diacritics': the part before the '+'."""
    return kind.split("+")[0]


def score_by_kind(results: Sequence[QueryResult]) -> dict[str, Scores]:
    """Scores per query kind, in order of first appearance."""
    groups: dict[str, list[QueryResult]] = {}
    for r in results:
        groups.setdefault(base_kind(r.query.kind), []).append(r)
    return {kind: score(group) for kind, group in groups.items()}


# --- the configurations we compare ------------------------------------------

RAW = NormalizeConfig.none()


def standard_configs() -> dict[str, RetrieverConfig]:
    """BM25 / n-gram / hybrid, each with and without normalization."""
    return {
        "BM25, raw": RetrieverConfig(normalize=RAW, use_ngram=False),
        "BM25, normalized": RetrieverConfig(use_ngram=False),
        "n-gram, raw": RetrieverConfig(normalize=RAW, use_bm25=False),
        "n-gram, normalized": RetrieverConfig(use_bm25=False),
        "Hybrid, raw": RetrieverConfig(normalize=RAW),
        "Hybrid, normalized": RetrieverConfig(),
    }


def ablation_configs(label: str = "Hybrid", **ranker_flags: bool) -> dict[str, RetrieverConfig]:
    """One config per normalization rule, with exactly that rule switched off.

    `ranker_flags` picks the rankers, e.g. `use_ngram=False` for BM25 alone;
    `label` names the rows."""
    return {
        f"{label} without {name}": RetrieverConfig(
            normalize=DEFAULT_CONFIG.without(name), **ranker_flags
        )
        for name, _ in RULES
    }
