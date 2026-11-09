"""HybridRetriever: the whole pipeline in one place.

    query -> normalize -> tokenize -> BM25        -+
                       \\-> char n-grams (TF-IDF)  -+-> reciprocal-rank fusion -> results

The SAME normalization is applied to the corpus (at indexing time) and to every
query. If the two sides were normalized differently, matches would silently
vanish; that is the failure this project is about.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

from arabic_rag.bm25 import BM25Ranker
from arabic_rag.data import Document
from arabic_rag.fusion import DEFAULT_K, reciprocal_rank_fusion, rrf_score
from arabic_rag.ngram import CharNgramRanker
from arabic_rag.normalize import DEFAULT_CONFIG, NormalizeConfig, normalize
from arabic_rag.ranking import Hit
from arabic_rag.tokenizer import tokenize


@dataclass(frozen=True)
class RetrieverConfig:
    normalize: NormalizeConfig = DEFAULT_CONFIG
    use_bm25: bool = True
    use_ngram: bool = True
    ngram_n: int = 3
    rrf_k: int = DEFAULT_K
    candidates: int = 50  # how deep each ranker's list is before fusion


@dataclass
class SearchTrace:
    """Every intermediate result of one search, for the demo script."""

    query: str
    normalized_query: str
    tokens: list[str]
    rankings: dict[str, list[Hit]]  # ranker name -> its hits, best first
    fused: list[Hit]
    # doc_id -> {ranker name: (rank, rrf contribution)}
    contributions: dict[str, dict[str, tuple[int, float]]] = field(default_factory=dict)


class HybridRetriever:
    """Index a corpus once, then search it.

    Turn individual pieces off through the config (`use_bm25`, `use_ngram`,
    `normalize`) to build the baselines the evaluation script compares.
    """

    def __init__(self, documents: Sequence[Document], config: RetrieverConfig = RetrieverConfig()):
        if not config.use_bm25 and not config.use_ngram:
            raise ValueError("enable at least one ranker")
        self.config = config
        self.doc_ids = [d.id for d in documents]
        self._bm25 = None
        self._ngram = None

        if config.use_bm25:
            tokenized = [tokenize(d.text, config.normalize) for d in documents]
            self._bm25 = BM25Ranker(self.doc_ids, tokenized)
        if config.use_ngram:
            normalized = [normalize(d.text, config.normalize) for d in documents]
            self._ngram = CharNgramRanker(self.doc_ids, normalized, n=config.ngram_n)

    def trace(self, query: str, top_k: int = 10) -> SearchTrace:
        """Run a search and keep every intermediate step."""
        cfg = self.config
        normalized_query = normalize(query, cfg.normalize)
        tokens = tokenize(query, cfg.normalize)

        rankings: dict[str, list[Hit]] = {}
        if self._bm25 is not None:
            rankings["bm25"] = self._bm25.search(tokens, cfg.candidates)
        if self._ngram is not None:
            rankings["ngram"] = self._ngram.search(normalized_query, cfg.candidates)

        fused = reciprocal_rank_fusion(
            [[h.doc_id for h in hits] for hits in rankings.values()], k=cfg.rrf_k
        )

        contributions: dict[str, dict[str, tuple[int, float]]] = {}
        for name, hits in rankings.items():
            for rank, hit in enumerate(hits, start=1):
                contributions.setdefault(hit.doc_id, {})[name] = (
                    rank,
                    rrf_score(rank, cfg.rrf_k),
                )

        return SearchTrace(query, normalized_query, tokens, rankings, fused[:top_k], contributions)

    def search(self, query: str, top_k: int = 10) -> list[Hit]:
        """Best documents for `query`, fused across the enabled rankers."""
        return self.trace(query, top_k).fused
