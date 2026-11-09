"""BM25 ranker: a thin wrapper around the `rank_bm25` library.

BM25 is a bag-of-words score: a document scores higher the more query terms it
contains, with diminishing returns for repeats (k1) and a penalty for long
documents (b). It only ever matches EXACT tokens, which is why the text
pipeline in front of it matters so much: "المحكمة" and "المحكمه" are completely
unrelated strings to BM25 unless normalization maps them to the same token.
"""

from __future__ import annotations

from typing import Sequence

from rank_bm25 import BM25Okapi

from arabic_rag.ranking import Hit, top_k_hits


class BM25Ranker:
    """Okapi BM25 over pre-tokenized documents.

    Uses rank_bm25's defaults: k1=1.5, b=0.75, and its idf variant
    log((N - n + 0.5) / (n + 0.5)). That idf goes negative for terms found in
    more than half of the documents; rank_bm25 replaces negative values with
    0.25 * the average idf, so very common words add little but never subtract.
    """

    def __init__(
        self,
        doc_ids: Sequence[str],
        tokenized_docs: Sequence[Sequence[str]],
        k1: float = 1.5,
        b: float = 0.75,
    ):
        self.doc_ids = list(doc_ids)
        self._bm25 = BM25Okapi([list(d) for d in tokenized_docs], k1=k1, b=b)

    def search(self, query_tokens: Sequence[str], top_k: int = 10) -> list[Hit]:
        """Best-matching documents for a tokenized query. Documents that share
        no token with the query are not returned."""
        scores = self._bm25.get_scores(list(query_tokens))
        return top_k_hits(self.doc_ids, scores, top_k)
