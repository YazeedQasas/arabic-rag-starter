"""Character n-gram ranker: TF-IDF over character 3-grams + cosine similarity.

Where BM25 needs a whole-word match, this compares the *pieces* of words, so it
is forgiving about things normalization can't fix:

    "المحكمة"  ->  _ال  الم  لمح  محك  حكم  كمة  مة_      (n=3, "_" = word edge)
    "بالمحكمة" ->  _با  بال  الم  لمح  محك  حكم  كمة  مة_

The two words share 6 of their n-grams, so they score as similar even though
BM25 sees two different tokens. That makes this ranker a cheap stand-in for a
stemmer: it absorbs Arabic prefixes (ال، و، ب، ل) and suffixes, and small typos.
"""

from __future__ import annotations

from typing import Sequence

from sklearn.feature_extraction.text import TfidfVectorizer

from arabic_rag.ranking import Hit, top_k_hits


class CharNgramRanker:
    """TF-IDF char n-grams, scored by cosine similarity.

    Expects already-normalized text; it does no cleaning of its own.
    """

    def __init__(self, doc_ids: Sequence[str], texts: Sequence[str], n: int = 3):
        self.doc_ids = list(doc_ids)
        # char_wb pads every word with a space on each side, so n-grams never
        # cross word boundaries (the "_" in the docstring above).
        self._vectorizer = TfidfVectorizer(
            analyzer="char_wb", ngram_range=(n, n), lowercase=False
        )
        self._matrix = self._vectorizer.fit_transform(texts)

    def search(self, query_text: str, top_k: int = 10) -> list[Hit]:
        """Most similar documents to a (normalized) query string. Documents that
        share no n-gram with the query are not returned."""
        query = self._vectorizer.transform([query_text])
        # Rows are L2-normalized, so the dot product IS the cosine similarity.
        scores = (self._matrix @ query.T).toarray().ravel()
        return top_k_hits(self.doc_ids, scores, top_k)
