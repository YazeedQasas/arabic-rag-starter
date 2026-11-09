from arabic_rag.bm25 import BM25Ranker
from arabic_rag.fusion import reciprocal_rank_fusion, rrf_score
from arabic_rag.ngram import CharNgramRanker

# --- BM25 --------------------------------------------------------------------

BM25_DOCS = {
    "d1": ["a", "a", "b"],
    "d2": ["a", "b", "b"],
    "d3": ["x"],
    "d4": ["y"],
    "d5": ["z"],
}


def make_bm25():
    return BM25Ranker(list(BM25_DOCS), list(BM25_DOCS.values()))


def test_bm25_more_occurrences_rank_higher():
    hits = make_bm25().search(["a"])
    assert [h.doc_id for h in hits] == ["d1", "d2"]  # tf 2 beats tf 1
    assert hits[0].score > hits[1].score > 0


def test_bm25_drops_documents_with_no_match():
    assert [h.doc_id for h in make_bm25().search(["z"])] == ["d5"]


def test_bm25_unknown_or_empty_query_returns_nothing():
    assert make_bm25().search(["nope"]) == []
    assert make_bm25().search([]) == []


def test_bm25_respects_top_k():
    assert len(make_bm25().search(["a"], top_k=1)) == 1


def test_bm25_needs_exact_tokens():
    # The reason normalization exists: near-identical spellings do not match.
    ranker = BM25Ranker(["d1", "d2", "d3"], [["المحكمة"], ["x"], ["y"]])
    assert ranker.search(["المحكمه"]) == []
    assert [h.doc_id for h in ranker.search(["المحكمة"])] == ["d1"]


# --- character n-grams -------------------------------------------------------

NGRAM_TEXTS = {
    "d1": "المحكمه العليا",
    "d2": "الزراعه العضويه",
    "d3": "كره القدم",
}


def make_ngram():
    return CharNgramRanker(list(NGRAM_TEXTS), list(NGRAM_TEXTS.values()))


def test_ngram_matches_through_a_typo():
    # "محكمم" is not a token in any document, but shares n-grams with المحكمه.
    hits = make_ngram().search("محكمم")
    assert hits[0].doc_id == "d1"


def test_ngram_matches_through_a_prefix():
    ranker = CharNgramRanker(["d1", "d2"], ["محكمه عليا", "زراعه عضويه"])
    assert ranker.search("بالمحكمه")[0].doc_id == "d1"


def test_ngram_unrelated_documents_are_not_returned():
    ids = [h.doc_id for h in make_ngram().search("محكمم")]
    assert "d2" not in ids and "d3" not in ids


def test_ngram_empty_query_returns_nothing():
    assert make_ngram().search("") == []


def test_ngram_scores_are_cosines():
    hits = CharNgramRanker(["d1"], ["محكمه"]).search("محكمه")
    assert abs(hits[0].score - 1.0) < 1e-9  # identical text -> cosine 1


# --- reciprocal-rank fusion --------------------------------------------------


def test_rrf_score_formula():
    assert rrf_score(1, k=60) == 1 / 61
    assert rrf_score(3, k=60) == 1 / 63


def test_rrf_hand_computed():
    hits = reciprocal_rank_fusion([["a", "b"], ["b", "c"]], k=60)
    scores = {h.doc_id: h.score for h in hits}
    assert scores["a"] == 1 / 61
    assert abs(scores["b"] - (1 / 62 + 1 / 61)) < 1e-12
    assert scores["c"] == 1 / 62
    assert [h.doc_id for h in hits] == ["b", "a", "c"]  # found by both beats rank-1 of one


def test_rrf_ties_break_by_first_appearance():
    hits = reciprocal_rank_fusion([["a", "b"], ["b", "a"]])
    assert [h.doc_id for h in hits] == ["a", "b"]
    assert hits[0].score == hits[1].score


def test_rrf_single_ranking_keeps_its_order():
    assert [h.doc_id for h in reciprocal_rank_fusion([["x", "y", "z"]])] == ["x", "y", "z"]


def test_rrf_empty_input():
    assert reciprocal_rank_fusion([]) == []
    assert reciprocal_rank_fusion([[], []]) == []


def test_rrf_accepts_any_number_of_rankings():
    hits = reciprocal_rank_fusion([["a", "b"], ["a", "c"], ["a", "d"]])
    assert hits[0].doc_id == "a"
    assert abs(hits[0].score - 3 / 61) < 1e-12
