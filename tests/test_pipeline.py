import pytest

from arabic_rag.data import DATA_DIR, load_corpus, load_queries
from arabic_rag.normalize import NormalizeConfig
from arabic_rag.pipeline import HybridRetriever, RetrieverConfig


@pytest.fixture(scope="module")
def docs():
    return load_corpus()


@pytest.fixture(scope="module")
def retriever(docs):
    return HybridRetriever(docs)


# --- the bundled data --------------------------------------------------------


def test_data_files_exist():
    assert (DATA_DIR / "corpus.jsonl").exists()
    assert (DATA_DIR / "queries.jsonl").exists()


def test_corpus_is_well_formed(docs):
    ids = [d.id for d in docs]
    assert len(ids) == len(set(ids)), "duplicate document ids"
    assert len(docs) >= 100
    assert all(d.text.strip() for d in docs)


def test_queries_point_at_real_documents(docs):
    ids = {d.id for d in docs}
    queries = load_queries()
    assert len(queries) >= 60
    for q in queries:
        assert q.text.strip()
        assert q.relevant_ids, f"query has no relevant ids: {q.text}"
        assert q.relevant_ids <= ids, f"unknown relevant id in: {q.text}"
        assert q.kind


def test_every_query_kind_has_enough_examples():
    # "diacritics+alef" counts towards "diacritics": the part before the "+".
    counts = {}
    for q in load_queries():
        kind = q.kind.split("+")[0]
        counts[kind] = counts.get(kind, 0) + 1
    assert {"exact", "exact_rare", "natural", "diacritics", "alef", "ta_marbuta",
            "ya", "tatweel", "digits", "morphology", "typo", "confusable"} <= set(counts)
    assert all(n >= 4 for n in counts.values()), counts


# --- the pipeline ------------------------------------------------------------


def test_smoke_known_query_finds_its_document(retriever):
    assert retriever.search("الاراضي والمساحه")[0].doc_id == "d14"


def test_diacritized_document_is_found_by_plain_query(retriever):
    # d03 is fully vowelled; the query has no diacritics.
    assert retriever.search("الشهادة زورا")[0].doc_id == "d03"


def test_without_normalization_the_same_query_silently_fails(docs):
    raw = HybridRetriever(
        docs, RetrieverConfig(normalize=NormalizeConfig.none(), use_ngram=False)
    )
    # Raw BM25 sees no matching token: no results, and no error.
    assert raw.search("الاراضي والمساحه") == []


def test_search_respects_top_k(retriever):
    assert len(retriever.search("المحكمة", top_k=3)) <= 3


def test_empty_query_returns_nothing(retriever):
    assert retriever.search("") == []


def test_trace_exposes_every_step(retriever):
    trace = retriever.trace("الأراضي والمساحة")
    assert trace.query == "الأراضي والمساحة"
    assert trace.normalized_query == "الاراضي والمساحه"
    assert trace.tokens == ["الاراضي", "والمساحه"]
    assert set(trace.rankings) == {"bm25", "ngram"}
    top = trace.fused[0]
    assert top.doc_id == "d14"
    # both rankers contributed to the winner, and the contributions add up
    parts = trace.contributions["d14"]
    assert set(parts) == {"bm25", "ngram"}
    assert abs(sum(c for _, c in parts.values()) - top.score) < 1e-12


def test_single_ranker_configs(docs):
    bm25_only = HybridRetriever(docs, RetrieverConfig(use_ngram=False))
    assert set(bm25_only.trace("المحكمة").rankings) == {"bm25"}
    ngram_only = HybridRetriever(docs, RetrieverConfig(use_bm25=False))
    assert set(ngram_only.trace("المحكمة").rankings) == {"ngram"}


def test_needs_at_least_one_ranker(docs):
    with pytest.raises(ValueError):
        HybridRetriever(docs, RetrieverConfig(use_bm25=False, use_ngram=False))


def test_morphological_variant_is_rescued_by_ngrams(retriever):
    # "المتهمون والمحامون" shares no token with d07 (للمتهم / بمحام).
    trace = retriever.trace("المتهمون والمحامون")
    assert trace.rankings["bm25"] == []
    assert trace.fused[0].doc_id == "d07"
