import os
import subprocess
import sys
from pathlib import Path

import pytest

from arabic_rag.data import Query, load_corpus, load_queries
from arabic_rag.evaluation import (
    QueryResult,
    ablation_configs,
    base_kind,
    run,
    score,
    score_by_kind,
    standard_configs,
)
from arabic_rag.normalize import RULES
from arabic_rag.pipeline import HybridRetriever

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def docs():
    return load_corpus()


@pytest.fixture(scope="module")
def queries():
    return load_queries()


@pytest.fixture(scope="module")
def mrr(docs, queries):
    """MRR of each standard configuration on the bundled data."""
    return {
        name: score(run(HybridRetriever(docs, cfg), queries)).mrr
        for name, cfg in standard_configs().items()
    }


# --- harness mechanics -------------------------------------------------------


def make_result(ranked, relevant, kind="exact"):
    return QueryResult(Query("q", frozenset(relevant), kind), tuple(ranked))


def test_score_averages_over_queries():
    results = [make_result(["a", "b"], {"a"}), make_result(["a", "b"], {"b"}), make_result(["a"], {"z"})]
    s = score(results, ks=(1, 2))
    assert s.n == 3
    assert s.recall == {1: pytest.approx(1 / 3), 2: pytest.approx(2 / 3)}
    assert s.mrr == pytest.approx((1 + 0.5 + 0) / 3)


def test_found_means_a_relevant_document_was_retrieved():
    assert make_result(["a", "b"], {"b"}).found
    assert not make_result(["a", "b"], {"z"}).found


def test_kinds_group_by_the_part_before_the_plus():
    assert base_kind("diacritics+alef") == "diacritics"
    results = [make_result(["a"], {"a"}, "alef"), make_result(["b"], {"a"}, "alef+ta_marbuta")]
    by_kind = score_by_kind(results)
    assert list(by_kind) == ["alef"]
    assert by_kind["alef"].n == 2 and by_kind["alef"].mrr == 0.5


def test_run_judges_only_the_requested_depth(docs, queries):
    results = run(HybridRetriever(docs), queries, depth=3)
    assert all(len(r.ranked_ids) <= 3 for r in results)


def test_one_ablation_config_per_normalization_rule():
    assert len(ablation_configs()) == len(RULES)
    assert all(not cfg.use_ngram for cfg in ablation_configs("BM25", use_ngram=False).values())


# --- the headline claims, kept as regression checks on the bundled data ---


def test_normalization_gives_a_clear_gain_over_the_raw_baseline(mrr):
    assert mrr["BM25, normalized"] - mrr["BM25, raw"] > 0.3
    assert mrr["Hybrid, normalized"] - mrr["Hybrid, raw"] > 0.15


def test_hybrid_is_at_least_as_good_as_either_single_ranker(mrr):
    assert mrr["Hybrid, normalized"] >= mrr["BM25, normalized"]
    assert mrr["Hybrid, normalized"] >= mrr["n-gram, normalized"]


def test_removing_diacritic_stripping_hurts(docs, queries):
    full = score(run(HybridRetriever(docs, standard_configs()["BM25, normalized"]), queries)).mrr
    ablated_cfg = ablation_configs("BM25", use_ngram=False)["BM25 without strip_diacritics"]
    assert score(run(HybridRetriever(docs, ablated_cfg), queries)).mrr < full


# --- the script ---------------------------------------------------------------


def test_evaluate_script_writes_a_report(tmp_path):
    out = tmp_path / "results.md"
    env = {**os.environ, "PYTHONPATH": str(ROOT / "src"), "PYTHONIOENCODING": "utf-8"}
    proc = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "evaluate.py"), "--out", str(out)],
        capture_output=True, text=True, encoding="utf-8", env=env, cwd=tmp_path,
    )
    assert proc.returncode == 0, proc.stderr
    report = out.read_text(encoding="utf-8")
    for heading in ("Retrieval configurations", "MRR by query kind", "Ablation", "Silent failures"):
        assert heading in report
    assert "Hybrid, normalized" in report
