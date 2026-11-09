import pytest

from arabic_rag.metrics import mean, recall_at_k, reciprocal_rank

RANKED = ["a", "b", "c", "d"]


def test_recall_single_relevant_is_a_hit_rate():
    assert recall_at_k(RANKED, {"b"}, 1) == 0.0
    assert recall_at_k(RANKED, {"b"}, 2) == 1.0
    assert recall_at_k(RANKED, {"b"}, 4) == 1.0


def test_recall_multiple_relevant_is_a_fraction():
    relevant = {"b", "d", "x"}  # "x" is never retrieved
    assert recall_at_k(RANKED, relevant, 1) == 0.0
    assert recall_at_k(RANKED, relevant, 3) == pytest.approx(1 / 3)
    assert recall_at_k(RANKED, relevant, 4) == pytest.approx(2 / 3)


def test_recall_nothing_relevant_retrieved():
    assert recall_at_k(RANKED, {"z"}, 4) == 0.0
    assert recall_at_k([], {"z"}, 3) == 0.0


def test_recall_k_larger_than_the_list_is_fine():
    assert recall_at_k(RANKED, {"d"}, 100) == 1.0


def test_recall_validates_input():
    with pytest.raises(ValueError):
        recall_at_k(RANKED, set(), 3)
    with pytest.raises(ValueError):
        recall_at_k(RANKED, {"a"}, 0)


def test_reciprocal_rank():
    assert reciprocal_rank(RANKED, {"a"}) == 1.0
    assert reciprocal_rank(RANKED, {"b"}) == 0.5
    assert reciprocal_rank(RANKED, {"d"}) == 0.25


def test_reciprocal_rank_uses_the_first_relevant_result_only():
    assert reciprocal_rank(RANKED, {"c", "d"}) == pytest.approx(1 / 3)


def test_reciprocal_rank_when_nothing_is_retrieved():
    assert reciprocal_rank(RANKED, {"z"}) == 0.0
    assert reciprocal_rank([], {"a"}) == 0.0


def test_mrr_is_the_mean_of_reciprocal_ranks():
    ranks = [reciprocal_rank(RANKED, {"a"}), reciprocal_rank(RANKED, {"b"}), reciprocal_rank(RANKED, {"z"})]
    assert mean(ranks) == pytest.approx((1 + 0.5 + 0) / 3)


def test_mean_of_nothing_is_zero():
    assert mean([]) == 0.0
