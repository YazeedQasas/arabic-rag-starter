from arabic_rag.normalize import NormalizeConfig
from arabic_rag.tokenizer import tokenize

FATHA = "َ"


def test_splits_on_whitespace_and_punctuation():
    assert tokenize("العدل، والقانون.") == ["العدل", "والقانون"]


def test_normalizes_before_splitting():
    assert tokenize("المحكمةُ العُليا") == ["المحكمه", "العليا"]


def test_variant_spellings_give_identical_tokens():
    assert tokenize("أحكام المحكمة") == tokenize("احكام المحكمه")


def test_latin_words_and_numbers_survive():
    assert tokenize("قانون GDPR رقم ١٢ لسنة 2024") == [
        "قانون",
        "gdpr",
        "رقم",
        "12",
        "لسنه",
        "2024",
    ]


def test_hyphenated_latin_splits():
    assert tokenize("COVID-19") == ["covid", "19"]


def test_empty_and_punctuation_only_give_no_tokens():
    assert tokenize("") == []
    assert tokenize("  ،؛!؟  ") == []


def test_no_normalization_baseline_keeps_surface_forms():
    word = f"ك{FATHA}ت{FATHA}ب{FATHA}"
    # Raw baseline: punctuation still splits tokens, but spelling is untouched,
    # and diacritics stay attached to their word.
    assert tokenize(f"{word}، أحمد", NormalizeConfig.none()) == [word, "أحمد"]
