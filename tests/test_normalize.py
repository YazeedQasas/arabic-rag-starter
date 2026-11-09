import pytest

from arabic_rag.normalize import (
    RULES,
    NormalizeConfig,
    map_digits,
    normalize,
    normalize_steps,
    strip_diacritics,
    strip_punctuation,
    strip_tatweel,
    unify_alef,
    unify_ta_marbuta,
    unify_ya,
)

# Written as escapes because combining marks are invisible in most editors.
FATHA, DAMMA, KASRA, SHADDA, SUKUN = "َ", "ُ", "ِ", "ّ", "ْ"
SUPERSCRIPT_ALEF = "ٰ"
TATWEEL = "ـ"


# --- one test per rule -------------------------------------------------------


def test_strip_diacritics():
    assert strip_diacritics(f"ك{FATHA}ت{FATHA}ب{FATHA}") == "كتب"
    assert strip_diacritics(f"م{DAMMA}ح{KASRA}ك{SUKUN}م{FATHA}ة{DAMMA}") == "محكمة"
    assert strip_diacritics(f"ال{SHADDA}ل{SUPERSCRIPT_ALEF}ه") == "الله"


def test_strip_tatweel():
    assert strip_tatweel(f"ال{TATWEEL}ع{TATWEEL}دل") == "العدل"


@pytest.mark.parametrize(
    "before, after",
    [("أحمد", "احمد"), ("إسلام", "اسلام"), ("آمن", "امن"), ("ٱلكتاب", "الكتاب")],
)
def test_unify_alef(before, after):
    assert unify_alef(before) == after


def test_unify_ya():
    assert unify_ya("على") == "علي"


def test_unify_ta_marbuta():
    assert unify_ta_marbuta("محكمة") == "محكمه"


def test_map_digits():
    assert map_digits("٢٠٢٤") == "2024"  # Arabic-Indic
    assert map_digits("۲۰۲۴") == "2024"  # Eastern Arabic-Indic


def test_strip_punctuation():
    assert normalize("العدل،  والقانون.") == "العدل والقانون"
    assert normalize("«المادة ١٢؟»") == "الماده 12"  # ة -> ه applies too
    assert "،" not in strip_punctuation("أ،ب")


def test_strip_punctuation_keeps_diacritics_attached():
    # Diacritics are combining marks; punctuation stripping must not split the
    # word around them when diacritic-stripping is off.
    word = f"ك{FATHA}ت{FATHA}ب{FATHA}"
    cfg = NormalizeConfig(strip_diacritics=False)
    assert normalize(word, cfg) == word


def test_nfkc_folds_presentation_forms():
    # Isolated/initial/final presentation forms of م ح ك م ة, as pasted from a PDF
    assert normalize("ﻣﺣﻛﻣﺔ") == normalize("محكمة")


def test_lowercase_latin_only():
    assert normalize("DNA test") == "dna test"


# --- the property that matters ----------------------------------------------


def test_spelling_variants_collapse_to_one_form():
    variants = [
        "المحكمة",
        f"ال{FATHA}م{DAMMA}ح{KASRA}ك{FATHA}م{FATHA}ة{DAMMA}",
        f"المحك{TATWEEL}مة",
        "المحكمه",
        "المحكمة،",
    ]
    assert len({normalize(v) for v in variants}) == 1


def test_hamza_variants_collapse():
    assert normalize("أمانة") == normalize("امانه") == normalize("إمانة")


# --- robustness --------------------------------------------------------------

SAMPLES = [
    "",
    "   ",
    f"ال{FATHA}م{DAMMA}ح{KASRA}ك{FATHA}م{FATHA}ة{DAMMA}، سنة ٢٠٢٤!",
    "على أحمد أن يذهب إلى المدرسة؟",
    "Hello, World! 123",
    "ﻣﺤﻜﻤﺔ ٱلعدل",
]


@pytest.mark.parametrize("text", SAMPLES)
def test_idempotent(text):
    once = normalize(text)
    assert normalize(once) == once


def test_empty_and_non_arabic_input():
    assert normalize("") == ""
    assert normalize("   \n\t ") == ""
    assert normalize("Hello, World!") == "hello world"
    assert normalize("12345") == "12345"


# --- configuration -----------------------------------------------------------


def test_none_config_leaves_text_alone():
    raw = f"ال{FATHA}م{DAMMA}حكمة، ٢٠٢٤ أحمد Hello"
    assert normalize(raw, NormalizeConfig.none()) == raw


@pytest.mark.parametrize(
    "rule, text",
    [
        ("strip_diacritics", f"ك{FATHA}تب"),
        ("strip_tatweel", f"ك{TATWEEL}تب"),
        ("unify_alef", "أحمد"),
        ("unify_ya", "على"),
        ("unify_ta_marbuta", "محكمة"),
        ("map_digits", "٢٠٢٤"),
        ("lowercase", "ABC"),
    ],
)
def test_each_rule_can_be_switched_off(rule, text):
    assert normalize(text, NormalizeConfig().without(rule)) == text
    assert normalize(text) != text  # ...and it does change the text when on


def test_without_rejects_unknown_rule():
    with pytest.raises(ValueError):
        NormalizeConfig().without("not_a_rule")


def test_every_rule_has_a_config_flag():
    assert {name for name, _ in RULES} == set(NormalizeConfig.__dataclass_fields__)


def test_normalize_steps_reports_only_rules_that_changed_text():
    steps = normalize_steps(f"أ{FATHA}حمد")
    assert [name for name, _ in steps] == ["strip_diacritics", "unify_alef"]
    assert steps[-1][1] == "احمد"
