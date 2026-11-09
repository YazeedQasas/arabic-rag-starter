"""Arabic text normalization.

Arabic retrieval usually fails silently here: the same word can be written
several valid ways (with or without diacritics, with different alef forms,
ة or ه ...), and a search engine compares raw characters. If the document says
"المُحَاكَمَة" and the query says "المحاكمه", nothing matches and nothing errors.

The fix is to rewrite text into one canonical form, using the SAME function for
the corpus and for the queries. Every rule below is a tiny function so you can
read it on its own, and every rule can be switched off through
`NormalizeConfig`, which is what the ablation in scripts/evaluate.py does.

All rules are deliberately lossy. We trade a few ambiguous words (على "on" vs
علي "Ali") for far fewer missed matches.
"""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass, replace

# --- character tables --------------------------------------------------------

# Short-vowel marks and friends: fathatan..sukun (U+064B-U+0652), the extra
# combining marks up to U+065F, and the superscript alef (U+0670).
_DIACRITICS = {chr(c): None for c in range(0x064B, 0x0660)} | {"ٰ": None}

# أ إ آ ٱ -> ا
_ALEF_VARIANTS = {c: "ا" for c in "أإآٱ"}

# Arabic-Indic (٠-٩) and Eastern Arabic-Indic (۰-۹) digits -> 0-9
_DIGITS = {chr(0x0660 + i): str(i) for i in range(10)} | {
    chr(0x06F0 + i): str(i) for i in range(10)
}

# --- rules: one small function each -----------------------------------------


def nfkc(text: str) -> str:
    """Unicode NFKC. Folds Arabic presentation forms (common in text copied out
    of PDFs, e.g. ﻣﺤﻜﻤﺔ) back to the ordinary letters."""
    return unicodedata.normalize("NFKC", text)


def lowercase(text: str) -> str:
    """Lowercase Latin letters. Arabic has no case, so this only touches the
    English words and acronyms that show up in Arabic text."""
    return text.lower()


def strip_tatweel(text: str) -> str:
    """Remove tatweel/kashida (ـ), a decorative stretch: الـعـدل -> العدل."""
    return text.replace("ـ", "")


def strip_diacritics(text: str) -> str:
    """Remove tashkeel: كَتَبَ -> كتب. Documents (books, laws, Quran-style text)
    often carry them; queries almost never do."""
    return text.translate(str.maketrans(_DIACRITICS))


def unify_alef(text: str) -> str:
    """أ إ آ ٱ -> ا. People type hamza inconsistently, especially on phones."""
    return text.translate(str.maketrans(_ALEF_VARIANTS))


def unify_ya(text: str) -> str:
    """ى -> ي. Alef maqsura and ya look alike and are often swapped
    (Egyptian-style writing). Lossy: على ("on") becomes علي ("Ali")."""
    return text.replace("ى", "ي")


def unify_ta_marbuta(text: str) -> str:
    """ة -> ه. Ta marbuta and ha are the other classic swap: محكمة -> محكمه.
    The direction is arbitrary (some pipelines map ه -> ة); what matters is that
    both spellings collapse to one form. Lossy for rare words that differ only
    in this letter."""
    return text.replace("ة", "ه")


def map_digits(text: str) -> str:
    """Arabic-Indic digits -> ASCII: ٢٠٢٤ -> 2024."""
    return text.translate(str.maketrans(_DIGITS))


def strip_punctuation(text: str) -> str:
    """Replace punctuation and symbols with spaces (، ؛ ؟ « » . , ! ...).

    Keeps letters (L), numbers (N), combining marks (M) and whitespace. Marks
    must be kept: tashkeel is category Mn, so a naive "not a word character"
    regex would turn كَتَبَ into "ك ت ب" whenever diacritic-stripping is off."""
    return "".join(
        ch if ch.isspace() or unicodedata.category(ch)[0] in "LNM" else " "
        for ch in text
    )


def collapse_whitespace(text: str) -> str:
    """Squeeze runs of whitespace into single spaces and trim the ends."""
    return " ".join(text.split())


# --- configuration and composition ------------------------------------------


@dataclass(frozen=True)
class NormalizeConfig:
    """Which rules run. Field names match the rule function names."""

    nfkc: bool = True
    lowercase: bool = True
    strip_tatweel: bool = True
    strip_diacritics: bool = True
    unify_alef: bool = True
    unify_ya: bool = True
    unify_ta_marbuta: bool = True
    map_digits: bool = True
    strip_punctuation: bool = True

    @classmethod
    def none(cls) -> "NormalizeConfig":
        """Every rule off: the 'no normalization' baseline."""
        return cls(**{name: False for name, _ in RULES})

    def without(self, *rules: str) -> "NormalizeConfig":
        """A copy with the named rules switched off (for ablations)."""
        unknown = set(rules) - {name for name, _ in RULES}
        if unknown:
            raise ValueError(f"unknown rule(s): {sorted(unknown)}")
        return replace(self, **{rule: False for rule in rules})


# Order matters: fold Unicode forms first, drop marks before unifying letters,
# strip punctuation last.
RULES = [
    ("nfkc", nfkc),
    ("lowercase", lowercase),
    ("strip_tatweel", strip_tatweel),
    ("strip_diacritics", strip_diacritics),
    ("unify_alef", unify_alef),
    ("unify_ya", unify_ya),
    ("unify_ta_marbuta", unify_ta_marbuta),
    ("map_digits", map_digits),
    ("strip_punctuation", strip_punctuation),
]

DEFAULT_CONFIG = NormalizeConfig()


def normalize_steps(
    text: str, config: NormalizeConfig = DEFAULT_CONFIG
) -> list[tuple[str, str]]:
    """Run the enabled rules in order and return [(rule_name, text_after)] for
    each rule that actually changed the text. Used by the demo to show which
    rule fixed what."""
    steps = []
    for name, rule in RULES:
        if getattr(config, name):
            new_text = rule(text)
            if new_text != text:
                steps.append((name, new_text))
                text = new_text
    return steps


def normalize(text: str, config: NormalizeConfig = DEFAULT_CONFIG) -> str:
    """Normalize `text` with the enabled rules, then collapse whitespace."""
    for name, rule in RULES:
        if getattr(config, name):
            text = rule(text)
    return collapse_whitespace(text)
