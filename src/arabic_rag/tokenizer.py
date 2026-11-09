"""Tokenizer: normalized text -> list of word tokens.

Deliberately simple. No stemming or lemmatization (see the README's next steps);
the character n-gram ranker covers prefix/suffix variation instead.

Named `tokenizer.py`, not `tokenize.py`, so it can never shadow the standard
library's `tokenize` module.
"""

from __future__ import annotations

import re

from arabic_rag.normalize import DEFAULT_CONFIG, NormalizeConfig, normalize

# A token is a run of word characters (letters, digits, underscore, so Latin
# words and numbers mixed into Arabic text survive) plus combining marks.
# Marks are included so that diacritics stay attached to their word when
# strip_diacritics is switched off, instead of splitting it apart.
_TOKEN = re.compile(r"[\wً-ٰٟ]+")


def tokenize(text: str, config: NormalizeConfig = DEFAULT_CONFIG) -> list[str]:
    """Normalize `text`, then split it into tokens."""
    return _TOKEN.findall(normalize(text, config))
