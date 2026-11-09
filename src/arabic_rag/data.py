"""Load the bundled corpus and labelled queries (JSON Lines, UTF-8)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

# src/arabic_rag/data.py -> repository root -> data/
DATA_DIR = Path(__file__).resolve().parents[2] / "data"


@dataclass(frozen=True)
class Document:
    id: str
    text: str


@dataclass(frozen=True)
class Query:
    text: str
    relevant_ids: frozenset[str]
    kind: str  # what this query tests, e.g. "alef", "diacritics", "morphology"


def _read_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_corpus(path: Path = DATA_DIR / "corpus.jsonl") -> list[Document]:
    return [Document(row["id"], row["text"]) for row in _read_jsonl(path)]


def load_queries(path: Path = DATA_DIR / "queries.jsonl") -> list[Query]:
    return [
        Query(row["query"], frozenset(row["relevant_ids"]), row.get("kind", ""))
        for row in _read_jsonl(path)
    ]
