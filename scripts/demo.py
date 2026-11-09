"""Watch one query travel through the pipeline, stage by stage.

    python scripts/demo.py                     # a default example
    python scripts/demo.py "الشهادة زورا"      # your own query
    python scripts/demo.py --top 3 "ابداء الراي"

The query is run twice: with the default normalization, then with it switched
off, so you can see the silent failure the project is about.

Needs the package installed (pip install -e .) or PYTHONPATH=src.
"""

from __future__ import annotations

import argparse
import sys

from arabic_rag.data import load_corpus, load_queries
from arabic_rag.evaluation import RAW
from arabic_rag.normalize import DEFAULT_CONFIG, normalize_steps
from arabic_rag.pipeline import HybridRetriever, RetrieverConfig, SearchTrace
from arabic_rag.ranking import Hit

# A diacritics query: the matching passage is vowelled, the query is not.
DEFAULT_QUERY = "الشهادة زورا"
PREVIEW = 60  # characters of each passage to show


def heading(text: str) -> None:
    print(f"\n{text}\n{'-' * len(text)}")


def describe(hit: Hit, texts: dict[str, str], relevant: frozenset[str]) -> str:
    mark = "*" if hit.doc_id in relevant else " "
    text = texts[hit.doc_id]
    if len(text) > PREVIEW:
        text = text[:PREVIEW] + "..."
    return f"{mark} {hit.doc_id}  {hit.score:7.3f}  {text}"


def show_ranking(title: str, hits: list[Hit], top: int, texts, relevant) -> None:
    heading(f"{title}  ({len(hits)} passages matched at all)")
    if not hits:
        print("  (nothing matched, and no error was raised)")
    for hit in hits[:top]:
        print("  " + describe(hit, texts, relevant))


def show_fused(trace: SearchTrace, top: int, texts, relevant) -> None:
    heading("Fused ranking (reciprocal-rank fusion)")
    if not trace.fused:
        print("  (nothing to fuse)")
    for hit in trace.fused[:top]:
        contributions = trace.contributions[hit.doc_id]  # ranker -> (rank, rrf part)
        sum_ = " + ".join(f"{part:.4f}" for _, part in contributions.values())
        where = "; ".join(f"{name} rank {rank}" for name, (rank, _) in contributions.items())
        print("  " + describe(hit, texts, relevant))
        print(f"       = {sum_}   ({where})")


def run_stage(retriever: HybridRetriever, query: str, top: int, texts, relevant) -> SearchTrace:
    trace = retriever.trace(query, top_k=top)
    heading("Normalized query")
    print(f"  {trace.normalized_query!r}")
    heading("Tokens (what BM25 sees)")
    print(f"  {trace.tokens}")
    for name, label in (("bm25", "BM25 ranking"), ("ngram", "Character n-gram ranking")):
        if name in trace.rankings:
            show_ranking(label, trace.rankings[name], top, texts, relevant)
    show_fused(trace, top, texts, relevant)
    return trace


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("query", nargs="?", default=DEFAULT_QUERY, help="the query to trace")
    parser.add_argument("--top", type=int, default=5, help="results shown per ranking")
    args = parser.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")  # Windows consoles default to cp1252

    docs = load_corpus()
    texts = {d.id: d.text for d in docs}
    # If this is one of the labelled queries, mark its right answer with a "*".
    relevant = next(
        (q.relevant_ids for q in load_queries() if q.text == args.query), frozenset()
    )

    print(f"Query: {args.query!r}   ({len(docs)} passages in the corpus)")
    if relevant:
        print(f"Expected passage: {', '.join(sorted(relevant))}   (marked * below)")
    else:
        print("Not a labelled query, so no expected passage is marked.")

    # 1. Normalization, rule by rule ------------------------------------------
    print("\n=== WITH NORMALIZATION (the default) ===")
    heading("Raw query")
    print(f"  {args.query!r}")
    heading("Normalization: only the rules that changed something")
    steps = normalize_steps(args.query, DEFAULT_CONFIG)
    if not steps:
        print("  (no rule changed this query)")
    for name, text in steps:
        print(f"  {name:<18} -> {text!r}")

    # 2. The full pipeline ------------------------------------------------------
    normalized = run_stage(HybridRetriever(docs), args.query, args.top, texts, relevant)

    # 3. The same query with normalization off -----------------------------------
    print("\n=== SAME QUERY, NORMALIZATION OFF ===")
    raw = run_stage(
        HybridRetriever(docs, RetrieverConfig(normalize=RAW)), args.query, args.top, texts, relevant
    )

    # 4. Verdict ------------------------------------------------------------------
    heading("What changed")
    for label, trace in (("normalized", normalized), ("raw", raw)):
        ids = [h.doc_id for h in trace.fused]
        bm25 = len(trace.rankings.get("bm25", []))
        if relevant:
            rank = next((i for i, d in enumerate(ids, 1) if d in relevant), None)
            where = (
                f"expected passage at rank {rank}"
                if rank
                else f"expected passage NOT in the top {args.top}"
            )
        else:
            where = f"top result {ids[0]}" if ids else "no results"
        print(f"  {label:<10} BM25 matched {bm25:>3} passages; fused: {where}")


if __name__ == "__main__":
    main()
