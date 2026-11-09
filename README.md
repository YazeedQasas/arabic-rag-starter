# arabic-rag-starter

Hybrid retrieval for Arabic text: normalization, BM25, character n-gram
similarity and reciprocal-rank fusion, measured with recall@k and MRR. It is a
small starting point you can read in an evening, **not a production system**.

## The lesson

Arabic retrieval usually fails **silently in the text pipeline**. The same word
can be written several valid ways: with or without diacritics (tashkeel), with
different alef forms (أ إ آ ا), ة or ه, ى or ي. A search engine compares
characters, so a query for `الشهادة` does not match a passage that contains
`الشَّهَادَةُ`, and nothing raises an error. You just get no results, or worse,
plausible wrong ones.

Here is the bundled demo query, first with normalization and then without:

```
normalized  BM25 matched   1 passages; fused: expected passage at rank 1
raw         BM25 matched   0 passages; fused: expected passage NOT in the top 5
```

Without normalization, BM25 returned nothing for 37 of the 65 test queries, and
none of them produced an error. This repo shows where each stage of the pipeline
helps and where it does not.

## The pipeline

```
          corpus (indexed once)                    query (every search)
                  |                                        |
        normalize each passage                       normalize the query    <- the SAME function
           |              |                           |              |
        tokenize     keep as text                  tokenize      keep as text
           |              |                           |              |
      BM25 index    n-gram index   ------------>  BM25 search   n-gram search
                                                      |              |
                                                      +--> RRF  <----+
                                                            |
                                                      top-k passages
```

| Stage | Module | What it does |
|---|---|---|
| Normalize | [normalize.py](src/arabic_rag/normalize.py) | Nine small rules (diacritics, alef, ة/ه, ى/ي, tatweel, digits, Unicode forms, case, punctuation). Each is a function and each can be switched off, which is what the ablation uses. |
| Tokenize | [tokenizer.py](src/arabic_rag/tokenizer.py) | Splits normalized text into words. |
| BM25 | [bm25.py](src/arabic_rag/bm25.py) | Exact-word matching, a thin wrapper over `rank_bm25`. |
| Char n-grams | [ngram.py](src/arabic_rag/ngram.py) | Character 3-gram TF-IDF with cosine similarity, via scikit-learn. Matches words that look alike. |
| Fusion | [fusion.py](src/arabic_rag/fusion.py) | Reciprocal-rank fusion: `score = sum of 1 / (60 + rank)` over the rankers. Uses ranks only, so no score calibration is needed. |
| Pipeline | [pipeline.py](src/arabic_rag/pipeline.py) | `HybridRetriever` wires it together, with a `trace()` that keeps every intermediate result. |
| Metrics | [metrics.py](src/arabic_rag/metrics.py), [evaluation.py](src/arabic_rag/evaluation.py) | recall@k, MRR, per-query-kind scores, the configurations compared. |

## Quickstart

Python 3.10 or newer.

```
git clone https://github.com/YazeedQasas/arabic-rag-starter
cd arabic-rag-starter
python -m venv .venv
.venv\Scripts\activate          # macOS/Linux: source .venv/bin/activate
pip install -e ".[dev]"

pytest                          # run the tests
python scripts/evaluate.py      # print the comparison tables and write results.md
python scripts/demo.py          # trace one query through every stage
python scripts/demo.py "حماية المستهلك"
```

`demo.py` prints the raw query, each normalization rule that changed it, the
tokens, the top results from each ranker, the fused ranking with each passage's
RRF contributions, and then the same query again with normalization off.

Search from your own code:

```python
from arabic_rag.data import load_corpus
from arabic_rag.pipeline import HybridRetriever

retriever = HybridRetriever(load_corpus())
for hit in retriever.search("الشهادة زورا", top_k=3):
    print(hit.doc_id, hit.score)
```

## Results

104 hand-written passages and 65 labelled queries, each judged on its top 10
results. Full tables, including the per-rule ablations and the list of silent
failures, are in [results.md](results.md). The data is described in
[data/README.md](data/README.md).

| configuration      | recall@1 | recall@3 | recall@5 |  MRR |
|--------------------|---------:|---------:|---------:|-----:|
| BM25, raw          |     0.42 |     0.43 |     0.43 | 0.42 |
| BM25, normalized   |     0.82 |     0.82 |     0.82 | 0.82 |
| n-gram, raw        |     0.66 |     0.86 |     0.86 | 0.76 |
| n-gram, normalized |     0.86 |     0.97 |     0.98 | 0.92 |
| Hybrid, raw        |     0.74 |     0.82 |     0.85 | 0.78 |
| Hybrid, normalized |     0.95 |     0.97 |     0.98 | 0.97 |

`raw` = no normalization. `Hybrid` = BM25 + n-gram fused with RRF.

What it shows:

- **Normalization is what makes BM25 work** (0.42 to 0.82). Switching off a
  single rule hurts: without `strip_diacritics` or `unify_alef`, BM25 loses 0.09 MRR.
- **BM25 and n-grams fail in opposite ways.** BM25 needs the exact word, so it
  scores 0.00 on morphology and typo queries. N-grams handle those (0.86 and
  0.82) but rank a short passage with a *similar* word above the passage with the
  *exact* word (`الكتاب` vs `الكتابة`), scoring 0.77 on those "confusable" queries.
  BM25 gets all of them right.
- **The hybrid is best overall (0.97) but not best everywhere.** On morphology
  it is slightly below n-grams alone, because RRF sees only ranks and cannot tell
  when BM25 is confidently wrong.
- **The hybrid hides normalization failures.** The n-gram ranker rescues many
  queries that BM25 loses, so a missing rule shows up as a small drop, not an
  obvious break. Test the pieces separately, as `evaluate.py` does.

## Limitations

Read the numbers above as *which kinds of failure each ranker has*, not as
accuracy you would see on real traffic.

- **The data is tiny and synthetic.** The passages and queries were drafted by
  an AI model and reviewed by the repo owner for natural phrasing. Another
  independent native review would still be worthwhile.
- **The queries were written to expose failures**, so the per-kind scores are not
  error rates. The `confusable` set in particular was built to expose the n-gram
  weakness.
- **There is no held-out set.** The corpus was extended after seeing results (the
  first 34 passages were too easy), so the numbers are partly shaped by that tuning.
- **Small samples.** Each query kind has 4 to 10 queries, and every query has one
  relevant passage, so recall@k is a hit rate. A gap of 0.05 is about one query.
- **Three normalization rules are not exercised by the data:** `nfkc`, `lowercase`
  and `strip_punctuation` show 0.00 effect here because the test text does not
  contain PDF presentation forms, mixed-case Latin words or punctuation-sensitive
  queries. That does not mean they are useless on real text.
- **Normalization is lossy on purpose.** For example `على` (on) and `علي` (Ali)
  become the same token. That is the right trade for matching, but use the
  original text for display.
- **No stemming and no dense retrieval.** Both are deliberately out of scope.
- `k = 60` for RRF and the BM25 parameters are library defaults, not tuned.

## Next steps

- A light stemmer or prefix stripper, to give BM25 a chance on morphology queries.
- Embeddings as a third ranker, fused with RRF like the other two.
- A larger real corpus with real user queries, and a held-out split you do not
  tune on.
- Queries with several relevant passages, so recall@k becomes a real fraction.
- Tune `k` in RRF, or try score-aware fusion, which could downweight a weak BM25 match.

## Layout

```
src/arabic_rag/   the pipeline, one small module per stage
scripts/          evaluate.py (tables, ablations), demo.py (trace one query)
data/             corpus.jsonl, queries.jsonl, and a README describing them
tests/            unit tests for every stage, plus the evaluation checkpoint
results.md        generated by scripts/evaluate.py
```
