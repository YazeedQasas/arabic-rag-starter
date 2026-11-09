# arabic-rag-starter

Hybrid retrieval for Arabic text. It combines normalization, BM25, character
n-gram similarity and reciprocal-rank fusion, and it measures results with
recall@k and MRR.

Arabic retrieval usually fails silently in the text pipeline: diacritics
(tashkeel), alef variants and ة/ه each need a normalization rule, and without
them a query simply doesn't match the document. This repo is a small starting
point you can read in an evening, not a production system.

> Work in progress. Full documentation lands in the last phase.
