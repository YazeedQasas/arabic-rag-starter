# Data

Hand-written and tiny **by design**, so you can read all of it in a few minutes.
It is not a benchmark and results on it should not be generalized.

- `corpus.jsonl`: 34 short Arabic passages (`id`, `text`). About two thirds are
  legal/judiciary-flavoured, the rest are general topics that act as distractors.
  Some passages are written the way real documents are: `d03` and `d05` are fully
  vowelled (tashkeel), `d16` uses tatweel (kashida), `d29` and `d30` use
  Arabic-Indic digits.
- `queries.jsonl`: labelled queries (`query`, `relevant_ids`, `kind`). `kind` says
  what each query is meant to test:

  | kind | what it tests |
  |---|---|
  | `exact`, `exact_rare` | plain exact-word queries (both rankers should do well) |
  | `diacritics`, `alef`, `ta_marbuta`, `ya`, `tatweel`, `digits` (and combinations) | query and document spell the same word differently; only normalization can bridge it |
  | `morphology` | no whole word in common (prefixes/suffixes differ); only character n-grams can bridge it |
  | `typo` | misspelled query terms |
  | `natural` | question-style queries padded with common words |

The passages and queries were drafted by an AI model (Claude) and should be reviewed
by a native Arabic reader for natural phrasing.
