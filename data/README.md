# Data

Hand-written and small **by design**, so the whole thing is readable. It is not a
benchmark: the passages and queries were written to expose specific failure modes,
so the numbers below show *which kinds* of failure each ranker has, not how often
they happen on real text.

The passages and queries were drafted by an AI model (Claude) and reviewed by the
repo owner for natural phrasing. An independent native-speaker review would still
be worthwhile before relying on them for anything beyond learning.

## Files

- `corpus.jsonl`: 104 short Arabic passages (`id`, `text`). About two thirds are
  legal/judiciary-flavoured; the rest are general topics that act as distractors.
  Some are written the way real documents are: `d03`, `d05`, `d59`-`d62` are
  vowelled (tashkeel), `d16` and `d72`-`d75` use tatweel (kashida), and `d29`,
  `d30`, `d76`, `d77` use Arabic-Indic digits.
- `queries.jsonl`: 65 labelled queries (`query`, `relevant_ids`, `kind`).

| kind | n | what it tests |
|---|---|---|
| `exact`, `exact_rare` | 9 | plain whole-word queries |
| `natural` | 4 | question-style queries padded with common words |
| `diacritics`, `alef`, `ta_marbuta`, `ya`, `tatweel`, `digits` | 30 | query and document spell the same word differently; only normalization can bridge it (a `+` means several at once) |
| `morphology` | 7 | no whole word in common (prefixes/suffixes differ) |
| `typo` | 5 | misspelled query words |
| `confusable` | 10 | the document has the exact word; a *short* distractor has a near-identical word with a different meaning (الكتاب / الكتابة, المدين / المدينة, الحجز / الحجر ...) |

## What the comparison shows

MRR by query kind (1.00 = the right passage is always first). "raw" = no
normalization, "norm" = default normalization.

| kind | BM25 raw | BM25 norm | n-gram raw | n-gram norm | hybrid |
|---|---|---|---|---|---|
| exact (5) | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| exact_rare (4) | 1.00 | 1.00 | 0.88 | 0.88 | 1.00 |
| natural (4) | 1.00 | 1.00 | 0.88 | 0.88 | 1.00 |
| diacritics (6) | 0.00 | 1.00 | 0.00 | 1.00 | 1.00 |
| alef (6) | 0.00 | 1.00 | 0.92 | 1.00 | 1.00 |
| ta_marbuta (4) | 0.25 | 1.00 | 1.00 | 1.00 | 1.00 |
| ya (5) | 0.00 | 1.00 | 0.80 | 1.00 | 1.00 |
| tatweel (5) | 0.20 | 1.00 | 0.73 | 1.00 | 1.00 |
| digits (4) | 0.62 | 1.00 | 0.62 | 1.00 | 1.00 |
| morphology (7) | 0.00 | 0.00 | 0.86 | 0.86 | 0.82 |
| typo (5) | 0.00 | 0.00 | 0.82 | 0.82 | 0.82 |
| confusable (10) | 1.00 | 1.00 | 0.73 | 0.77 | 1.00 |
| **all (65)** | **0.42** | **0.82** | **0.76** | **0.92** | **0.97** |

1. **Normalization is what makes BM25 work.** Without it BM25 scores about 0 on every
   spelling-variant kind and returns *nothing*, silently. With it, 1.00.
2. **BM25 is blind to morphology and typos** (0.00 even with normalization): it only
   matches whole tokens, and `المتهمون` is not `للمتهم`.
3. **Character n-grams cover exactly that gap** (0.86 / 0.82) but are less *precise*.
   On `confusable` queries a short passage with a near-identical word outranks the
   passage with the exact word (`الكتاب`, `المدين`, `التركة` put the right passage
   2nd; `المحاكمة` puts it 5th), because cosine similarity favours short documents
   that are almost entirely made of matching n-grams. BM25 gets all of these right.
4. **N-grams are partly robust to spelling variants even without normalization**
   (`ta_marbuta` 1.00) because only the last n-gram changes, but not to diacritics
   (0.00): marks inside a word change every n-gram around them.
5. **Hybrid is best overall (0.97), but not best everywhere.** It fixes the two
   rankers' opposite weaknesses, yet on `morphology` it is slightly below n-grams
   alone (0.82 vs 0.86). For `حماية المستهلك`, BM25 confidently matches three unrelated
   passages on the word `حماية` (and misses the right one), and RRF has no way to know
   BM25 is wrong, so it drags the right passage from 2nd to 4th.

The `confusable` set was designed to expose the n-gram weakness, so its 0.77 should
not be read as a general error rate.
