# comp345-counterfactual-rag

COMP 345 term project (Donovan Sappet and Zane Braunstein): how much counterfactual retrieved
context does it take to flip Llama 3.1 8B Instruct off a fact, and can linear probes on its hidden
states predict the flip? The approved proposal is the spec; changes from it are listed under
"Design decisions" below.

## Design in brief

- **Corpus.** Baker Institute `/research/` pages plus the PDF reports they link to, split into
  passages of about 120 words.
- **Models.** Retriever all-MiniLM-L6-v2 (top k = 5). Generator Llama 3.1 8B Instruct. Both frozen.
- **Facts.** Short answers (years, names, numbers), each stated in 5 passages from 5 different
  documents, with one counterfactual value (e.g. NAFTA took effect in 1994, edited to 1996).
- **Conditions.** 1 closed-book (sorts facts into parametric vs extractive). 2 clean RAG (m = 0).
  3 dose: m = 1 to 5 of the 5 passages edited.
- **Per prompt.** One greedy answer (the label), 10 sampled answers (true-answer rate), and the last
  prompt token's hidden state at all 32 layers (for the probes).
- **Scoring.** Regex script: true / cf / both / neither. Both and neither are checked by hand. No LLM judge.

`examples/nafta_start.md` walks through all of this for one fact.

## Setup

```bash
python -m venv .venv --system-site-packages
.venv/Scripts/python -m pip install -r requirements.txt
```

Run everything from the repo root with `PYTHONPATH=src` set, for example
`PYTHONPATH=src .venv/Scripts/python -m crag.passages`.

### Getting the data

`data/raw/` (the corpus and the passage index, about 200 MB) is not in git. Two ways to get it:

1. **Ask Donovan for `crag_corpus.zip`** (about 110 MB, shared on Drive since it is too big for
   GitHub) and unzip it into `data/raw/`. It has
   `baker_pages.jsonl`, `baker_pdfs.jsonl`, `passages.jsonl`, and `passage_embeddings.npy`, which is
   everything except the PDF files themselves (1.1 GB, only needed to rerun `pdfs.py`).
2. **Rebuild it** with the first three pipeline commands below. Scraping and the PDF step fetch one
   page per second, so this takes several hours.

`data/facts.json` (the facts and their passage text) is in git, so running the experiment only
needs the zip if you want to search the corpus or add facts.

## How the code fits together

```
 Baker website                                   data/facts.json (hand-checked)
      |                                                   |
  scrape.py --> baker_pages.jsonl                         |  fact, its 5+ passages,
  pdfs.py   --> baker_pdfs.jsonl (linked PDF reports)     |  true / counterfactual value
      |                                                   |
  passages.py --> passages.jsonl + embeddings             |
      |   (grep() and search() help us find facts) -------+
      |                                                   |
      |                       run_baseline.py / worked_example.py
      |                            for each fact, wording, m:
      |                              rank()          passages.py   order the 5 passages
      |                              build_context() edit.py       edit m of them
      |                              rag_prompt()    prompt.py     fill the template
      |                              llm.answer()    generate.py   greedy + samples + hidden states
      |                              label()         score.py      true / cf / both / neither
      |                                                   |
      |                                   results/baseline.jsonl, results/hidden/*.npy
```

| File | What it does | Change it when |
|---|---|---|
| `src/crag/config.py` | Every setting: paths, model names, k, number of samples, seed | You want a different model, k, passage length, ... |
| `src/crag/schema.py` | Typed record shapes: `Document`, `Passage`, `Fact`, `Answer`, `Row` | A record gains a new field |
| `src/crag/scrape.py` | Downloads `/research/` pages from the sitemap, and records their PDF links | The site layout changes |
| `src/crag/pdfs.py` | Downloads linked PDFs, strips headers/page numbers/references, saves the text | PDF text comes out messy |
| `src/crag/passages.py` | Splits pages into passages, MiniLM embeddings, `search`, `rank`, `grep` | Changing how passages are cut or retrieved |
| `src/crag/mine_facts.py` | Ranks candidate facts (subject + value pairs) by how many documents state them | Tuning what counts as a candidate |
| `src/crag/find_fact.py` | Lists passages matching a regex, grouped by document, to help pick facts | |
| `src/crag/facts.py` | Loads `data/facts.json`; `fill` copies in passage text and checks the edits | Adding fields to a fact |
| `src/crag/edit.py` | Swaps the true value for the counterfactual; picks edited slots | Changing the edit rule or slot choice |
| `src/crag/prompt.py` | The fixed prompt template | Only if we change the proposal |
| `src/crag/generate.py` | Llama wrapper: `greedy`, `sample`, `hidden_states`, `answer` | Decoding settings, saving more layers or tokens |
| `src/crag/score.py` | Labels an answer true / cf / both / neither; failure type | Scoring rules |
| `src/crag/run_baseline.py` | Conditions 1 and 2. `rag(llm, fact, wording, m, ...)` already works for any m | Writing Condition 3 (loop m = 1..5) |
| `src/crag/worked_example.py` | Writes `examples/<fact>.md` for one fact | Changing what we show the TA |
| `notebooks/demo.ipynb` | Midterm demo: corpus, retrieval, one example, baseline tables | |

## Pipeline commands

| Step | Command | Output |
|---|---|---|
| Scrape (about 1 hour) | `python -m crag.scrape` | `data/raw/baker_pages.jsonl` |
| PDF reports | `python -m crag.pdfs` | `data/raw/pdfs/`, `data/raw/baker_pdfs.jsonl` |
| Passages + index | `python -m crag.passages` | `data/raw/passages.jsonl`, `passage_embeddings.npy` |
| Rank fact candidates (about 2 min) | `python -m crag.mine_facts` (`--min-documents 3` for a longer list) | `data/fact_candidates.csv` |
| Find a fact's passages | `python -m crag.find_fact "<regex>"` | printed list of passage ids |
| Add facts | write them into `data/facts.json`, then `python -m crag.facts fill` | `data/facts.json` |
| Worked example | `python -m crag.worked_example <fact id> --run` | `examples/<fact id>.md` |
| Conditions 1 and 2 | `python -m crag.run_baseline` | `results/baseline.jsonl`, `results/hidden/` |

## Design decisions (changes from or additions to the proposal)

Each one is found while reading the code or results, and is the same in every condition.

| Date | Decision | Why | Where in the code |
|---|---|---|---|
| 2026-10-25 | Answers may be up to 64 new tokens (was 32) | With edited passages Llama often names both values ("... 1994. However, Passage 3 states 1996"); 32 tokens cut that off and hid mixed answers | `MAX_NEW_TOKENS`, `config.py` |
| 2026-10-28 | Answers may be up to 192 new tokens (was 64) | Test on 50 answers with a 512-token cap: at 64, 3 labels were wrong (the second value came later, so "both" was scored "cf"); from 128 on every label matched the uncut answer, and by 192, 96% of answers had ended on their own. One sample ran on to 512 tokens of nonsense, so a cap is still needed | `MAX_NEW_TOKENS`, `config.py` |
| 2026-10-26 | The chat template's hidden system message says `Today Date: 26 Oct 2026` (default was 26 Jul 2024) | With the default date, Llama called 2026 events "hypothetical" while reading 2025-2026 passages | `CHAT_DATE`, `config.py` |
| 2026-10-26 | Edited slots are nested: one random slot order per fact and wording, edit its first m | Each step of m adds exactly one wrong passage, so per-fact curves are a clean dose-response; position still varies across facts and wordings | `edited_slots`, `edit.py` |
| 2026-10-26 | Report three sample rates per prompt: true, counterfactual, and both (mixed) | Many answers name both values; the true-answer rate keeps the proposal's definition and the other two show where the rest go | `make_row`, `run_baseline.py` |

**Open, to ask the TA:** should a fact need 5 different documents (our current rule), or 5 passages
that are not copies of each other (closer to the proposal's "at least 5 separate passages")? The
looser rule makes about 114 more `mine_facts.py` candidates eligible.

## Type checking

Every function has type hints and a docstring. Check the whole package with

```bash
.venv/Scripts/python -m mypy
```

## Compute

Llama runs on a CUDA GPU if one is visible and on the CPU otherwise. On Donovan's laptop (no GPU,
bf16 on the CPU), measured with a 1,020-token RAG prompt:

| Step | Time |
|---|---|
| Load the model | 44 s |
| Greedy answer | 55 s |
| Hidden states | 26 s |
| 10 sampled answers | 284 s |
| Closed-book prompt (short) | about 3 s |

These were measured with a 64-token cap. At 192 tokens the sampled answers take longer, since
sampling runs until the longest of the 10 answers ends: about 13 minutes per RAG example on a CPU.
That is fine for a few facts, too slow for the full
1,800 examples, which go on the Rice cluster GPU. Don't run Llama live in a demo; show saved outputs.

## Results

`results/baseline.jsonl` is being rerun (2026-10-26) with the pinned chat date and nested slots.
`results/archive_2026-10-25_default-date/` is the earlier run, kept only to compare against. Don't
quote its numbers. `results/hidden/` (hidden-state arrays) is not in git.
