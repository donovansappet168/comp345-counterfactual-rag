# comp345-counterfactual-rag

COMP 345 term project by Donovan Sappet and Zane Braunstein. We measure how much counterfactual (wrong) retrieved context it takes to flip Llama 3.1 8B Instruct off a fact, and we test whether linear probes on its hidden states can predict the flip. The approved proposal is the spec, and every change from the proposal is listed under "Design decisions" below.

## Design

The corpus is the Baker Institute's `/research/` pages plus the PDF reports those pages link to, split into passages of at least 120 words. The retriever (all-MiniLM-L6-v2) orders each fact's 5 passages by similarity to the question, and the generator (Llama 3.1 8B Instruct) answers from them. Neither model is trained.

Each fact is a short answer (a year, name, or number) stated in 5 passages that are not copies of each other, with one counterfactual value. For instance, NAFTA took effect in 1994, and the edited passages say 1996. We test Llama under three conditions, which differ only in the passages placed in the prompt:

* Condition 1 (closed-book) asks the question with no passages and sorts the facts into parametric (Llama knows the fact) and extractive (Llama does not).
* Condition 2 (clean RAG, m = 0) provides the 5 unedited passages.
* Condition 3 (dose, m = 1 to 5) edits m of the 5 passages.

For each prompt we save one greedy answer (which sets the label), 10 sampled answers (which give the true-answer rate), and the last prompt token's hidden state at all 32 layers (which the probes use). A regex script labels each answer as true, cf (counterfactual), both, or neither. Answers labelled both or neither are checked by hand, and no LLM judge is used.

`examples/nafta_start.md` shows every step for one fact.

## Setup

```bash
python -m venv .venv --system-site-packages
.venv/Scripts/python -m pip install -r requirements.txt
```

Run everything from the repo root with `PYTHONPATH=src` set, for example `PYTHONPATH=src .venv/Scripts/python -m crag.passages`.

### Getting the data

`data/raw/` (the corpus and the passage index, about 200 MB) is too large for git. There are two ways to get the folder:

1. Ask Donovan for `crag_corpus.zip` (about 110 MB, shared on Google Drive) and unzip it into `data/raw/`. The zip holds `baker_pages.jsonl`, `baker_pdfs.jsonl`, `passages.jsonl`, and `passage_embeddings.npy`. It leaves out only the PDF files themselves (1.1 GB), which are needed only to rerun `pdfs.py`.
2. Rebuild the folder with the first three pipeline commands below. Scraping and the PDF step fetch one page per second, so a rebuild takes several hours.

`data/facts.json` (the facts and their passage text) is in git, so the experiment itself runs without the zip. The zip is needed only to search the corpus or add facts.

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
      |                                   results/<run>/baseline.jsonl, hidden/*.npy
      |                                                   |
      |                       analyze.py, figures.py, probes.py --> results/<run>/analysis/
```

| File | What it does | Change it when |
|---|---|---|
| `src/crag/config.py` | Every setting: paths, model names, k, number of samples, seed | You want a different model, k, passage length, ... |
| `src/crag/schema.py` | Typed record shapes: `Document`, `Passage`, `Fact`, `Answer`, `Row` | A record gains a new field |
| `src/crag/scrape.py` | Downloads `/research/` pages from the sitemap and records their PDF links | The site layout changes |
| `src/crag/pdfs.py` | Downloads linked PDFs, strips headers, page numbers, and references, and saves the text | PDF text comes out messy |
| `src/crag/passages.py` | Splits pages into passages, builds the MiniLM embeddings, and provides `search`, `rank`, and `grep` | Changing how passages are cut or ordered |
| `src/crag/mine_facts.py` | Ranks candidate facts (subject and value pairs) by how many non-identical passages state them | Tuning what counts as a candidate |
| `src/crag/find_fact.py` | Lists passages matching a regex, grouped by document, with copies marked, to help pick a fact's passages | |
| `src/crag/facts.py` | Loads `data/facts.json`; `fill` copies in passage text and checks the edits | Adding fields to a fact |
| `src/crag/edit.py` | Swaps the true value for the counterfactual and picks the edited slots | Changing the edit rule or slot choice |
| `src/crag/prompt.py` | The fixed prompt template | Only if we change the proposal |
| `src/crag/generate.py` | Llama wrapper: `greedy`, `sample`, `hidden_states`, `answer` | Decoding settings, saving more layers or tokens |
| `src/crag/score.py` | Labels an answer true, cf, both, or neither, and gives its failure type | Scoring rules |
| `src/crag/run_baseline.py` | Conditions 1, 2, and 3. `rag(llm, fact, wording, m, ...)` works for any m, and `--conditions` picks which conditions to run | Adding a condition |
| `src/crag/worked_example.py` | Writes `examples/<fact>.md` for one fact | Changing what we show the TA |
| `src/crag/analyze.py` | Tables for one run: groups, dose curves, flip points, sigmoid fits, passage-position effects, and the hand-labelling files | Adding a metric |
| `src/crag/figures.py` | Report figures for one run, using the functions in `analyze.py` | Changing a chart |
| `src/crag/probes.py` | Probe pilot: one logistic regression per layer on the saved hidden states, split by fact, with the m-only baseline and the m = 0 control | Probe settings (in `config.py`) |
| `notebooks/demo.ipynb` | Midterm demo: corpus, retrieval, one example, baseline tables | |
| `report/build.py` | Turns the midterm report into `report/midterm.docx` and `report/latex/main.tex`. The report is written in Donovan's notes, and `report/latex/report.md` is the copy the script reads | The report changes |
| `report/make_reference.py` | Writes `report/reference.docx`, the Word style for the report (Times New Roman, formal layout) | Changing the report's look |

## Pipeline commands

| Step | Command | Output |
|---|---|---|
| Scrape (about 1 hour) | `python -m crag.scrape` | `data/raw/baker_pages.jsonl` |
| PDF reports | `python -m crag.pdfs` | `data/raw/pdfs/`, `data/raw/baker_pdfs.jsonl` |
| Passages and index | `python -m crag.passages` | `data/raw/passages.jsonl`, `passage_embeddings.npy` |
| Rank fact candidates (about 2 min) | `python -m crag.mine_facts` (`--min-passages 3` for a longer list) | `data/fact_candidates.csv` |
| Find a fact's passages | `python -m crag.find_fact "<regex>"` | Printed list of passage ids |
| Add facts | Write them into `data/facts.json`, then run `python -m crag.facts fill` | `data/facts.json` |
| Draft facts (not used by any code) | `data/fact_drafts.json` lists every fact Claude drafted, with `"review"` set to `approved` (now in `facts.json`) or `dropped` (with the reason). `data/fact_review.md` shows the edited passages of each approved fact | `data/facts.json` |
| Worked example | `python -m crag.worked_example <fact id> --run` | `examples/<fact id>.md` |
| Conditions 1 and 2 | `python -m crag.run_baseline` | `results/baseline.jsonl`, `results/hidden/` |
| Condition 1 only (fast) | `python -m crag.run_baseline --conditions 1` | `results/baseline.jsonl` |
| Condition 3 (m = 1 to 5) | `python -m crag.run_baseline --conditions 3` | `results/baseline.jsonl`, `results/hidden/` |
| Analyze a run | `python -m crag.analyze <run folder>`, then `python -m crag.figures <run folder>` | `results/<run>/analysis/` |
| Probes (about 6 min on CPU; needs the run's `hidden/` folder) | `python -m crag.probes <run folder>` | `results/<run>/analysis/probes.md`, `probe_layers.csv`, `fig_probe_layers.png` |
| Midterm report | `python report/build.py`, then `latexmk -pdf main.tex` in `report/latex/` | `report/midterm.docx`, `report/latex/main.pdf` |

## Design decisions (changes from or additions to the proposal)

We made each of these decisions after reading the code or the results, and each applies to every condition.

| Date | Decision | Why | Where in the code |
|---|---|---|---|
| 2026-10-25 | Answers may be up to 64 new tokens (was 32) | With edited passages, Llama often names both values ("... 1994. However, Passage 3 states 1996"), and a 32-token cap cut off the second value and hid mixed answers | `MAX_NEW_TOKENS`, `config.py` |
| 2026-10-26 | The chat template's hidden system message says `Today Date: 26 Oct 2026` (the default was 26 Jul 2024) | With the default date, Llama called 2026 events "hypothetical" while reading 2025 and 2026 passages | `CHAT_DATE`, `config.py` |
| 2026-10-26 | Edited slots are nested: one random slot order per fact and wording, and each m edits the first m slots of that order | Each step of m adds exactly one wrong passage, so the change at each step measures one passage's effect. Position still varies across facts and wordings | `edited_slots`, `edit.py` |
| 2026-10-26 | Each prompt gets three sampled-answer rates: true, counterfactual, and both | Many answers name both values. The true-answer rate keeps the proposal's definition, and the other two rates show where the remaining answers go | `make_row`, `run_baseline.py` |
| 2026-10-28 | Answers may be up to 192 new tokens (was 64) | In a test of 50 answers with a 512-token cap, a 64-token cap gave 3 wrong labels (the second value came later, so "both" was scored "cf"). From 128 tokens on, every label matched the uncut answer, and by 192 tokens, 96% of answers had ended on their own. One sample ran to 512 tokens of nonsense, so a cap is still needed | `MAX_NEW_TOKENS`, `config.py` |
| 2026-10-28 | A fact's 5 passages must state the fact in 5 non-identical sentences and may come from the same document (was 5 different documents) | This matches the proposal's "at least 5 separate passages". Fewer Baker documents cover recent events, which the extractive group needs, so the old rule left too few extractive facts. A web page often copies sentences from its own PDF, so a copy still counts once: two sentences are copies if 90% of the shorter one's words appear in the other, in order. This rule found the PDF copy for 150 of 178 sampled page sentences, and the rest were page-only text | `near_copy`, `passages.py`; `COPY_SIMILARITY`, `config.py` |

## Type checking

Every function has type hints and a docstring. To check the whole package, run

```bash
.venv/Scripts/python -m mypy
```

## Compute

Llama runs on a CUDA GPU if one is available and on the CPU otherwise. On Donovan's laptop (no GPU, bf16 on the CPU), with a 1,020-token RAG prompt and a 64-token cap, each step took:

| Step | Time |
|---|---|
| Load the model | 44 s |
| Greedy answer | 55 s |
| Hidden states | 26 s |
| 10 sampled answers | 284 s |
| Closed-book prompt (short) | about 3 s |

With the 192-token cap, sampling takes longer, because sampling runs until the longest of the 10 answers ends. One RAG example then takes about 13 minutes on the CPU. The CPU works for testing a few facts, but the full 1,800 examples need a GPU. The midterm run used a Google Colab L4 GPU. Llama is too slow on the CPU to run live in a demo, so the demo notebook shows saved outputs.

## Results

Each folder in `results/` holds one run. Every number in the midterm report comes from the Colab GPU run, `results/colab-l4-run1/`.

| Folder | What it holds | Use it for |
|---|---|---|
| `colab-l4-run1/` | Conditions 1 to 3 for the 20-fact midterm sample on a Colab L4 GPU (420 prompts, 2026-10-29). `run_info.json` records the commit and GPU, and `analysis/` holds the tables (`summary.md`), the figures, the probe pilot, and the hand-labelling files | Every midterm number |
| `screen_c1_cpu_2026-10-28/` | Closed-book greedy answers for the first 59 facts, run on the laptop CPU | How the midterm sample of facts was picked |
| `archive_2026-10-25_default-date/` | The first baseline and worked example on 8 facts, run with the default chat date and with slots drawn separately for each m | Evidence for the chat-date and nested-slot decisions. Do not quote its numbers |
| `archive_2026-10-26_partial-64-tokens/` | A rerun with the fixed chat date, nested slots, and a 64-token cap, stopped after 17 of 48 rows | Reference only. Replaced by the 192-token runs. Do not quote its numbers |

Hidden-state arrays (the `hidden/` folders) are not in git, because they are large and can be regenerated.
