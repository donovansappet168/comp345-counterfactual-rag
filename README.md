# Counterfactual RAG: How Much Wrong Context Flips an LLM's Answer?

COMP 345 term project by Donovan Sappet and Zane Braunstein.

We measure how much counterfactual (wrong) retrieved context it takes to flip Llama 3.1 8B Instruct off a fact, and we test whether linear probes on its hidden states can predict the flip. This repository holds the full pipeline: collecting the corpus, building the facts, running Llama, scoring its answers, and training the probes. We built the pipeline to follow our approved project proposal, and every change we made to the proposal's design is listed under "Design decisions" below. The midterm report is submitted on Canvas.

## Where to start

The saved outputs show the whole project without any setup or GPU. These four files are a good place to start:

1. `examples/nafta_start.md` follows one fact (the year NAFTA went into effect) through every step: the question, the 5 passages, the edits, and Llama's answers at each m.
2. `notebooks/demo.ipynb` is the notebook from our midterm demo session on Wednesday, October 28. It covers the corpus, retrieval, the facts, one fact end to end, which facts Llama knows closed-book, and NAFTA at m = 0 to 5. The notebook shows saved outputs, so it opens without a GPU.
3. `results/colab-l4-run1/analysis/summary.md` holds the tables behind every number in the midterm report.
4. `src/crag/` holds the pipeline code, with one short module for each step (see "How the code fits together").

## Design

The corpus is the Baker Institute's `/research/` pages plus the PDF reports those pages link to. We split these documents into passages of at least 120 words. The retriever (all-MiniLM-L6-v2) orders each fact's 5 passages by similarity to the question, and the generator (Llama 3.1 8B Instruct) answers from them. We use both models as they are and do not train either one.

Each fact is a short answer (a year, name, or number) stated in 5 passages that are not copies of each other, with one counterfactual value. For instance, NAFTA took effect in 1994, and the edited passages say 1996. We test Llama under three conditions, which differ only in the passages placed in the prompt:

* Condition 1 (closed-book) asks the question with no passages and sorts the facts into parametric (Llama knows the fact) and extractive (Llama does not).
* Condition 2 (clean RAG, m = 0) provides the 5 unedited passages.
* Condition 3 (dose, m = 1 to 5) edits m of the 5 passages.

For each prompt we save one greedy answer (which sets the label), 10 sampled answers (which give the true-answer rate), and the last prompt token's hidden state at all 32 layers (which the probes use). A regex script labels each answer as true, cf (counterfactual), "both", or "neither", and no LLM judge is used. The midterm reports "both" as its own label. For the final report, we will read the "both" and "neither" answers and label them by hand.

## Setup

```bash
python -m venv .venv --system-site-packages
.venv/Scripts/python -m pip install -r requirements.txt
```

`--system-site-packages` lets the environment reuse a PyTorch that is already installed, which saves a large download. On macOS and Linux, use `.venv/bin/python` wherever this README says `.venv/Scripts/python`.

Run everything from the repo root with `PYTHONPATH=src` set, for instance `PYTHONPATH=src .venv/Scripts/python -m crag.passages`. The models download from Hugging Face on first use, and neither needs an access request (see `GENERATOR` in `config.py`).

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
| `src/crag/score.py` | Labels an answer true, cf, "both", or "neither", and gives its failure type | Scoring rules |
| `src/crag/run_baseline.py` | Conditions 1, 2, and 3. `rag(llm, fact, wording, m, ...)` works for any m, and `--conditions` picks which conditions to run | Adding a condition |
| `src/crag/worked_example.py` | Writes `examples/<fact>.md` for one fact | Changing the worked example shown in the demo session |
| `src/crag/analyze.py` | Tables for one run: groups, dose curves, flip points, sigmoid fits, passage-position effects, and the hand-labelling files | Adding a metric |
| `src/crag/figures.py` | Report figures for one run, using the functions in `analyze.py` | Changing a chart |
| `src/crag/probes.py` | Probe pilot: one logistic regression per layer on the saved hidden states, split by fact, with the m-only baseline and the m = 0 control | Probe settings (in `config.py`) |
| `notebooks/demo.ipynb` | Midterm demo with saved outputs (see "Where to start") | |
| `notebooks/colab_run.ipynb` | Runs Conditions 1 to 3 on a Colab GPU and saves the run folder to Google Drive | Starting a new GPU run |

## Data files

| File | What it holds |
|---|---|
| `data/facts.json` | The 62 hand-checked facts: true and counterfactual values, 3 question wordings, and the text and source URL of each passage. Every run reads this file |
| `data/fact_candidates.csv` | The ranked candidate facts from `mine_facts.py`, which we used as a starting point |
| `data/fact_drafts.json` | The 98 facts that Claude drafted from the candidates, each marked `approved` (54, now in `facts.json`) or `dropped` (44, with the reason). No code reads this file |
| `data/fact_review.md` | The review sheet for those drafts: each approved fact with its 5 passages, then each dropped draft with the reason. We read every approved fact's passages before adding the fact to `facts.json` |

## Pipeline commands

| Step | Command | Output |
|---|---|---|
| Scrape (about 1 hour) | `python -m crag.scrape` | `data/raw/baker_pages.jsonl` |
| PDF reports | `python -m crag.pdfs` | `data/raw/pdfs/`, `data/raw/baker_pdfs.jsonl` |
| Passages and index | `python -m crag.passages` | `data/raw/passages.jsonl`, `passage_embeddings.npy` |
| Rank fact candidates (about 2 min) | `python -m crag.mine_facts` (`--min-passages 3` for a longer list) | `data/fact_candidates.csv` |
| Find a fact's passages | `python -m crag.find_fact "<regex>"` | Printed list of passage ids |
| Add facts | Write them into `data/facts.json`, then run `python -m crag.facts fill` | `data/facts.json` |
| Worked example | `python -m crag.worked_example <fact id> --run` | `examples/<fact id>.md` |
| Conditions 1 and 2 | `python -m crag.run_baseline` | `results/baseline.jsonl`, `results/hidden/` |
| Condition 1 only (fast) | `python -m crag.run_baseline --conditions 1` | `results/baseline.jsonl` |
| Condition 3 (m = 1 to 5) | `python -m crag.run_baseline --conditions 3` | `results/baseline.jsonl`, `results/hidden/` |
| Analyze a run | `python -m crag.analyze <run folder>`, then `python -m crag.figures <run folder>` | `results/<run>/analysis/` |
| Probes (about 6 min on CPU; needs the run's `hidden/` folder) | `python -m crag.probes <run folder>` | `results/<run>/analysis/probes.md`, `probe_layers.csv`, `fig_probe_layers.png` |

A local run writes to `results/baseline.jsonl` and `results/hidden/`. To keep a run, move both into a folder of its own, such as `results/colab-l4-run1/`, which is the folder the analysis commands take. The Colab notebook saves its run folder this way.

## Design decisions (changes from or additions to the proposal)

We made each of these decisions after reading the code or the results, and each applies to every condition.

| Decision | Why | Where in the code |
|---|---|---|
| Answers may be up to 64 new tokens (was 32) | With edited passages, Llama often names both values ("... 1994. However, Passage 3 states 1996"), and a 32-token cap cut off the second value and hid mixed answers | `MAX_NEW_TOKENS`, `config.py` |
| The chat template's hidden system message says `Today Date: 26 Oct 2026` (the default was 26 Jul 2024) | With the default date, Llama called 2026 events "hypothetical" while reading 2025 and 2026 passages | `CHAT_DATE`, `config.py` |
| Edited slots are nested: one random slot order per fact and wording, and each m edits the first m slots of that order | Each step of m adds exactly one wrong passage, so the change at each step measures one passage's effect. Position still varies across facts and wordings | `edited_slots`, `edit.py` |
| Each prompt gets three sampled-answer rates: true, counterfactual, and both | Many answers name both values. The true-answer rate keeps the proposal's definition, and the other two rates show where the remaining answers go | `make_row`, `run_baseline.py` |
| Answers may be up to 192 new tokens (was 64) | In a test of 50 answers with a 512-token cap, a 64-token cap gave 3 wrong labels (the second value came later, so "both" was scored "cf"). From 128 tokens on, every label matched the uncut answer, and by 192 tokens, 96% of answers had ended on their own. One sample ran to 512 tokens of nonsense, so a cap is still needed | `MAX_NEW_TOKENS`, `config.py` |
| A fact's 5 passages must state the fact in 5 non-identical sentences and may come from the same document (was 5 different documents) | This matches the proposal's "at least 5 separate passages". Fewer Baker documents cover recent events, which the extractive group needs, so the old rule left too few extractive facts. A web page often copies sentences from its own PDF, so a copy still counts once: two sentences are copies if 90% of the shorter one's words appear in the other, in order. This rule found the PDF copy for 150 of 178 sampled page sentences, and the rest were page-only text | `near_copy`, `passages.py`; `COPY_SIMILARITY`, `config.py` |

## Type checking

Every function has type hints and a docstring, so a reader can see what each function takes and returns without reading its body. mypy checks the hints against how each function is called, which catches mistakes such as passing a passage where a fact is expected. To check the whole package, run

```bash
.venv/Scripts/python -m mypy
```

## Compute

Llama runs on a CUDA GPU if one is available and on the CPU otherwise. Each prompt produces one greedy answer, 10 sampled answers, and the hidden states. The midterm run used a Google Colab L4 GPU and took about 2 hours for 420 prompts:

| Prompt type | Prompts | Time per prompt (L4 GPU) |
|---|---|---|
| Closed-book (Condition 1) | 60 | 10 s |
| Clean RAG (Condition 2) | 60 | 15 s |
| Edited RAG (Condition 3) | 300 | 19 s |

At these speeds, the full experiment (about 100 facts, 2,100 prompts) should take about 10 hours on the same GPU. On a laptop CPU, one RAG prompt takes about 13 minutes, so the CPU is practical only for testing one or two facts. For the same reason, the demo notebook shows saved outputs rather than running Llama live.

## Results

Each folder in `results/` holds one run. Every number in the midterm report comes from the Colab GPU run, `results/colab-l4-run1/`.

| Folder | What it holds | Use it for |
|---|---|---|
| `colab-l4-run1/` | Conditions 1 to 3 for the 20-fact midterm sample on a Colab L4 GPU (420 prompts). `run_info.json` records the commit and GPU, and `analysis/` holds the tables (`summary.md`), the figures, the probe pilot, and the hand-labelling files | Every midterm number |
| `screen_c1_cpu_2026-10-28/` | Closed-book greedy answers for the first 59 facts, run on the laptop CPU | How the midterm sample of facts was picked |
| `archive_2026-10-25_default-date/` | The first baseline and worked example on 8 facts, run with the default chat date and with slots drawn separately for each m | Evidence for the chat-date and nested-slot decisions. The report does not use its numbers |
| `archive_2026-10-26_partial-64-tokens/` | A rerun with the fixed chat date, nested slots, and a 64-token cap, stopped after 17 of 48 rows | Reference only, replaced by the 192-token runs. The report does not use its numbers |

Hidden-state arrays (the `hidden/` folders) are not in git, because they are large and can be regenerated.
