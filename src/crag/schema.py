"""The shapes of the records the pipeline uses.

Each record is one JSON object, which Python loads as a plain dict. These are
TypedDicts: at runtime they are ordinary dicts, but mypy knows which keys each
one has and what type each value is, so a typo like fact["questions "] or a
missing key is caught before the code runs.

NotRequired marks keys that only some records have.

How the records connect:

    Document    (a scraped page or PDF)
       |    passages.chunk()
    Passage     (~120+ words of one document)
       |    picked by hand, 5 per fact
    Fact        (question, true/counterfactual value, its 5 passages)
       |    run_baseline.py builds a prompt, Llama answers
    Answer    (Llama's raw output for one prompt)
       |    score.py labels it
    Row       (one scored example, saved to results/baseline.jsonl)

Each class docstring below has a shortened real-looking example.
"""

from typing import NotRequired, TypedDict

import numpy as np
from numpy.typing import NDArray


class Document(TypedDict):
    """A scraped web page (scrape.py) or the text of one linked PDF (pdfs.py).

    Example (a web page; text shortened):

        {
          "url": "https://www.bakerinstitute.org/research/whither-nafta",
          "title": "Whither NAFTA?",
          "date": "January 13, 2017",
          "authors": ["..."],
          "tags": ["Mexico", "Trade"],
          "paragraphs": ["NAFTA and the United States Since 1994, trade between ...", "..."],
          "pdfs": []
        }

    A PDF has no authors/tags/pdfs; instead "source" is "pdf" and "page_url"
    points back to the web page that links to it.
    """

    url: str
    title: str
    date: str                       # e.g. "September 25, 2025", or "" if not found
    paragraphs: list[str]           # for PDFs, these are sentences
    authors: NotRequired[list[str]] # pages only
    tags: NotRequired[list[str]]    # pages only
    pdfs: NotRequired[list[str]]    # pages only: links to Baker PDFs
    page_url: NotRequired[str]      # PDFs only: the page that links to the PDF
    source: NotRequired[str]        # "pdf" for PDFs; pages leave it out


class Passage(TypedDict):
    """One chunk of a document, at least CHUNK_WORDS (120) words (passages.py).

    Note that the length is approximate because chunk() only cuts between paragraphs,
    never inside one (see passages.chunk for the numbers).

    Example (text shortened):

        {
          "id": "tn-visas-role-us-labor-market-and-its-uncertain-future#0",
          "url": "https://www.bakerinstitute.org/research/tn-visas-role-us-labor-market-and-its-uncertain-future",
          "page_url": "https://www.bakerinstitute.org/research/tn-visas-role-us-labor-market-and-its-uncertain-future",
          "title": "TN Visa's Role in the US Labor Market and Its Uncertain Future",
          "date": "August 12, 2025",
          "source": "page",
          "text": "Introduction The North American Free Trade Agreement (NAFTA), which entered into force in 1994, ...",
          "score": 0.61     # only after search() or rank()
        }

    "id" is the last part of the URL plus the passage's position in the document
    (#0 is the first passage). For a PDF passage, "url" is the PDF and "page_url"
    is the page that links to it, so two passages with the same page_url come
    from the same document.
    """

    id: str        # "<slug>#<n>", e.g. "elderly-undocumented-workers#3"
    url: str       # the page or PDF the text came from
    page_url: str  # the page; same as url for web pages
    title: str
    date: str
    source: str    # "page" or "pdf"
    text: str
    score: NotRequired[float]  # cosine similarity, set by search() and rank()


class Fact(TypedDict):
    """One hand-checked fact from data/facts.json (see facts.py).

    Example (passages shortened):

        {
          "id": "nafta_start",
          "group": null,
          "true_value": "1994",
          "cf_value": "1996",
          "questions": ["In what year did NAFTA go into effect?",
                        "When did the North American Free Trade Agreement enter into force?",
                        "NAFTA took effect in which year?"],
          "true_patterns": ["\\\\b1994\\\\b"],
          "cf_patterns": ["\\\\b1996\\\\b"],
          "edits": [["1994", "1996"]],
          "passage_ids": ["tn-visas-role-us-labor-market-and-its-uncertain-future#0", "..."],
          "passages": [{...Passage...}, ...]
        }

    group: whether Llama already knows the fact. Condition 1 (closed-book: the
    question with no passages) decides it. If Llama answers correctly for all 3
    wordings, the fact is "parametric" (known from pretraining); otherwise it is
    "extractive" (it can only come from the passages). Until Condition 1 has been
    run and checked, we do not know yet, so the value is None (null in the JSON).

    The value appears in three forms, because it is used in three different places:

    - true_value / cf_value: the value written once, plainly, for people to read
      (tables, the worked example, the report). The code never matches on these.
    - true_patterns / cf_patterns: regexes that score.py searches for in Llama's
      ANSWER. Answers can phrase the value in ways the corpus does not, so a
      fact can list several patterns. For a name fact the true patterns might be
      ["Sheinbaum"] so that "Claudia Sheinbaum" and "President Sheinbaum" both count.
      \\b means "word boundary", so \\b1994\\b does not match inside "19945".
    - edits: exact text replacements that edit.py makes in the PASSAGES (the input).
      Each pair is [text as written in the corpus, text to put instead]. A fact
      lists one pair per way the corpus writes the value, e.g.
      [["Claudia Sheinbaum", "Xóchitl Gálvez"], ["Sheinbaum", "Gálvez"]], so that
      every mention in a passage changes and the edited passage reads naturally.

    In short: patterns read the model's output, edits write the model's input.
    """

    id: str
    group: str | None               # "parametric", "extractive", or None until Condition 1 is run
    true_value: str
    cf_value: str
    questions: list[str]            # 3 wordings
    true_patterns: list[str]        # regexes for score.py
    cf_patterns: list[str]
    edits: list[list[str]]          # [true form, counterfactual form] pairs for edit.py
    passage_ids: list[str]          # 5 passages from 5 different documents
    passages: list[Passage]         # copied in by `python -m crag.facts fill`
    note: NotRequired[str]
    checked_by: NotRequired[str]    # who confirmed the values and passages; "claude" for facts Claude checked (decided 2026-10-26)


class Answer(TypedDict):
    """What Llama.answer() returns for one prompt (generate.py).

    Example:

        {
          "prompt_tokens": 1020,
          "greedy": "NAFTA went into effect in 1994.",
          "samples": ["1994.", "NAFTA took effect on January 1, 1994.", ...],   # 10 of these
          "hidden": array of shape (32, 4096)
        }

    "greedy" is the single most likely answer and gives the example's label.
    "samples" are random draws from the model and give the true-answer rate.
    "hidden" is the last prompt token's vector at each of the 32 layers (probe input).
    """

    prompt_tokens: int
    greedy: str
    samples: list[str]
    hidden: NDArray[np.float32] | None  # shape (32, 4096), or None if not requested


class Row(TypedDict):
    """One line of results/baseline.jsonl (run_baseline.py).

    Example (a Condition 2 row; prompt and samples shortened):

        {
          "fact": "nafta_start", "condition": 2, "wording": 0, "m": 0,
          "question": "In what year did NAFTA go into effect?",
          "prompt": "Use the passages below to answer the question. ...",
          "prompt_tokens": 1020,
          "greedy": "NAFTA went into effect in 1994.",
          "label": "true", "failure": null,
          "samples": ["1994.", ...], "sample_labels": ["true", ...], "true_rate": 0.9, "cf_rate": 0.0, "both_rate": 0.1,
          "edited_slots": [], "passage_ids": ["...#0", ...], "retrieval_scores": [0.61, ...]
        }

    "wording" and "edited_slots" are 0-based (wording 0 is the first question;
    slot 0 is Passage 1). A closed-book row (condition 1) has m = None and no
    edited_slots, passage_ids, or retrieval_scores.
    """

    fact: str
    condition: int                  # 1, 2, or 3
    wording: int                    # 0, 1, or 2
    m: int | None                   # None for closed-book
    question: str
    prompt: str
    prompt_tokens: int
    greedy: str
    label: str                      # "true", "cf", "both", or "neither"
    failure: str | None
    samples: list[str]
    sample_labels: list[str]
    true_rate: float | None         # None when no samples were drawn
    cf_rate: NotRequired[float | None]    # fraction of samples stating only the counterfactual
    both_rate: NotRequired[float | None]  # fraction of samples stating both values (mixed)
    edited_slots: NotRequired[list[int]]       # RAG conditions only
    passage_ids: NotRequired[list[str]]
    retrieval_scores: NotRequired[list[float]]
