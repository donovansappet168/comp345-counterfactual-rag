"""Loading and saving data/facts.json.

Each fact is a Fact record (see schema.py). A filled-in example:

    {
      "id": "nafta_start",
      "group": null,                      # "parametric"/"extractive", set after Condition 1
      "true_value": "1994",
      "cf_value": "1996",
      "questions": ["wording 1", "wording 2", "wording 3"],
      "true_patterns": ["\\\\b1994\\\\b"],    # regexes used by score.py
      "cf_patterns": ["\\\\b1996\\\\b"],
      "edits": [["1994", "1996"]],        # [true form, counterfactual form] pairs for edit.py
      "passage_ids": ["...#3", ...],      # 5 passages from 5 different documents
      "passages": [...]                   # filled in by `fill`
    }

To add a fact: find passages with `python -m crag.find_fact "<regex>"`, write the fact
with its passage_ids into data/facts.json (with "passages": []), then run

    python -m crag.facts fill

which copies each passage into the fact and checks that the edit works on it.
"""

import json
import sys

from crag.config import FACTS, MAX_FACT_PASSAGE_WORDS
from crag.schema import Fact


def load() -> list[Fact]:
    """Read every fact from data/facts.json.

    Returns:
        The facts, in file order.
    """
    with open(FACTS, encoding="utf-8") as f: # Open data/facts.json
        facts: list[Fact] = json.load(f) # Parse the JSON list of facts
    return facts # Return the facts


def get(fact_id: str) -> Fact:
    """Look up one fact by id.

    Args:
        fact_id: The fact's id, e.g. "nafta_start".

    Returns:
        The fact.

    Raises:
        KeyError: If no fact has that id.
    """
    for fact in load(): # Sweep over facts
        if fact["id"] == fact_id: # If this is the fact we want
            return fact # Return it
    raise KeyError(fact_id) # No fact with that id


def save(facts: list[Fact]) -> None:
    """Write the facts back to data/facts.json.

    Args:
        facts: Every fact (the file is overwritten).
    """
    with open(FACTS, "w", encoding="utf-8") as f: # Open data/facts.json for writing (overwrites it)
        json.dump(facts, f, indent=2, ensure_ascii=False) # Write the facts as readable JSON, keeping accents (e.g. Gálvez) as-is


def fill() -> None:
    """Copy each fact's passages (by passage_ids) from passages.jsonl into facts.json.

    Also checks each fact: the edit must change every chosen passage, the
    passages should come from different documents, and each should be at most
    MAX_FACT_PASSAGE_WORDS long, since MiniLM only reads about the first 200 words,
    and the counterfactual value must not already appear in an unedited passage
    (e.g. a passage that mentions "May 2021" for some other reason cannot be used
    when the counterfactual is 2021, or the clean m = 0 context would contain it).
    A warning is printed for the last three.
    """
    from crag.edit import edit_passage # Imported here so `facts.load()` stays fast (no MiniLM / passages needed)
    from crag.passages import load as load_passages
    from crag.score import clean, found

    passages, _ = load_passages() # Load every passage in the corpus (embeddings not needed)
    by_id = {p["id"]: p for p in passages} # Look-up table: passage id -> passage
    facts = load() # Load the facts
    for fact in facts: # Sweep over facts
        fact["passages"] = [] # Start with an empty passage list
        for pid in fact["passage_ids"]: # Sweep over the fact's chosen passage ids
            fact["passages"].append(by_id[pid]) # Copy the full passage into the fact
            edit_passage(by_id[pid]["text"], fact) # Check the edit changes this passage (raises an error if not)
            n_words = len(by_id[pid]["text"].split()) # Passage length in words
            if n_words > MAX_FACT_PASSAGE_WORDS: # If longer than MiniLM can read
                print(f"warning: {fact['id']} passage {pid} has {n_words} words") # Warn
            if found(fact["cf_patterns"], clean(by_id[pid]["text"])): # If the counterfactual value already appears in the clean passage
                print(f"warning: {fact['id']} passage {pid} already contains the counterfactual value") # Warn (it would leak into the m = 0 context)
        documents = {by_id[pid]["page_url"] for pid in fact["passage_ids"]} # The documents the passages come from (a page and its PDF count as one)
        if len(documents) < len(fact["passage_ids"]): # If two passages share a document
            print(f"warning: {fact['id']} uses two passages from the same document") # Warn
    save(facts) # Write the filled facts back to facts.json
    print(f"filled {len(facts)} facts") # Report how many facts were filled


if __name__ == "__main__":
    if sys.argv[1:] == ["fill"]: # Only `python -m crag.facts fill` does anything
        fill() # Fill and check every fact
