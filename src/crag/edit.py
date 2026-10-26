"""Counterfactual edits.

Each fact in data/facts.json has an "edits" list of [true form, counterfactual form]
pairs, e.g. [["1994", "1996"]]. Editing a passage replaces every true form with its
counterfactual form. The edited slots are nested: each (fact, wording) gets one random
order of the 5 slots (fixed seed), and the first m slots of that order are edited.
"""

import random
import re

from crag.config import K, SEED
from crag.schema import Fact, Passage


def edit_passage(text: str, fact: Fact) -> str:
    """Replace every true form of the fact's value with its counterfactual form.

    Longer forms are replaced first, so "Claudia Sheinbaum" becomes "Xóchitl Gálvez"
    before "Sheinbaum" on its own is replaced (otherwise we would get "Claudia Gálvez").
    A form only matches as a whole number or word: "46" does not match inside
    "1946" or "2.46", and "Sheinbaum" does not match inside "SheinbaumPardo".

    Args:
        text: The original passage text.
        fact: The fact, whose "edits" list gives the replacements.

    Returns:
        The edited passage text.

    Raises:
        ValueError: If the edit would leave the passage unchanged (the passage
            does not actually state the value in any of the listed forms).
    """
    new_text = text # Start edited text as the original text
    pairs = sorted(fact["edits"], key=lambda pair: len(pair[0]), reverse=True) # Sort pairs longest true form first, so longer forms (e.g. first + last name) are edited before shorter forms inside them (e.g. last name)
    for true_form, cf_form in pairs: # Sweep over pairs of edits, longest first
        pattern = r"(?<![\w.])" + re.escape(true_form) + r"(?!\w)" # Match the form only on its own, not inside a longer number or word (e.g. "1994" but not "11994" or "0.1994")
        new_text = re.sub(pattern, cf_form, new_text) # Replace every match with the counterfactual form
    if new_text == text: # If nothing changed, the passage never states the value in a listed form
        raise ValueError(f"edit for {fact['id']} did not change this passage: {text[:80]}") # Raise error
    return new_text # Return edited text


def edited_slots(fact_id: str, wording: int, m: int) -> list[int]:
    """Pick which m of the 5 slots to edit.

    The slots are NESTED: one random order of the 5 slots is drawn for each
    (fact, wording), and the first m slots of that order are edited. So going from
    m to m + 1 adds exactly one more edited slot and changes nothing else, and each
    fact's curve over m is a clean dose-response. Position still varies, because
    every fact and wording gets its own random order (decided 2026-10-26; the
    proposal drew a separate set of slots for every m).

    The random generator is seeded with the fact and wording, so a rerun always
    gives the same order.

    Args:
        fact_id: The fact's id.
        wording: Which question wording (0, 1, or 2).
        m: How many slots to edit (0 to 5).

    Returns:
        The edited slots as 0-based indices in increasing order, e.g. [1, 3].
    """
    rng = random.Random(f"{SEED}|{fact_id}|{wording}") # New RNG seeded with SEED + fact id + wording, so each fact + wording always gets the same order (m is left out so the slots nest)
    order = rng.sample(range(K), K) # Random order of all 5 slots (0-based), e.g. [3, 0, 4, 1, 2]
    return sorted(order[:m]) # Edit the first m slots of that order, return them sorted


def build_context(passages: list[Passage], fact: Fact, wording: int,
                  m: int) -> tuple[list[str], list[int]]:
    """Make the 5 passage texts for one example, with m of them edited.

    Args:
        passages: The fact's 5 passages in slot order.
        fact: The fact being tested.
        wording: Which question wording (0, 1, or 2).
        m: How many passages to edit.

    Returns:
        The 5 passage texts and the list of edited slots (0-based).
    """
    slots = edited_slots(fact["id"], wording, m) # Choose which passages to edit
    texts = [] # Updated slot texts
    for i, p in enumerate(passages): # Sweep over passages
        if i in slots: # If passage is to be edited
            texts.append(edit_passage(p["text"], fact)) # Edit it: swap in the counterfactual value
        else:
            texts.append(p["text"]) # Otherwise add original passage
    return texts, slots # Return the updated texts, and the slots that were edited
