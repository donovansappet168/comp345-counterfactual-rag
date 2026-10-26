"""Scoring answers with a script (no LLM judge).

Each fact lists regex patterns for its true value ("true_patterns") and its
counterfactual value ("cf_patterns"). An answer gets one of four labels:

    "true"     only the true value is found
    "cf"       only the counterfactual value is found
    "both"     both are found     -> label by hand
    "neither"  neither is found   -> label by hand
"""

import re

from crag.schema import Fact

# Failure taxonomy from the proposal, for answers that are not "true".
# Refusals are guessed from common phrases; check these by hand.
REFUSAL_WORDS = ["cannot", "can't", "unable to", "not mentioned", "not stated",
                 "not provided", "no information", "don't know", "do not know",
                 "don't have", "do not have"] # e.g. "I don't have information on ..."


def clean(answer: str) -> str:
    """Remove commas so number patterns do not have to handle them.

    Args:
        answer: The model's answer.

    Returns:
        The answer with commas removed, e.g. "1,246" becomes "1246".
    """
    return answer.replace(",", "") # Drop every comma ("1,246" -> "1246")


def found(patterns: list[str], text: str) -> bool:
    """Check whether any of the regex patterns appears in the text (case-insensitive).

    Args:
        patterns: Regex patterns, e.g. [r"\\b1994\\b"].
        text: The (cleaned) answer.

    Returns:
        True if at least one pattern matches.
    """
    for pattern in patterns: # Sweep over patterns
        if re.search(pattern, text, re.IGNORECASE): # If the pattern appears anywhere in the text (any capitalization)
            return True # One match is enough
    return False # No pattern matched


def label(answer: str, fact: Fact) -> str:
    """Label an answer as "true", "cf", "both", or "neither".

    Args:
        answer: The model's answer.
        fact: The fact the question was about (supplies the patterns).

    Returns:
        One of "true", "cf", "both", "neither".
    """
    text = clean(answer) # Remove commas first
    has_true = found(fact["true_patterns"], text) # Does the answer state the true value?
    has_cf = found(fact["cf_patterns"], text) # Does it state the counterfactual value?
    if has_true and has_cf: # Both values -> mixed answer, labelled by hand later
        return "both"
    if has_true: # Only the true value
        return "true"
    if has_cf: # Only the counterfactual value
        return "cf"
    return "neither" # Neither value (refusal or other wrong answer), labelled by hand later


def failure_type(answer: str, lab: str) -> str | None:
    """Sort an answer that is not "true" into the proposal's failure categories.

    Args:
        answer: The model's answer.
        lab: The label from label().

    Returns:
        "adopted counterfactual", "mixed", "refusal", or "other wrong answer",
        or None if the answer was labelled "true".
    """
    if lab == "cf": # States only the counterfactual
        return "adopted counterfactual"
    if lab == "both": # States both values
        return "mixed"
    if lab == "neither": # States neither value
        if any(word in answer.lower() for word in REFUSAL_WORDS): # If it uses a refusal phrase (e.g. "I don't have information")
            return "refusal"
        return "other wrong answer" # Otherwise it gave some other value
    return None # "true" answers are not failures
