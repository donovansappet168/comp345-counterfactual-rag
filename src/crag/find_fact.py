"""Helper for picking facts by hand.

    python -m crag.find_fact "NAFTA.{0,80}1994"
    python -m crag.find_fact "NAFTA.{0,80}1994" --context 300

Prints every passage that matches the regex, grouped by document, with the passage id.
A passage whose matching sentence copies one printed earlier (often a web page and its
own PDF) is marked "copy of ...", so you can see at a glance how many *non-identical*
passages state the fact. A fact needs 5 of those; they may come from the same document.
Copy the ids you want into data/facts.json, then run `python -m crag.facts fill` to
copy in the text.
"""

import argparse
import re
from collections import defaultdict

from crag.passages import load, near_copy, sentence_words
from crag.schema import Passage


def find(pattern: str, passages: list[Passage],
         context: int = 150) -> dict[str, list[tuple[Passage, str, list[str]]]]:
    """Find passages matching a regex and group them by document.

    Args:
        pattern: The regex (case-insensitive).
        passages: All passages.
        context: How many characters to keep on each side of the match.

    Returns:
        A dict from page URL to a list of (passage, snippet around the match,
        words of the sentence that contains the match).
    """
    by_document: dict[str, list[tuple[Passage, str, list[str]]]] = defaultdict(list) # page URL -> list of (passage, snippet, sentence); a page and its PDF share a page URL
    for p in passages: # Sweep over passages
        match = re.search(pattern, p["text"], re.IGNORECASE) # Look for the pattern anywhere in the passage (any capitalization)
        if match: # If it matches
            start = max(0, match.start() - context) # Snippet start: `context` characters before the match (not before the text start)
            snippet = p["text"][start:match.end() + context] # Snippet: the match plus `context` characters on each side
            sentence = sentence_words(p["text"], match.start(), match.end()) # The sentence the match is in (for spotting copies)
            by_document[p["page_url"]].append((p, snippet, sentence)) # Group it under its document
    return by_document # Return the matches by document


def main() -> None:
    """Print the matches for the regex given on the command line."""
    parser = argparse.ArgumentParser() # Command-line options
    parser.add_argument("pattern") # The regex to search for
    parser.add_argument("--context", type=int, default=150, help="characters shown around the match") # --context N: characters shown around each match
    args = parser.parse_args() # Read the options

    passages, _ = load() # Load every passage (embeddings not needed)
    by_document = find(args.pattern, passages, args.context) # Find matches, grouped by document

    kept: list[tuple[str, list[str]]] = [] # (passage id, sentence) of every non-identical passage so far
    lines: list[str] = [] # Output, printed after the summary line
    for document, hits in by_document.items(): # Sweep over documents
        first = hits[0][0] # First matching passage (for the title and date)
        lines.append(f"=== {first['title']} ({first['date']})") # Document title and date
        lines.append(f"    {document}") # Document URL
        for p, snippet, sentence in hits: # Sweep over its matching passages
            copy_of = next((pid for pid, other in kept if near_copy(sentence, other)), None) # An earlier passage with the same sentence, if any
            if copy_of is None: # If this sentence is new
                kept.append((p["id"], sentence)) # Count it
                mark = "" # Nothing to mark
            else: # Otherwise
                mark = f" (copy of {copy_of})" # Mark it as a copy
            lines.append(f"  [{p['id']}] ({p['source']}){mark} ...{snippet}...") # Passage id, page or PDF, copy mark, and the snippet
        lines.append("") # Blank line between documents

    n_passages = sum(len(hits) for hits in by_document.values()) # Total matching passages
    print(f"{n_passages} passages in {len(by_document)} documents, "
          f"{len(kept)} non-identical\n") # Summary line: passages, documents, non-identical passages
    print("\n".join(lines)) # The matches


if __name__ == "__main__":
    main() # Print matches for the regex on the command line
