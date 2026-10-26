"""Helper for picking facts by hand.

    python -m crag.find_fact "NAFTA.{0,80}1994"
    python -m crag.find_fact "NAFTA.{0,80}1994" --context 300

Prints every passage that matches the regex, grouped by document, with the passage id.
A web page and its own PDF count as the same document (same "page_url"), so you can
see at a glance how many *different* documents state the fact. Copy the ids you
want into data/facts.json, then run `python -m crag.facts fill` to copy in the text.
"""

import argparse
import re
from collections import defaultdict

from crag.passages import load
from crag.schema import Passage


def find(pattern: str, passages: list[Passage],
         context: int = 150) -> dict[str, list[tuple[Passage, str]]]:
    """Find passages matching a regex and group them by document.

    Args:
        pattern: The regex (case-insensitive).
        passages: All passages.
        context: How many characters to keep on each side of the match.

    Returns:
        A dict from page URL to a list of (passage, snippet around the match).
    """
    by_document: dict[str, list[tuple[Passage, str]]] = defaultdict(list) # page URL -> list of (passage, snippet); a page and its PDF share a page URL
    for p in passages: # Sweep over passages
        match = re.search(pattern, p["text"], re.IGNORECASE) # Look for the pattern anywhere in the passage (any capitalization)
        if match: # If it matches
            start = max(0, match.start() - context) # Snippet start: `context` characters before the match (not before the text start)
            snippet = p["text"][start:match.end() + context] # Snippet: the match plus `context` characters on each side
            by_document[p["page_url"]].append((p, snippet)) # Group it under its document
    return by_document # Return the matches by document


def main() -> None:
    """Print the matches for the regex given on the command line."""
    parser = argparse.ArgumentParser() # Command-line options
    parser.add_argument("pattern") # The regex to search for
    parser.add_argument("--context", type=int, default=150, help="characters shown around the match") # --context N: characters shown around each match
    args = parser.parse_args() # Read the options

    passages, _ = load() # Load every passage (embeddings not needed)
    by_document = find(args.pattern, passages, args.context) # Find matches, grouped by document

    n_passages = sum(len(hits) for hits in by_document.values()) # Total matching passages
    print(f"{n_passages} passages in {len(by_document)} different documents\n") # Summary line: how many passages, how many different documents
    for document, hits in by_document.items(): # Sweep over documents
        first = hits[0][0] # First matching passage (for the title and date)
        print(f"=== {first['title']} ({first['date']})") # Document title and date
        print(f"    {document}") # Document URL
        for p, snippet in hits: # Sweep over its matching passages
            print(f"  [{p['id']}] ({p['source']}) ...{snippet}...") # Passage id, page or PDF, and the snippet
        print() # Blank line between documents


if __name__ == "__main__":
    main() # Print matches for the regex on the command line
