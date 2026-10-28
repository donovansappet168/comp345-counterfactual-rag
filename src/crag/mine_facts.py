"""Rank candidate facts by how many non-identical passages state them.

    python -m crag.mine_facts
    python -m crag.mine_facts --min-passages 8 --top 50

Writes data/fact_candidates.csv, a list to review by hand (with find_fact.py),
not a list of finished facts.

How it works:

1. Every document is split into passages (passages.chunk, so the passage ids match
   passages.jsonl) and every passage into sentences. Sentences that look like
   citations (URLs, "accessed", "et al.") are skipped.
2. In each sentence we find VALUES (full dates, years, percentages, dollar amounts,
   quantities with units) and SUBJECTS (runs of capitalized words, e.g.
   "Winter Storm Uri", "Water Treaty", "USMCA").
3. Each subject that is within MINE_WINDOW words of a value forms a candidate pair,
   e.g. ("NAFTA", "1994"). For each pair we collect the documents that state it.
4. Common pairs are not always facts: "Mexico" and "2024" appear together in
   hundreds of documents only because both are common. So each pair gets a LIFT:

       lift = (share of the subject's documents that also have this value next to it)
              / (share of all documents that mention the value at all)

   A lift of 10 means the value shows up next to this subject 10 times more often
   than it shows up in general. Real facts ("NAFTA" + "1994") have a high lift;
   coincidences ("Mexico" + "2024") have a lift near 1.
5. A web page often copies sentences from its own PDF, so passages that state the
   pair in the same sentence count once (passages.near_copy). Pairs stated in at least
   MINE_MIN_PASSAGES such non-identical passages short enough to use
   (MAX_FACT_PASSAGE_WORDS), with lift >= MINE_MIN_LIFT, are written out, ranked by
   that number. The passages may come from the same document.

Columns in the CSV:
    rank, subject, value, value_type
    documents        different documents that state the pair (page and its PDF count once)
    usable_documents of those, documents with a passage <= MAX_FACT_PASSAGE_WORDS
    passages         all passages stating the pair (a long report can state it several
                     times, and a page and its PDF often repeat the same sentence)
    distinct_passages of the usable passages, how many state it in non-identical
                     sentences (a fact needs 5)
    lift             see step 4
    recent_share     fraction of those documents dated 2024 or later (high = extractive candidate)
    other_values     other values of the same type often stated next to the subject
                     (a warning that the corpus may disagree with itself)
    in_facts_json    "yes" if a fact in data/facts.json already uses this value
    example_1/2      two example sentences from different documents
    passage_ids      up to 10 passage ids, for find_fact.py / facts.json

Note: run `python -m crag.passages` after new pages or PDFs are downloaded, so the
passage ids printed here exist in passages.jsonl when you run `facts fill`.
"""

import argparse
import csv
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field

from crag.config import (FACT_CANDIDATES, MAX_FACT_PASSAGE_WORDS, MINE_MIN_LIFT,
                         MINE_MIN_PASSAGES, MINE_WINDOW)
from crag.facts import load as load_facts
from crag.passages import chunk, count_distinct, read_documents

MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August",
          "September", "October", "November", "December"]
MONTH_ABBREVIATIONS = {"Jan.": "January", "Feb.": "February", "Mar.": "March", "Apr.": "April",
                       "Aug.": "August", "Sept.": "September", "Sep.": "September",
                       "Oct.": "October", "Nov.": "November", "Dec.": "December"}
UNITS = ["acre-feet", "barrels per day", "barrels a day", "bpd", "b/d", "mb/d", "people",
         "deaths", "megawatts", "gigawatts", "MW", "GW", "miles", "kilometers", "residents",
         "workers", "jobs", "students", "members", "countries", "states", "counties"]

# Value patterns, most specific first. A span taken by an earlier pattern is not
# matched again by a later one (so the year inside a full date is not counted twice).
MONTH_NAMES = "|".join(MONTHS + [re.escape(a) for a in MONTH_ABBREVIATIONS])
VALUE_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("date", re.compile(rf"\b(?:{MONTH_NAMES}) \d{{1,2}}, (?:19|20)\d\d\b")),
    ("money", re.compile(r"\$\d[\d,.]*(?: (?:million|billion|trillion))?")),
    ("percent", re.compile(r"\b\d[\d,.]*(?:%| percent\b)")),
    ("quantity", re.compile(r"\b\d[\d,.]*(?: (?:million|billion|thousand))? (?:"
                            + "|".join(re.escape(u) for u in UNITS) + r")\b")),
    ("month_year", re.compile(rf"\b(?:{MONTH_NAMES}) (?:19|20)\d\d\b")),
    ("year", re.compile(r"\b(?:19|20)\d\d\b")),
]

# Capitalized words that are not useful subjects on their own.
STOP_SUBJECTS = {
    "the", "a", "an", "in", "on", "at", "for", "and", "but", "this", "that", "these", "those",
    "it", "its", "as", "by", "from", "with", "however", "since", "after", "before", "during",
    "while", "when", "although", "according", "also", "under", "between", "both", "while",
    "u.s.", "us", "united states", "america", "american", "americans", "figure", "table",
    "source", "sources", "note", "notes", "baker institute", "rice university", "president",
    "we", "our", "they", "their", "he", "she", "his", "her", "i", "if", "of", "to",
    "first", "second", "third", "new", "one", "two", "three", "last", "next", "today",
    "ci", "gw", "mw", "kw", "twh", "bcf", "tcf", # Units and statistics, not subjects
}
# Months never belong in a subject: "In March 2020" should not give the subject "March".
MONTH_WORDS = {m.lower() for m in MONTHS} | {a.lower().rstrip(".") for a in MONTH_ABBREVIATIONS}
# Subjects containing these are the institute's own boilerplate ("Since 1993, the Baker Institute ...").
BOILERPLATE_SUBJECTS = ["baker institute", "rice university", "policy playbook"]
CONNECTORS = {"of", "the", "and", "for", "de", "del", "la", "on"}
# Sentences that look like citations, including in-text ones such as
# "(Zodrow and Diamond 2013)" or "Goulder (1989)", are skipped.
CITATION = re.compile(r"https?://|www\.|doi\.org|\baccessed\b|\bet al\.|\bRetrieved\b|\bpp\."
                      r"|\([A-Z][\w.&,’' -]{1,60} (?:19|20)\d\d[a-z]?\)"
                      r"|\b[A-Z][a-z]+ \((?:19|20)\d\d[a-z]?\)", re.IGNORECASE)
SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z\"“])")


@dataclass
class Candidate:
    """Everything we collect about one (subject, value) pair while scanning."""

    subject: str
    value: str
    value_type: str
    documents: set[str] = field(default_factory=set)
    usable_documents: set[str] = field(default_factory=set)
    recent_documents: set[str] = field(default_factory=set)
    passages: set[str] = field(default_factory=set) # Every passage id stating the pair
    passage_ids: list[str] = field(default_factory=list) # The first 10, for the CSV
    usable_sentences: dict[str, list[str]] = field(default_factory=dict) # Usable passage id -> words of its sentence stating the pair
    distinct_passages: int = 0 # Set in scan(): usable passages with non-identical sentences
    examples: dict[str, str] = field(default_factory=dict) # Document -> one example sentence
    spellings: Counter[str] = field(default_factory=Counter) # How often each capitalization appears
    lift: float = 0.0 # Set in scan()


def normalize_value(text: str, value_type: str) -> str:
    """Write a value one standard way, so "Feb. 28, 2026" and "February 28, 2026" match.

    Args:
        text: The value as written in the sentence.
        value_type: Which pattern matched ("date", "percent", ...).

    Returns:
        The value in standard form.
    """
    text = text.strip() # Trim spaces
    for short, full in MONTH_ABBREVIATIONS.items(): # Sweep over month abbreviations
        if text.startswith(short): # If the value starts with one (e.g. Feb.)
            text = full + text[len(short):] # Write the full month name instead
    if value_type == "percent": # If a percentage
        text = text.replace(" percent", "%") # 20 percent -> 20%
    return text # Return the standard form


def find_values(sentence: str) -> list[tuple[str, str, int]]:
    """Find the values in a sentence.

    Args:
        sentence: One sentence.

    Returns:
        (normalized value, value type, character position) for each value found.
    """
    taken: list[tuple[int, int]] = [] # Character spans already matched
    values = [] # (value, type, position) found
    for value_type, pattern in VALUE_PATTERNS: # Sweep over value patterns, most specific first
        for match in pattern.finditer(sentence): # Sweep over its matches in the sentence
            start, end = match.span() # Where the match starts and ends
            if any(start < t_end and end > t_start for t_start, t_end in taken): # If it overlaps a span already taken (e.g. the year inside a full date)
                continue # Skip it
            taken.append((start, end)) # Mark the span taken
            values.append((normalize_value(match.group(0), value_type), value_type, start)) # Keep the value in standard form
    return values # Return the values


def clean_word(word: str) -> str:
    """Strip punctuation and a possessive "’s" from the ends of a word.

    Args:
        word: One whitespace-separated token.

    Returns:
        The word without surrounding punctuation, e.g. "NAFTA’s," becomes "NAFTA".
    """
    word = word.strip("()[]{}\"“”‘’'.,;:!?") # Strip brackets, quotes, and punctuation from both ends
    for suffix in ("’s", "'s"): # Sweep over the two apostrophe kinds
        if word.endswith(suffix): # If the word ends with a possessive
            word = word[: -len(suffix)] # Drop it
    return word # Return the cleaned word


def is_capitalized(word: str) -> bool:
    """Check whether a word can be part of a subject: capitalized, and not a month.

    Args:
        word: A cleaned word.

    Returns:
        True if the first character is an uppercase letter and the word is not a
        month name or month abbreviation.
    """
    return bool(word) and word[0].isupper() and word.lower() not in MONTH_WORDS # Non-empty, starts uppercase, not a month


def ends_clause(word: str) -> bool:
    """Check whether a word ends with punctuation that closes a name (",", ";", ":", ".").

    Abbreviations such as "U.S." and "Jr." end with a period but do not end the name.

    Args:
        word: One whitespace-separated token, before cleaning.

    Returns:
        True if a subject should stop after this word.
    """
    if word.endswith(("U.S.", "Jr.", "St.")): # If an abbreviation with a period
        return False # It does not end the name
    return word[-1] in ",;:." # Ends with , ; : or .


def find_subjects(sentence: str) -> list[tuple[str, int]]:
    """Find runs of capitalized words that could be the subject of a fact.

    A run may contain small connector words ("Strait of Hormuz", "Bank of Mexico")
    but must start and end with a capitalized word. Runs that are only a common word
    ("The", "However") or a month are dropped.

    Args:
        sentence: One sentence.

    Returns:
        (subject, character position) for each subject found.
    """
    words = [(m.group(0), m.start()) for m in re.finditer(r"\S+", sentence)] # (word, character position) for every word
    subjects = [] # (subject, position) found
    i = 0 # Current word
    while i < len(words): # Sweep over words
        if not is_capitalized(clean_word(words[i][0])): # If the word cannot start a subject
            i += 1 # Move on
            continue # Next word
        run = [i] # Word positions in this run
        j = i + 1 # Next word to look at
        # A run ends at punctuation like "Mexico," so a list does not merge into one subject.
        while j < len(words) and not ends_clause(words[run[-1]][0]): # Extend the run until a word closes it with punctuation
            word = clean_word(words[j][0]) # The next word, cleaned
            if is_capitalized(word): # If capitalized
                run.append(j) # Add it
            elif word.lower() in CONNECTORS and j + 1 < len(words) \
                    and is_capitalized(clean_word(words[j + 1][0])): # If a connector (of, the, de, ...) followed by a capitalized word
                run.append(j) # Add the connector
            else: # Otherwise
                break # The run is over
            j += 1 # Next word
        next_i = run[-1] + 1 # Where to continue after this run
        # Drop common words at the start of the run: "In Mexico" -> "Mexico", "The USMCA" -> "USMCA".
        while run and clean_word(words[run[0]][0]).lower() in STOP_SUBJECTS: # While the run starts with a common word
            run = run[1:] # Drop that word
        text = " ".join(clean_word(words[k][0]) for k in run) # The subject text
        boilerplate = any(b in text.lower() for b in BOILERPLATE_SUBJECTS) # Is it the institute's own boilerplate?
        if text and text.lower() not in STOP_SUBJECTS and not boilerplate: # If something useful is left
            subjects.append((text, words[run[0]][1])) # Keep it with its position
        i = next_i # Continue after the run
    return subjects # Return the subjects


def word_distance(sentence: str, a: int, b: int) -> int:
    """Count the words between two character positions in a sentence.

    Args:
        sentence: The sentence.
        a: First character position.
        b: Second character position.

    Returns:
        Number of whitespace-separated words between the two positions.
    """
    low, high = min(a, b), max(a, b) # Smaller and larger position
    return len(sentence[low:high].split()) # Words in the text between them


def document_year(date: str) -> int:
    """Read the year from a date like "August 12, 2025".

    Args:
        date: The document's date string.

    Returns:
        The year, or 0 if the date is empty or has no year.
    """
    match = re.search(r"(19|20)\d\d$", date) # A 4-digit year at the end of the date
    return int(match.group(0)) if match else 0 # The year, or 0 if none


def scan(min_passages: int, min_lift: float) -> list[Candidate]:
    """Scan the whole corpus and return the candidates that pass both filters.

    Args:
        min_passages: Minimum number of usable, non-identical passages stating the pair.
        min_lift: Minimum lift (see the module docstring).

    Returns:
        The candidates, not yet sorted.
    """
    candidates: dict[tuple[str, str], Candidate] = {} # (subject, value) -> Candidate
    subject_documents: dict[str, set[str]] = defaultdict(set) # Subject -> documents that mention it
    value_documents: dict[str, set[str]] = defaultdict(set) # Value -> documents that mention it
    all_documents: set[str] = set() # Every document

    for doc in read_documents(): # Sweep over documents (pages, then PDFs)
        document = doc.get("page_url", doc["url"]) # The page URL (a page and its PDF count as one document)
        all_documents.add(document) # Count it
        recent = document_year(doc["date"]) >= 2024 # Dated 2024 or later?
        slug = doc["url"].split("/")[-1] # Last part of the URL, used in passage ids
        for n, text in enumerate(chunk(doc["paragraphs"])): # Sweep over passages (same chunking as passages.py, so ids match)
            passage_id = f"{slug}#{n}" # Passage id, e.g. whither-nafta#1
            usable = len(text.split()) <= MAX_FACT_PASSAGE_WORDS # Short enough for a fact passage?
            for sentence in SENTENCE_SPLIT.split(text): # Sweep over sentences
                if CITATION.search(sentence): # If it looks like a citation
                    continue # Skip it
                values = find_values(sentence) # Values in the sentence
                subjects = find_subjects(sentence) # Subjects in the sentence
                for subject, _ in subjects: # Sweep over subjects
                    subject_documents[subject.lower()].add(document) # Remember this document mentions it
                for value, _, _ in values: # Sweep over values
                    value_documents[value].add(document) # Remember this document mentions it
                for value, value_type, v_pos in values: # Sweep over values
                    for subject, s_pos in subjects: # Sweep over subjects
                        if word_distance(sentence, v_pos, s_pos) > MINE_WINDOW: # If too far apart in the sentence
                            continue # Not a pair
                        key = (subject.lower(), value) # Pair key (subject lowercased)
                        if key not in candidates: # If this pair is new
                            candidates[key] = Candidate(subject, value, value_type) # Start a Candidate for it
                        c = candidates[key] # The pair's Candidate
                        c.spellings[subject] += 1 # Count this spelling of the subject
                        c.documents.add(document) # Add the document
                        if usable: # If the passage is short enough
                            c.usable_documents.add(document) # Add it to the usable documents
                            if passage_id not in c.usable_sentences: # First sentence stating the pair in this passage
                                c.usable_sentences[passage_id] = re.findall(r"\w+", sentence.lower()) # Keep its words, for spotting copies
                        if recent: # If the document is recent
                            c.recent_documents.add(document) # Add it to the recent documents
                        c.passages.add(passage_id) # Add the passage id
                        if passage_id not in c.passage_ids and len(c.passage_ids) < 10:
                            c.passage_ids.append(passage_id) # Keep up to 10 ids for the CSV
                        if document not in c.examples:
                            c.examples[document] = sentence # Keep one example sentence per document

    kept = [] # Candidates that pass both filters
    n_all = len(all_documents) # Number of documents in the corpus
    for (subject_key, value), c in candidates.items(): # Sweep over candidates
        if len(c.usable_sentences) < min_passages: # If too few usable passages, even counting copies
            continue # Skip it (cheap check first)
        c.distinct_passages = count_distinct(list(c.usable_sentences.values())) # Usable passages that are not copies of each other
        if c.distinct_passages < min_passages: # If too few non-identical passages
            continue # Skip it
        c.lift = lift_of(c, subject_documents[subject_key], value_documents[value], n_all) # Compute its lift
        if c.lift >= min_lift: # If the lift is high enough
            c.subject = c.spellings.most_common(1)[0][0] # Use the most common spelling of the subject
            kept.append(c) # Keep it
    return kept # Return the kept candidates


def lift_of(c: Candidate, subject_docs: set[str], value_docs: set[str], n_all: int) -> float:
    """Compute the lift of a candidate (see step 4 of the module docstring).

    Args:
        c: The candidate.
        subject_docs: Documents that mention the subject anywhere.
        value_docs: Documents that mention the value anywhere.
        n_all: Number of documents in the corpus.

    Returns:
        The lift; 1.0 means the pair appears together no more often than chance.
    """
    share_with_subject = len(c.documents) / len(subject_docs) # Share of the subject's documents that state this value next to it
    share_overall = len(value_docs) / n_all # Share of all documents that mention the value at all
    return share_with_subject / share_overall # Lift = how many times more often than chance


def other_values(c: Candidate, everything: list[Candidate]) -> str:
    """List other values of the same type that are often stated next to the same subject.

    Args:
        c: The candidate.
        everything: All kept candidates.

    Returns:
        e.g. "$58 billion (7 docs); $65 billion (5 docs)", or "" if there are none.
    """
    others = [o for o in everything
              if o.subject.lower() == c.subject.lower() and o.value != c.value
              and o.value_type == c.value_type] # Same subject and value type, different value
    others.sort(key=lambda o: -len(o.documents)) # Most documents first
    return "; ".join(f"{o.value} ({len(o.documents)} docs)" for o in others[:3]) # The top 3, e.g. $58 billion (7 docs)


def write_csv(kept: list[Candidate], top: int | None) -> None:
    """Sort the candidates and write them to data/fact_candidates.csv.

    Args:
        kept: Candidates from scan().
        top: Write only the first `top` rows (None for all).
    """
    kept.sort(key=lambda c: (-c.distinct_passages, -len(c.documents))) # Most non-identical passages first, then most documents
    rows = kept if top is None else kept[:top] # All rows, or the top N

    used_values = set() # Values already used by facts.json
    for fact in load_facts(): # Sweep over facts
        used_values.add(fact["true_value"]) # Remember its true value

    FACT_CANDIDATES.parent.mkdir(parents=True, exist_ok=True) # Make data/ if needed
    with open(FACT_CANDIDATES, "w", encoding="utf-8", newline="") as f: # Open the CSV for writing
        writer = csv.writer(f) # CSV writer
        writer.writerow(["rank", "subject", "value", "value_type", "documents",
                         "usable_documents", "passages", "distinct_passages", "lift",
                         "recent_share", "other_values",
                         "in_facts_json", "example_1", "example_2", "passage_ids"]) # Header row
        for rank, c in enumerate(rows, start=1): # Sweep over candidates, ranked from 1
            examples = list(c.examples.values())[:2] + ["", ""] # Two example sentences (blank if fewer)
            writer.writerow([
                rank, c.subject, c.value, c.value_type, len(c.documents),
                len(c.usable_documents), len(c.passages), c.distinct_passages, f"{c.lift:.1f}",
                f"{len(c.recent_documents) / len(c.documents):.2f}",
                other_values(c, kept),
                "yes" if c.value in used_values else "",
                examples[0], examples[1], " ".join(c.passage_ids),
            ]) # One CSV row per candidate
    print(f"wrote {len(rows)} candidates to {FACT_CANDIDATES}") # Report where


def main() -> None:
    """Scan the corpus and write the ranked candidate list."""
    parser = argparse.ArgumentParser() # Command-line options
    parser.add_argument("--min-passages", type=int, default=MINE_MIN_PASSAGES) # --min-passages N
    parser.add_argument("--min-lift", type=float, default=MINE_MIN_LIFT) # --min-lift X
    parser.add_argument("--top", type=int, default=None, help="only write the top N rows") # --top N
    args = parser.parse_args() # Read the options

    kept = scan(args.min_passages, args.min_lift) # Scan the corpus
    write_csv(kept, args.top) # Write the ranked CSV


if __name__ == "__main__":
    main() # Rank fact candidates
