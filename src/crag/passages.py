"""Passages and retrieval.

    python -m crag.passages

reads data/raw/baker_pages.jsonl (and baker_pdfs.jsonl if it exists), splits every
document into passages of at least CHUNK_WORDS words (whole paragraphs only, so most
passages run a little longer; see chunk()), embeds them with all-MiniLM-L6-v2,
and saves
    data/raw/passages.jsonl          one passage per line
    data/raw/passage_embeddings.npy  one 384-d unit vector per passage (same order)
"""

import json
import re

import numpy as np
from numpy.typing import NDArray
from sentence_transformers import SentenceTransformer

from crag.config import CHUNK_WORDS, INDEX, PASSAGES, RAW_PAGES, RAW_PDFS, RETRIEVER
from crag.schema import Document, Passage

retriever: SentenceTransformer | None = None


def get_retriever() -> SentenceTransformer:
    """Load MiniLM the first time it is needed and reuse it after that.

    Returns:
        The sentence-embedding model.
    """
    global retriever # Use the module-level retriever (shared across calls)
    if retriever is None: # If MiniLM is not loaded yet
        retriever = SentenceTransformer(RETRIEVER) # Load it (downloads the first time)
    return retriever # Return the loaded model


def embed(texts: list[str]) -> NDArray[np.float32]:
    """Embed texts with MiniLM.

    The vectors are normalized to length 1, so the dot product of two vectors
    is their cosine similarity.

    Args:
        texts: The texts to embed.

    Returns:
        Array of shape (len(texts), 384).
    """
    vectors: NDArray[np.float32] = get_retriever().encode(texts, normalize_embeddings=True) # One 384-number unit vector per text
    return vectors # Return the vectors


def chunk(paragraphs: list[str]) -> list[str]:
    """Group paragraphs into passages of roughly CHUNK_WORDS words.

    Whole paragraphs are added to the current passage until it has at least
    CHUNK_WORDS words, then a new passage starts. Leftovers under 20 words
    (a caption, a one-line sign-off) are dropped.

    Why passage length is approximate: we never split inside a paragraph, so that
    a sentence (and the fact in it) is never cut in half. CHUNK_WORDS is therefore
    a minimum, not an exact length. A passage ends at the first paragraph break
    after it reaches CHUNK_WORDS, so it overshoots by part of one paragraph. The
    last passage of a document can be shorter (20 words or more). With
    CHUNK_WORDS = 120, the full corpus built on 2026-10-26 (2,047 pages and 747
    PDFs, 50,852 passages) had a median of 139 words, 90% of passages under 199
    words, and a few very long ones (up to 4,839 words) where a single paragraph,
    or PDF text with no sentence breaks, was already long.

    Known limitation: MiniLM reads only the first 256 tokens (about 200 words) of
    a passage, so in the ~10% of passages over 200 words the retriever does not
    see the end. The fact passages we pick by hand are checked to be shorter.

    Args:
        paragraphs: The document's paragraphs in order (sentences, for PDFs).

    Returns:
        The passage texts.
    """
    passages = [] # Finished passages
    current = "" # Passage being built
    for paragraph in paragraphs: # Sweep over paragraphs (sentences for PDFs)
        current = (current + " " + paragraph).strip() # Add the whole paragraph to the current passage
        if len(current.split()) >= CHUNK_WORDS: # If it has reached CHUNK_WORDS words
            passages.append(current) # Close it
            current = "" # Start a new one
    if current: # If a leftover passage is being built
        passages.append(current) # Keep it (can be shorter than CHUNK_WORDS)
    return [p for p in passages if len(p.split()) >= 20] # Drop tiny passages (< 20 words), e.g. captions


def read_documents() -> list[Document]:
    """Read the scraped web pages, plus the PDF texts if pdfs.py has been run.

    Returns:
        All documents, pages first.
    """
    documents: list[Document] = [] # Every document: pages, then PDFs
    with open(RAW_PAGES, encoding="utf-8") as f: # Open the scraped pages
        for line in f: # Sweep over lines (one page per line)
            documents.append(json.loads(line)) # Parse and keep the page
    if RAW_PDFS.exists(): # If pdfs.py has been run
        with open(RAW_PDFS, encoding="utf-8") as f: # Open the PDF texts
            for line in f: # Sweep over lines (one PDF per line)
                documents.append(json.loads(line)) # Parse and keep the PDF
    return documents # Return all documents


def build() -> None:
    """Split every document into passages, embed them, and save both to data/raw/."""
    passages: list[Passage] = [] # Every passage in the corpus
    for doc in read_documents(): # Sweep over documents
        slug = doc["url"].split("/")[-1] # Last part of the URL, used in passage ids
        for i, text in enumerate(chunk(doc["paragraphs"])): # Sweep over the document's passages
            passages.append({
                "id": f"{slug}#{i}",
                "url": doc["url"],
                "page_url": doc.get("page_url", doc["url"]), # For a PDF, the page that links it
                "title": doc["title"],
                "date": doc["date"],
                "source": doc.get("source", "page"),
                "text": text,
            }) # Add the passage record

    with open(PASSAGES, "w", encoding="utf-8") as f: # Open passages.jsonl for writing
        for p in passages: # Sweep over passages
            f.write(json.dumps(p, ensure_ascii=False) + "\n") # One passage per line, keeping accents as-is

    texts = [p["text"] for p in passages] # Just the texts, in the same order
    vectors = get_retriever().encode(texts, batch_size=64, normalize_embeddings=True,
                                     show_progress_bar=True) # Embed every passage, 64 at a time (~25 min on CPU for the full corpus)
    np.save(INDEX, vectors.astype(np.float32)) # Save the vectors in the same order as passages.jsonl
    print(f"saved {len(passages)} passages") # Report the count


def load() -> tuple[list[Passage], NDArray[np.float32]]:
    """Load the saved passages and their embeddings.

    Returns:
        The passages and an array of shape (number of passages, 384) in the same order.
    """
    with open(PASSAGES, encoding="utf-8") as f: # Open passages.jsonl
        passages: list[Passage] = [json.loads(line) for line in f] # Parse one passage per line
    vectors: NDArray[np.float32] = np.load(INDEX) # Load the matching vectors
    return passages, vectors # Return both


def with_score(passage: Passage, score: float) -> Passage:
    """Copy a passage and attach its similarity score.

    Args:
        passage: The passage.
        score: Its cosine similarity to the question.

    Returns:
        A copy of the passage with "score" set.
    """
    scored = passage.copy() # Copy, so the stored passage is not changed
    scored["score"] = float(score) # Attach the similarity
    return scored # Return the copy


def search(question: str, passages: list[Passage], vectors: NDArray[np.float32],
           k: int = 5) -> list[Passage]:
    """Find the k passages in the whole corpus most similar to the question.

    Args:
        question: The question text.
        passages: All passages, from load().
        vectors: Their embeddings, from load().
        k: How many passages to return.

    Returns:
        The top k passages, most similar first, each with a "score".
    """
    scores = vectors @ embed([question])[0] # Cosine similarity of the question to every passage (vectors are unit length)
    best = np.argsort(-scores)[:k] # Indexes of the k highest scores, highest first
    return [with_score(passages[i], scores[i]) for i in best] # Return those passages with their scores


def rank(question: str, passages: list[Passage]) -> list[Passage]:
    """Sort a given list of passages by similarity to the question.

    Used to order a fact's own 5 passages into slots 1 to 5.

    Args:
        question: The question text.
        passages: The passages to sort.

    Returns:
        The same passages, most similar first, each with a "score".
    """
    q = embed([question])[0] # Question vector
    p = embed([x["text"] for x in passages]) # One vector per given passage
    scores = p @ q # Cosine similarity of each passage to the question
    order = np.argsort(-scores) # Indexes sorted from most to least similar
    return [with_score(passages[i], scores[i]) for i in order] # Return the passages in that order, with scores


def grep(pattern: str, passages: list[Passage]) -> list[Passage]:
    """Find passages whose text matches a regex (case-insensitive).

    Args:
        pattern: The regex, e.g. r"NAFTA.{0,80}1994".
        passages: The passages to search.

    Returns:
        The matching passages.
    """
    return [p for p in passages if re.search(pattern, p["text"], re.IGNORECASE)] # Keep passages where the regex appears (any capitalization)


if __name__ == "__main__":
    build() # Build passages.jsonl + passage_embeddings.npy
