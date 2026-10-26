"""Download the PDFs linked from research pages and pull out their text.

    python -m crag.pdfs

Run after scrape.py. For every page in baker_pages.jsonl, downloads each linked
Baker PDF into data/raw/pdfs/ and saves its text as one line of data/raw/baker_pdfs.jsonl,
in the same format as a page, so passages.py can treat both the same way.
Pages scraped before we saved PDF links have no "pdfs" field; those pages are
fetched again to find their links. One request per second. Safe to rerun.
"""

import json
import logging
import re
import time
from collections import Counter
from pathlib import Path

import requests
from bs4 import BeautifulSoup
from pypdf import PdfReader

from crag.config import PDF_DIR, RAW_PAGES, RAW_PDFS
from crag.schema import Document
from crag.scrape import DELAY, HEADERS, get_pdf_links

MAX_MB = 25 # Skip very large files (usually books or scanned images)

# Sections at the back that we do not want as passages.
END_HEADINGS = {"endnotes", "notes", "references", "bibliography", "works cited"}

# pypdf prints a warning for every small formatting problem in a PDF. Only show real errors.
logging.getLogger("pypdf").setLevel(logging.ERROR)


def pdf_links(page: Document) -> list[str]:
    """Get a page's PDF links, fetching the page again if they were not saved.

    Args:
        page: A scraped page.

    Returns:
        The page's Baker PDF URLs.
    """
    if "pdfs" in page: # If the scraper saved the page's PDF links
        return page["pdfs"] # Use them
    html = requests.get(page["url"], headers=HEADERS, timeout=30).text # Otherwise download the page again
    time.sleep(DELAY) # Wait between requests
    return get_pdf_links(BeautifulSoup(html, "html.parser")) # Pull the PDF links out of it


def download(url: str) -> Path | None:
    """Download one PDF into data/raw/pdfs/, unless it is already there.

    Args:
        url: The PDF URL.

    Returns:
        The saved file's path, or None if the download failed or the file was too big.
    """
    path = PDF_DIR / url.split("/")[-1].replace("%20", "_") # Local file name, e.g. data/raw/pdfs/report.pdf
    if path.exists(): # If already downloaded
        return path # Reuse it
    response = requests.get(url, headers=HEADERS, timeout=120) # Download the PDF
    time.sleep(DELAY) # Wait between requests
    if response.status_code != 200: # If it failed
        print(response.status_code, url) # Print the error code
        return None # Skip it
    if len(response.content) > MAX_MB * 1024 * 1024: # If bigger than MAX_MB
        print("too big, skipped", url) # Print it
        return None # Skip it
    path.write_bytes(response.content) # Save the file
    return path # Return its path


def pdf_lines(path: Path) -> list[list[str]]:
    """Extract the text of a PDF, one list of lines per PDF page.

    Args:
        path: The PDF file.

    Returns:
        For each PDF page, its text lines with surrounding spaces removed.
    """
    reader = PdfReader(path) # Open the PDF
    pages = [] # Lines of each PDF page
    for page in reader.pages: # Sweep over PDF pages
        text = page.extract_text() or "" # Its text (empty if pypdf finds none, e.g. a scanned image)
        pages.append([line.strip() for line in text.split("\n")]) # Split into lines, trim spaces
    return pages # Return lines per page


def clean_lines(pages: list[list[str]]) -> list[str]:
    """Remove headers, footers, page numbers, footnote numbers, and the back matter.

    Running headers and footers (e.g. the report title at the top of every page)
    repeat on many pages, so any line that shows up on more than a third of the
    pages is dropped. Reading stops at a heading like "Endnotes" or "References"
    after the first two pages.

    Args:
        pages: Output of pdf_lines().

    Returns:
        The remaining lines of the whole PDF, in order.
    """
    counts: Counter[str] = Counter() # How many PDF pages each line appears on
    for lines in pages: # Sweep over PDF pages
        counts.update(set(lines)) # Count each distinct line once per page
    repeated = set() # Lines to drop as headers/footers
    if len(pages) >= 3: # Only with 3+ pages (otherwise every line looks repeated)
        repeated = {line for line, n in counts.items() if n > len(pages) / 3} # Lines on more than a third of the pages

    kept: list[str] = [] # Lines kept
    for page_number, lines in enumerate(pages): # Sweep over PDF pages
        for line in lines: # Sweep over their lines
            if not line or line in repeated: # If blank or a header/footer
                continue # Skip it
            if line.isdigit(): # If only a number (a page number)
                continue # Skip it
            if page_number > 1 and line.lower().rstrip(":") in END_HEADINGS: # If a back-matter heading (Endnotes, References, ...) after the first 2 pages
                return kept # Stop reading here
            # A footnote number at the start of a line, e.g. "1 as it prevents ...".
            line = re.sub(r"^\d{1,3} (?=[a-z])", "", line) # Remove the footnote number
            kept.append(line) # Keep the line
    return kept # Return the kept lines


def to_sentences(lines: list[str]) -> list[str]:
    """Join the lines back into running text and split it into sentences.

    passages.chunk() later groups the sentences into ~120-word passages.

    Args:
        lines: Output of clean_lines().

    Returns:
        The sentences, in order.
    """
    text = "\n".join(lines) # Back into one text, line breaks kept for now
    # pypdf sometimes returns half of a math symbol, which cannot be saved as UTF-8. Drop those.
    text = text.encode("utf-8", errors="ignore").decode("utf-8") # Drop half-symbols that cannot be saved as UTF-8
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)  # words split across lines: "petro-\nleum" # Re-join words split across lines (petro-/leum -> petroleum)
    text = re.sub(r"\s+", " ", text) # All whitespace (incl. line breaks) -> single spaces
    return re.split(r"(?<=[.!?])\s+(?=[A-Z\"“])", text) # Split after . ! ? when the next word starts with a capital or a quote


def main() -> None:
    """Download and extract every linked PDF not already in baker_pdfs.jsonl."""
    PDF_DIR.mkdir(parents=True, exist_ok=True) # Make data/raw/pdfs/ if needed
    done = set() # PDF URLs already done
    if RAW_PDFS.exists(): # If an earlier run saved PDFs
        with open(RAW_PDFS, encoding="utf-8") as f: # Open them
            for line in f: # Sweep over saved PDFs
                done.add(json.loads(line)["url"]) # Remember the URL

    with open(RAW_PAGES, encoding="utf-8") as f: # Open the scraped pages
        pages: list[Document] = [json.loads(line) for line in f] # Parse one page per line
    print(f"{len(pages)} pages, {len(done)} PDFs already done") # Progress

    with open(RAW_PDFS, "a", encoding="utf-8") as out: # Append to baker_pdfs.jsonl
        for i, page in enumerate(pages, start=1): # Sweep over pages
            if i % 100 == 0: # Every 100 pages
                print(f"{i}/{len(pages)} pages checked") # Print progress
            try:
                links = pdf_links(page) # The page's PDF links
            except requests.RequestException as error: # If fetching the page failed
                print("error", page["url"], error) # Print the error
                continue # Skip the page
            for url in links: # Sweep over its PDF links
                if url in done: # If this PDF is done (several pages can link one PDF)
                    continue # Skip it
                done.add(url) # Mark it done
                try:
                    path = download(url) # Download it
                    if path is None: # If the download failed or was too big
                        continue # Skip it
                    sentences = to_sentences(clean_lines(pdf_lines(path))) # PDF -> lines -> cleaned lines -> sentences
                except Exception as error: # Broken or unusual PDFs: skip them
                    print("error", url, error) # Print the error
                    continue # Skip it
                record: Document = {
                    "url": url,
                    "page_url": page["url"],
                    "title": page["title"],
                    "date": page["date"],
                    "source": "pdf",
                    "paragraphs": sentences,
                } # Same shape as a page record, source = pdf
                out.write(json.dumps(record, ensure_ascii=False) + "\n") # Save it as one JSON line
                out.flush() # Write it to disk now


if __name__ == "__main__":
    main() # Download and extract every linked PDF not done yet
