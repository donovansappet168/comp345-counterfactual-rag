"""Download Baker Institute research pages.

    python -m crag.scrape

Gets every /research/<slug> URL from the public sitemap and saves each page's title,
date, authors, tags, body paragraphs, and PDF links as one line of
data/raw/baker_pages.jsonl. robots.txt disallows /research-library, so we never fetch it.
One request per second. If the run stops, start it again: saved pages are skipped.
"""

import json
import re
import time

import requests
from bs4 import BeautifulSoup, Tag

from crag.config import RAW_PAGES
from crag.schema import Document

SITEMAP_PAGES = [f"https://www.bakerinstitute.org/sitemap.xml?page={i}" for i in (1, 2, 3)]
HEADERS = {"User-Agent": "Mozilla/5.0 (COMP 345 course project, Rice University)"}
DELAY = 1.0 # Seconds between requests
DATE = re.compile(r"(January|February|March|April|May|June|July|August|September|"
                  r"October|November|December) \d{1,2}, \d{4}")


def research_urls() -> list[str]:
    """Collect every /research/ URL listed in the site's sitemap.

    Returns:
        The URLs, sorted alphabetically.
    """
    urls = set() # URLs found (a set, so no duplicates)
    for sitemap in SITEMAP_PAGES: # Sweep over the 3 sitemap pages
        xml = requests.get(sitemap, headers=HEADERS, timeout=60).text # Download the sitemap XML
        for url in re.findall(r"<loc>(.*?)</loc>", xml): # Sweep over every <loc> URL in it
            if "/research/" in url: # If it is a publication page
                urls.add(url) # Keep it
        time.sleep(DELAY) # Wait between requests (be polite to the server)
    return sorted(urls) # Return the URLs in alphabetical order


def get_title(soup: BeautifulSoup) -> str:
    """Read the page title from its og:title meta tag.

    Args:
        soup: The parsed page.

    Returns:
        The title without the " | Baker Institute" suffix, or "" if missing.
    """
    tag = soup.find("meta", property="og:title") # The og:title meta tag
    if not isinstance(tag, Tag): # If it is missing
        return "" # No title
    return str(tag["content"]).replace(" | Baker Institute", "") # Its content, without the site name


def get_date(soup: BeautifulSoup) -> str:
    """Find the publication date.

    The date is the first <span> whose text looks like "September 25, 2025".

    Args:
        soup: The parsed page.

    Returns:
        The date as written on the page, or "" if none was found.
    """
    for span in soup.find_all("span"): # Sweep over every <span> on the page
        match = DATE.search(span.get_text(strip=True)) # Look for a date like 'September 25, 2025'
        if match: # If found
            return match.group(0) # Return the first date
    return "" # No date found


def get_authors(soup: BeautifulSoup) -> list[str]:
    """Read the author names from the page's JSON-LD metadata block.

    Args:
        soup: The parsed page.

    Returns:
        The author names (empty if the page lists none).
    """
    authors = [] # Author names found
    for script in soup.find_all("script", type="application/ld+json"): # Sweep over JSON-LD metadata blocks
        try:
            graph = json.loads(script.string)["@graph"] # Parse the block, take its list of entries
        except (TypeError, KeyError, ValueError): # If it is empty or not the usual shape
            continue # Skip it
        for node in graph: # Sweep over entries
            author = node.get("author") # Its author: one dict, a list of dicts, or None
            if isinstance(author, dict): # If a single author
                author = [author] # Make it a list of one
            for a in author or []: # Sweep over authors (none if None)
                authors.append(a.get("name", "")) # Keep the name
    return authors # Return the names


def get_tags(soup: BeautifulSoup) -> list[str]:
    """Read the topic tags, which link to the research library search.

    Args:
        soup: The parsed page.

    Returns:
        The tag names, sorted, e.g. ["Mexico", "Trade"].
    """
    tags = set() # Tags found (a set, so no duplicates)
    for link in soup.find_all("a", href=True): # Sweep over every link on the page
        if str(link["href"]).startswith("/research-library?search="): # If it links to a research-library search (that is what a tag is)
            tags.add(link.get_text(strip=True)) # Keep the link text
    return sorted(tags) # Return tags in alphabetical order


def get_paragraphs(soup: BeautifulSoup) -> list[str]:
    """Read the body text of the publication.

    Args:
        soup: The parsed page.

    Returns:
        The paragraphs, list items, and headings of the body, in order,
        without the copyright line at the bottom of every page.
    """
    body = soup.select_one(".js-research-body") # The publication body
    if body is None: # If the page has no body
        return [] # No paragraphs
    paragraphs = [] # Paragraphs found
    for element in body.find_all(["p", "li", "h2", "h3"]): # Sweep over paragraphs, list items, and headings, in order
        text = element.get_text(" ", strip=True) # Their text, with spaces between inline pieces
        if text and not text.startswith("This material may be quoted or reproduced"): # If not empty and not the copyright line
            paragraphs.append(text) # Keep it
    return paragraphs # Return the paragraphs


def get_pdf_links(soup: BeautifulSoup) -> list[str]:
    """Find links to Baker-hosted PDFs (full reports are often attached as PDFs).

    Args:
        soup: The parsed page.

    Returns:
        Absolute PDF URLs on bakerinstitute.org, sorted.
    """
    links = set() # PDF links found (a set, so no duplicates)
    for link in soup.find_all("a", href=True): # Sweep over every link on the page
        href = str(link["href"]) # The link target
        if href.lower().endswith(".pdf"): # If it points to a PDF
            if href.startswith("/"): # If it is relative (e.g. /sites/...)
                href = "https://www.bakerinstitute.org" + href # Make it a full URL
            if href.startswith("https://www.bakerinstitute.org/"): # If it is on the Baker site
                links.add(href) # Keep it
    return sorted(links) # Return links in alphabetical order


def parse_page(url: str, html: str) -> Document:
    """Pull everything we keep out of one research page.

    Args:
        url: The page URL.
        html: The page's HTML.

    Returns:
        The page as a Document record.
    """
    soup = BeautifulSoup(html, "html.parser") # Parse the HTML
    return {
        "url": url,
        "title": get_title(soup),
        "date": get_date(soup),
        "authors": get_authors(soup),
        "tags": get_tags(soup),
        "paragraphs": get_paragraphs(soup),
        "pdfs": get_pdf_links(soup),
    } # Return the page record


def main() -> None:
    """Fetch every research page not already saved and append it to baker_pages.jsonl."""
    RAW_PAGES.parent.mkdir(parents=True, exist_ok=True) # Make data/raw/ if needed
    saved = set() # URLs already saved
    if RAW_PAGES.exists(): # If an earlier run saved pages
        with open(RAW_PAGES, encoding="utf-8") as f: # Open them
            for line in f: # Sweep over saved pages
                saved.add(json.loads(line)["url"]) # Remember the URL

    todo = [url for url in research_urls() if url not in saved] # Pages still to fetch
    print(f"{len(saved)} pages already saved, {len(todo)} to fetch") # Progress

    with open(RAW_PAGES, "a", encoding="utf-8") as out: # Append to baker_pages.jsonl
        for i, url in enumerate(todo, start=1): # Sweep over pages to fetch
            try:
                response = requests.get(url, headers=HEADERS, timeout=30) # Download the page
                if response.status_code == 200: # If it worked
                    page = parse_page(url, response.text) # Pull out title, date, text, ...
                    out.write(json.dumps(page, ensure_ascii=False) + "\n") # Save it as one JSON line
                    out.flush() # Write it to disk now
                else: # Otherwise
                    print(response.status_code, url) # Print the error code
            except requests.RequestException as error: # If the request failed (timeout, connection)
                print("error", url, error) # Print the error and keep going
            if i % 100 == 0: # Every 100 pages
                print(f"{i}/{len(todo)}") # Print progress
            time.sleep(DELAY) # Wait between requests


if __name__ == "__main__":
    main() # Scrape every research page not saved yet
