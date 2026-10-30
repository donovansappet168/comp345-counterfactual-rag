"""Turn the midterm report (Obsidian Markdown in the vault) into a .docx and a LaTeX project.

The report is written in the vault, so this script copies it here, fixes the
Obsidian-only parts, and calls pandoc twice:
  1. report/midterm.docx           (upload to Google Drive, open as a Google Doc)
  2. report/latex/main.tex + figures/   (compile in VS Code or upload to Overleaf)

Run from the repo root:
    python report/build.py
"""

import re
import shutil
import subprocess
from pathlib import Path

VAULT = Path(r"C:\Users\donov\OneDrive\Documents\Obsidian\llm-wiki\raw\_inbox") # Where the report and figures live
REPORT = VAULT / "COMP345 Midterm Report Draft 2.md" # The report itself
OUT = Path(__file__).resolve().parent # report/ in the repo
LATEX = OUT / "latex" # LaTeX project folder (zip this for Overleaf)
TITLE = "How Much Incorrect Retrieved Context Does It Take to Flip an LLM's Answer?"
AUTHORS = "Zane Braunstein and Donovan Sappet"
SUBTITLE = "COMP 345 Midterm Report"
DATE = "October 30, 2026"


def clean(text: str) -> tuple[str, list[str]]:
    """Fix the Obsidian-only Markdown so pandoc can read the report.

    Drops the title lines and the draft note (pandoc adds its own title), turns
    each ![[figure.png]] plus its italic caption line into one captioned image,
    and escapes dollar signs in money ("$10,000") so pandoc does not read them as math.

    Args:
        text: The report's Markdown, as written in Obsidian.

    Returns:
        The cleaned Markdown and the figure file names it uses.
    """
    lines = text.splitlines() # Work line by line
    body = [] # Lines we keep
    for line in lines: # Sweep over lines
        if line.startswith("# ") or line.startswith("**COMP 345 Midterm Report**"): # Title lines - pandoc makes its own
            continue
        if line.startswith(">"): # The draft note (an Obsidian callout)
            continue
        body.append(line)
    text = "\n".join(body) # Back to one string
    figures = re.findall(r"!\[\[([^\]]+)\]\]", text) # Every embedded figure
    # Image line, blank line, italic caption line -> one image whose alt text is the caption
    # (blank line after, so two figures in a row stay two figures and not one paragraph)
    text = re.sub(r"!\[\[([^\]]+)\]\]\s*\n\s*\n\*(Figure [^\n]+?)\*\s*\n", r"![\2](figures/\1)\n\n", text)
    text = re.sub(r"\$(?=\d{1,3},\d{3}|\d{4,})", r"\\$", text) # "$10,000" is money, but "$2\ln 9$" is math
    return text.strip() + "\n", figures


def pandoc(source: Path, target: Path, extra: list[str]) -> None:
    """Run pandoc on the cleaned report.

    Args:
        source: The cleaned Markdown file.
        target: The output file; pandoc picks the format from its extension.
        extra: Extra pandoc options for this format.

    Raises:
        subprocess.CalledProcessError: If pandoc fails.
    """
    # --columns=50: every table counts as "wide", so its column widths follow the dashes under its header
    # --shift-heading-level-by=-1: the report's "## 1. Problem" becomes a top-level heading (the "#" title is dropped)
    command = ["pandoc", str(source), "-o", str(target), "--from", "markdown+tex_math_dollars", "--columns=50",
               "--shift-heading-level-by=-1",
               "--metadata", f"title={TITLE}", "--metadata", f"subtitle={SUBTITLE}",
               "--metadata", f"author={AUTHORS}", "--metadata", f"date={DATE}"] + extra
    subprocess.run(command, check=True, cwd=source.parent) # Run from the folder that holds figures/


def main() -> None:
    """Build report/midterm.docx and report/latex/main.tex from the vault copy of the report."""
    text, figures = clean(REPORT.read_text(encoding="utf-8")) # Read and fix the report
    (LATEX / "figures").mkdir(parents=True, exist_ok=True) # Figures sit next to main.tex
    for name in figures: # Copy every figure the report uses
        if (VAULT / name).exists(): # Figure 1 and 3 are still TODOs
            shutil.copy(VAULT / name, LATEX / "figures" / name)
    source = LATEX / "report.md" # Cleaned Markdown, kept so you can see what pandoc read
    source.write_text(text, encoding="utf-8")
    # reference.docx holds the Word styles (Times New Roman, formal layout); make_reference.py writes it
    pandoc(source, OUT / "midterm.docx", ["--reference-doc", str(OUT / "reference.docx")]) # Word file for Google Docs
    # Captions already say "Figure 2: ...", and Figures 1 and 3 are still TODOs, so turn off LaTeX's own numbering
    no_numbers = r"\usepackage{caption}\captionsetup[figure]{labelformat=empty}"
    pandoc(source, LATEX / "main.tex", ["--standalone", "-V", "geometry:margin=1in", "-V", "fontsize=12pt",
                                        "-V", "papersize=letter", "-V", "fontfamily=newtxtext,newtxmath", # Times
                                        "-V", "colorlinks=true", "-V", "linkcolor=black", "-V", "urlcolor=black",
                                        "-V", f"header-includes={no_numbers}"]) # LaTeX file
    print(f"wrote {OUT / 'midterm.docx'} and {LATEX / 'main.tex'}")


if __name__ == "__main__":
    main()
