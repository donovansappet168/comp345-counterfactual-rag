"""Make report/reference.docx, the Word style template build.py hands to pandoc.

Starts from pandoc's default template and changes it to a formal report style:
Times New Roman everywhere, 12pt justified body text, black bold headings, 10pt
tables, US Letter with 1-inch margins, and a page number centered in the footer.

Run from the repo root (only needed when the style changes):
    python report/make_reference.py
"""

import io
import re
import subprocess
import zipfile
from pathlib import Path

OUT = Path(__file__).resolve().parent / "reference.docx" # Template that build.py uses
FONT = "Times New Roman"
FONTS = f'<w:rFonts w:ascii="{FONT}" w:hAnsi="{FONT}" w:eastAsia="{FONT}" w:cs="{FONT}" />' # Same font for every script
# Style id -> (size in half-points, bold, italic, centered); 24 half-points = 12pt
HEADINGS = {"Title": (32, True, False, True), "Subtitle": (24, False, False, True),
            "Author": (24, False, False, True), "Date": (24, False, False, True),
            "Heading1": (28, True, False, False), "Heading2": (24, True, False, False),
            "Heading3": (24, True, True, False)}
FOOTER = ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
          '<w:ftr xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
          '<w:p><w:pPr><w:jc w:val="center" /></w:pPr>'
          '<w:r><w:fldChar w:fldCharType="begin" /></w:r>'
          '<w:r><w:instrText xml:space="preserve"> PAGE </w:instrText></w:r>'
          '<w:r><w:fldChar w:fldCharType="separate" /></w:r><w:r><w:t>1</w:t></w:r>'
          '<w:r><w:fldChar w:fldCharType="end" /></w:r></w:p></w:ftr>') # Centered page number field
# Letter paper, 1-inch margins (1440 twips = 1 inch), and the footer above
SECTION = ('<w:sectPr><w:footerReference w:type="default" r:id="rIdFooter1" />'
           '<w:pgSz w:w="12240" w:h="15840" />'
           '<w:pgMar w:top="1440" w:right="1440" w:bottom="1440" w:left="1440" '
           'w:header="720" w:footer="720" w:gutter="0" />')


def set_style(styles: str, style_id: str, size: int, bold: bool, italic: bool, centered: bool) -> str:
    """Replace one paragraph style's run formatting with black Times New Roman at the given size.

    Args:
        styles: The text of word/styles.xml.
        style_id: The style to change, e.g. "Heading1".
        size: Font size in half-points (28 = 14pt).
        bold: Make the style bold.
        italic: Make the style italic.
        centered: Center the paragraph.

    Returns:
        styles.xml with that style changed.
    """
    pattern = r'(<w:style [^>]*w:styleId="' + style_id + r'".*?)(<w:rPr>.*?</w:rPr>\s*)?(</w:style>)'
    run = FONTS # Font
    # Say "off" explicitly, or Subtitle, Author, and Date inherit bold from Title
    run += "<w:b />" if bold else '<w:b w:val="0" />'
    run += "<w:i />" if italic else '<w:i w:val="0" />'
    run += f'<w:color w:val="000000" /><w:sz w:val="{size}" /><w:szCs w:val="{size}" />' # Black, sized
    new_rpr = f"<w:rPr>{run}</w:rPr>"
    styles = re.sub(pattern, lambda m: m.group(1) + new_rpr + m.group(3), styles, count=1, flags=re.S)
    if centered: # Title block lines are centered already (they are based on Title)
        return styles
    return re.sub(r'(<w:style [^>]*w:styleId="' + style_id + r'".*?)<w:jc w:val="center" />',
                  r"\1", styles, count=1, flags=re.S) # Headings stay left-aligned


def fix_styles(styles: str) -> str:
    """Apply the formal report style to pandoc's default styles.xml.

    Args:
        styles: The text of word/styles.xml.

    Returns:
        The changed styles.xml.
    """
    styles = re.sub(r"<w:rFonts [^>]*/>", FONTS, styles, flags=re.S) # Times New Roman everywhere, no theme fonts
    for style_id, (size, bold, italic, centered) in HEADINGS.items(): # Title block and headings
        styles = set_style(styles, style_id, size, bold, italic, centered)
    # Body text: no space before, 6pt after, justified
    styles = styles.replace('<w:spacing w:before="180" w:after="180" />',
                            '<w:spacing w:before="0" w:after="120" /><w:jc w:val="both" />', 1)
    styles = styles.replace('<w:spacing w:before="36" w:after="36" />',
                            '<w:spacing w:before="0" w:after="40" /><w:jc w:val="left" />', 1) # Lists and table cells
    # Captions: 10pt italic
    styles = styles.replace("<w:rPr>\n      <w:i />\n    </w:rPr>",
                            '<w:rPr><w:i /><w:sz w:val="20" /><w:szCs w:val="20" /></w:rPr>', 1)
    # Tables: 10pt text (the table style's own rPr, added before its tblPr)
    styles = re.sub(r'(<w:style w:type="table" w:default="1" w:styleId="Table">.*?)(<w:tblPr>)',
                    r'\1<w:rPr><w:sz w:val="20" /><w:szCs w:val="20" /></w:rPr>\2', styles, count=1, flags=re.S)
    return styles


def main() -> None:
    """Write report/reference.docx from pandoc's default template plus the formal style."""
    default = subprocess.run(["pandoc", "--print-default-data-file", "reference.docx"],
                             check=True, capture_output=True).stdout # pandoc's own template, as bytes
    source = zipfile.ZipFile(io.BytesIO(default))
    with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED) as target: # Copy every part, changing four of them
        for name in source.namelist(): # Sweep over the parts of the .docx
            if not name.endswith((".xml", ".rels")): # Images and other binary parts are copied as they are
                target.writestr(name, source.read(name))
                continue
            text = source.read(name).decode("utf-8") # XML parts are text
            if name == "word/styles.xml":
                text = fix_styles(text)
            elif name == "word/document.xml": # Page setup and footer go in the section properties
                text = re.sub(r"<w:sectPr>", SECTION, text, count=1)
            elif name == "word/_rels/document.xml.rels": # Point rIdFooter1 at the footer part
                text = text.replace("</Relationships>",
                                    '<Relationship Type="http://schemas.openxmlformats.org/officeDocument/2006/'
                                    'relationships/footer" Id="rIdFooter1" Target="footer1.xml" /></Relationships>')
            elif name == "[Content_Types].xml": # Register the footer part
                text = text.replace("</Types>",
                                    '<Override PartName="/word/footer1.xml" ContentType="application/vnd.'
                                    'openxmlformats-officedocument.wordprocessingml.footer+xml" /></Types>')
            target.writestr(name, text)
        target.writestr("word/footer1.xml", FOOTER) # The new footer part
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
