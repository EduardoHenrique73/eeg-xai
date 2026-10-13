"""Finalize the browser-rendered PDF and map its headings back to the DOCX TOC."""

from __future__ import annotations

import io
import json
import re
import unicodedata
from pathlib import Path

from docx import Document
from pypdf import PdfReader, PdfWriter
from reportlab.pdfgen import canvas


ROOT = Path(__file__).resolve().parents[1]
PREVIEW = ROOT / "docs/tcc/render_revisao/TCC II - preview sem Word.pdf"
DOCX = ROOT / "docs/tcc/TCC II - Eduardo Marques - Versao Final Revisada.docx"
MAPPING = ROOT / "docs/tcc/render_revisao/toc_pages.json"
OUTPUT = ROOT / "docs/tcc/TCC II - Eduardo Marques - Versao Final Revisada.pdf"


def normalized(value: str) -> str:
    value = unicodedata.normalize("NFKD", value)
    value = "".join(char for char in value if not unicodedata.combining(char))
    return re.sub(r"\s+", " ", value).casefold().strip()


def numbered_page(page, number: int):
    width, height = float(page.mediabox.width), float(page.mediabox.height)
    memory = io.BytesIO()
    overlay = canvas.Canvas(memory, pagesize=(width, height))
    overlay.setFont("Helvetica", 10)
    overlay.drawRightString(width - 56, height - 45, str(number))
    overlay.save()
    memory.seek(0)
    page.merge_page(PdfReader(memory).pages[0])
    return page


def main():
    source = PdfReader(PREVIEW)
    pages = [page for page in source.pages if len(page.extract_text().strip()) > 1]
    texts = [normalized(page.extract_text()) for page in pages]
    headings = [p.text.strip() for p in Document(DOCX).paragraphs if p.style.name.startswith("Heading") and p.text.strip()]
    mapping = {}
    missing = []
    for heading in headings:
        needle = normalized(heading)
        if not needle:
            continue
        if heading == "REFERÊNCIAS":
            match = next((index + 1 for index, value in enumerate(texts[3:], start=3) if value.startswith(needle)), None)
        else:
            match = next((index + 1 for index, value in enumerate(texts[3:], start=3) if needle in value), None)
        if match is None:
            missing.append(heading)
        else:
            mapping[heading] = match
    if missing:
        raise RuntimeError(f"Headings absent from PDF: {missing}")
    MAPPING.write_text(json.dumps(mapping, ensure_ascii=False, indent=2), encoding="utf-8")

    writer = PdfWriter()
    for number, page in enumerate(pages, start=1):
        if number >= 4:
            numbered_page(page, number)
        writer.add_page(page)
    with OUTPUT.open("wb") as stream:
        writer.write(stream)
    print(json.dumps({"pdf": str(OUTPUT), "pages": len(pages), "headings": len(mapping)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
