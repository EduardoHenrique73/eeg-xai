"""Render local PDF contact sheets for visual review without an office app."""

from pathlib import Path
import sys

import pypdfium2 as pdfium
from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parents[1]
PDF = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "docs/tcc/TCC II - Eduardo Marques - Versao Final Revisada.pdf"
OUT = Path(sys.argv[2]) if len(sys.argv) > 2 else ROOT / "docs/tcc/render_revisao"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    document = pdfium.PdfDocument(str(PDF))
    columns, rows = 4, 3
    scale = 0.45
    width, height = 310, 470
    for start in range(0, len(document), columns * rows):
        sheet = Image.new("RGB", (columns * width, rows * height), "#d9dce0")
        drawing = ImageDraw.Draw(sheet)
        for offset in range(min(columns * rows, len(document) - start)):
            number = start + offset
            page = document[number]
            image = page.render(scale=scale).to_pil().convert("RGB")
            image.thumbnail((width - 10, height - 24))
            x = (offset % columns) * width + (width - image.width) // 2
            y = (offset // columns) * height + 20
            sheet.paste(image, (x, y))
            drawing.text(((offset % columns) * width + 8, (offset // columns) * height + 3), f"{number + 1}", fill="black")
        output = OUT / f"contact-{start // (columns * rows) + 1}.png"
        sheet.save(output)
        print(output)


if __name__ == "__main__":
    main()
