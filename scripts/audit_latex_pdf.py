"""Render the compiled thesis for visual review and record page boundaries."""
from pathlib import Path
import json
import pdfplumber
import subprocess
from PIL import Image, ImageDraw

root = Path(__file__).resolve().parents[1]
source = root / 'docs/tcc/overleaf_eeg_xai/main.pdf'
output = root / 'tmp/tcc_visual_audit'
output.mkdir(parents=True, exist_ok=True)
document = pdfplumber.open(source).pages
subprocess.run(['pdftoppm', '-scale-to', '1400', '-png', str(source), str(output / 'page')], check=True)
records = []
for index, page in enumerate(document):
    text = (page.extract_text() or '').splitlines()
    records.append({'page': index+1, 'first': text[:4], 'last': text[-5:], 'chars': len('\n'.join(text))})
for start in range(0, len(document), 8):
    sheet = Image.new('RGB', (1200, 1800), 'gray')
    draw = ImageDraw.Draw(sheet)
    for offset in range(min(8, len(document)-start)):
        page_image = Image.open(output / f'page-{start+offset+1:02}.png')
        page_image.thumbnail((290, 850))
        x, y = (offset % 4)*300, (offset // 4)*900
        sheet.paste(page_image, (x, y+25))
        draw.text((x+5, y+5), str(start+offset+1), fill='white')
    sheet.save(output / f'sheet-{start+1:02}.png')
(output / 'pages.json').write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding='utf-8')
print(json.dumps(records, ensure_ascii=False))
