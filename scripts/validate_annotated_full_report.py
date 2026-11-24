#!/usr/bin/env python3
"""Check report numbers, CSV, evidence links and every rendered PDF page."""
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import sys

from PIL import Image, ImageDraw
from pypdf import PdfReader
import pypdfium2 as pdfium

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from amodal_benchmark.dataset import sha256, write_json
from review_annotated_full import OUTPUT, read, require
from build_annotated_full_report import ORDER, interval


def main():
    audit = read(OUTPUT/'audit.json')
    build = read(OUTPUT/'report-build.json')
    pairs = read(OUTPUT/'paired-records.json')
    require(audit['passed'] and build['audit_sha256'] == sha256(OUTPUT/'audit.json'), 'Stale audit')
    require(build['source_sha256'] == sha256(ROOT/'scripts/build_annotated_full_report.py'), 'Stale builder')
    for relative, expected in build['evidence'].items():
        require(sha256(OUTPUT/relative) == expected, 'Evidence differs: '+relative)
    markdown = (OUTPUT/'REPORT.md').read_text(encoding='utf-8')
    require(build['report_sha256'] == sha256(OUTPUT/'REPORT.md'), 'Report text changed')
    require(build['pdf_sha256'] == sha256(OUTPUT/'REPORT.pdf'), 'PDF changed')
    links = re.findall(r'\]\(([^)]+)\)',markdown)
    for relative in links:
        require((OUTPUT/relative).is_file(), 'Missing report link: '+relative)
    with (OUTPUT/'paired-records.csv').open(newline='',encoding='utf-8') as stream:
        rows = list(csv.DictReader(stream))
    require(len(rows) == build['csv_rows'] == audit['targets'] == 2565, 'CSV target count differs')
    checks = 0
    for row, pair in zip(rows,pairs):
        require(row['id'] == pair['id'] and row['transition'] == pair['transition'], 'CSV identity differs')
        for side in ('baseline','surface'):
            for metric in ORDER:
                value = row[side+'_'+metric]
                expected = pair[side][metric]
                require(value == '' if expected is None else float(value) == expected, 'CSV metric differs')
                checks += 1
        require(float(row['absolute_lpips_gain']) == pair['baseline']['LPIPS']-pair['surface']['LPIPS'],
                'CSV LPIPS difference differs')
    pdf = PdfReader(OUTPUT/'REPORT.pdf')
    texts = [page.extract_text() for page in pdf.pages]
    text = '\n'.join(texts)
    require(all(len(page.strip()) > 80 for page in texts), 'Empty PDF page')
    for key in ('metrics','unseen_metrics'):
        for metric in ORDER:
            row = audit[key][metric]
            values = [f'{row["baseline"]:.6f}', f'{row["method"]:.6f}',
                      f'{row["relative_gain_percent"]:+.3f}%', interval(row['relative_ci_percent'])]
            for value in values:
                require(value in markdown, 'Missing Markdown metric '+value)
                require(value.replace(' ','') in text.replace(' ','').replace('\n',''), 'Missing PDF metric '+value)
    require('inconclusive' in texts[0] and '3%' in texts[0] and '1.191%' in texts[0], 'Missing first-page conclusion')
    require('209 to 199' in text, 'Cap-count direction missing')
    require('4.044%' in text and 'Conditional LPIPS' in text, 'Missing shared-success scope')
    review = OUTPUT/'pdf-review'
    review.mkdir(exist_ok=True)
    rendered = pdfium.PdfDocument(str(OUTPUT/'REPORT.pdf'))
    tiles, char_checks = [], 0
    for index in range(len(rendered)):
        page = rendered[index]
        textpage = page.get_textpage()
        width, height = page.get_size()
        for char in range(textpage.count_chars()):
            left,bottom,right,top = textpage.get_charbox(char)
            require(left >= -1 and right <= width+1 and bottom >= -1 and top <= height+1,
                    f'PDF character outside page {index+1}')
            char_checks += 1
        image = page.render(scale=1.5).to_pil().convert('RGB')
        image.save(review/f'page-{index+1:02d}.png')
        image.thumbnail((300,430))
        tile = Image.new('RGB',(320,460),'#E8EDF1')
        tile.paste(image,((320-image.width)//2,6))
        ImageDraw.Draw(tile).text((12,440),f'Page {index+1}',fill='black')
        tiles.append(tile)
        textpage.close()
        page.close()
    rendered.close()
    sheet = Image.new('RGB',(960,((len(tiles)+2)//3)*460),'white')
    for index,tile in enumerate(tiles):
        sheet.paste(tile,((index%3)*320,(index//3)*460))
    sheet.save(review/'contact-sheet.png')
    result = dict(passed=True, at=datetime.now(timezone.utc).isoformat(), targets=len(rows),
        csv_metric_checks=checks, report_links=len(links), pdf_pages=len(texts),
        pdf_character_bounds_checks=char_checks, all_metric_tables_match_audit=True,
        report_sha256=sha256(OUTPUT/'REPORT.md'), pdf_sha256=sha256(OUTPUT/'REPORT.pdf'),
        source_sha256=sha256(Path(__file__)), visual_review='Separate; see visual-review.json')
    write_json(OUTPUT/'validation.json',result)
    print(json.dumps(result,indent=2))


if __name__ == '__main__':
    main()
