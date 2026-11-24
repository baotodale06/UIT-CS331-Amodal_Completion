#!/usr/bin/env python3
"""Check report values, gallery coverage, native identities and exported PDF."""
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import sys
import zipfile

import numpy as np
from PIL import Image, ImageDraw
from pypdf import PdfReader
import pypdfium2 as pdfium

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from amodal_benchmark.dataset import sha256, write_json


def main():
    design = ROOT / 'output/surface-annotated-20261112-design'
    study = ROOT / json.loads((design/'active-study.json').read_text())['study']
    report = study/'report'
    audit = json.loads((report/'audit.json').read_text())
    content = json.loads((report/'report-content.json').read_text())
    pairs = json.loads((report/'paired_records.json').read_text())
    gallery = json.loads((report/'gallery-manifest.json').read_text())
    if not audit['passed'] or content['audit_sha256'] != sha256(report/'audit.json'):
        raise ValueError('Unverified or stale report')
    if content['builder_sha256'] != sha256(ROOT/'scripts/build_annotated_surface_report.py'):
        raise ValueError('Report builder changed after rendering')
    if content['methods_sha256'] != sha256(report/'METHODS.md') or content['methods_sha256'] != sha256(ROOT/'docs/SURFACE_ANNOTATED_METHODS_20261112.md'):
        raise ValueError('Mathematical methods appendix changed after rendering')
    if content['pool_audit_sha256'] is not None and content['pool_audit_sha256'] != sha256(report/'pool-audit.json'):
        raise ValueError('Stale pool diagnostic report')
    html = (report/'gallery.html').read_text(encoding='utf-8')
    for identifier, expected in (('direction', ['', 'gain', 'loss', 'tie']),
                                 ('transition', ['', 'shared', 'gained', 'lost', 'both_failed'])):
        select = re.search(r'<select id="' + identifier + r'"[^>]*>(.*?)</select>', html, re.S)
        if select is None or re.findall(r'<option value="([^"]*)">', select.group(1)) != expected:
            raise ValueError('Missing or malformed gallery filter options: ' + identifier)
        if re.search(r'</option\s+', select.group(1)):
            raise ValueError('Invalid gallery option closing tag')
    if html.count('<article ') != len(pairs) or gallery['cases'] != len(pairs) or len(pairs) != 204:
        raise ValueError('Gallery target coverage mismatch')
    for relative, row in gallery['panels'].items():
        if sha256(report/relative) != row['thumbnail_sha256'] or sha256(ROOT/row['source']) != row['source_sha256']:
            raise ValueError('Gallery source or displayed thumbnail differs')
        with Image.open(report/relative) as image:
            image.load()
            if image.size != (340,280):
                raise ValueError('Unexpected gallery geometry')
    examples = json.loads((report/'qualitative-examples.json').read_text())
    shared = [(row['id'], row['baseline']['LPIPS']-row['surface']['LPIPS'])
              for row in pairs if row['baseline_status'] == row['surface_status'] == 'success']
    expected_examples = sorted((row for row in shared if row[1] > 1e-12), key=lambda row: (-row[1],row[0]))[:2]
    expected_examples += sorted((row for row in shared if row[1] < -1e-12), key=lambda row: (row[1],row[0]))[:2]
    if [(row['id'],row['delta']) for row in examples['cases']] != expected_examples:
        raise ValueError('Qualitative examples do not follow the stated selection rule')
    for example in examples['cases']:
        for panel in example['panels']:
            if panel['sha256'] != gallery['panels'][panel['path']]['thumbnail_sha256']:
                raise ValueError('Qualitative panel differs from the complete gallery')
        for side, position in (('baseline', 2), ('surface', 3)):
            native = ROOT / gallery['panels'][example['panels'][position]['path']]['source']
            geometry = example['native_geometry'][side]
            with Image.open(native) as native_image:
                if [geometry['width'], geometry['height']] != list(native_image.size):
                    raise ValueError('Qualitative native dimensions differ')
                if geometry['nonzero_alpha_pixels'] != int(np.count_nonzero(np.array(native_image.convert('RGBA'))[..., 3])):
                    raise ValueError('Qualitative native foreground count differs')
            if geometry['native_sha256'] != sha256(native):
                raise ValueError('Qualitative native image identity differs')
            record = json.loads((native.parent / 'result.json').read_text())
            if geometry['offset_rc'] != record['geometry']['offset_rc']:
                raise ValueError('Qualitative native offset differs')
    for relative in re.findall(r'(?:src|href)="([^"]+)"', html):
        if not relative.startswith(('http:', 'https:', '#')) and not (report/relative).is_file():
            raise ValueError('Missing HTML file: '+relative)
    markdown = (report/'REPORT.md').read_text(encoding='utf-8')
    for relative in re.findall(r'\]\(([^)]+)\)', markdown):
        if not relative.startswith(('http:', 'https:', '#')) and not (report/relative).exists():
            raise ValueError('Missing Markdown file: '+relative)
    with (report/'paired_records.csv').open(encoding='utf-8',newline='') as stream:
        rows = list(csv.DictReader(stream))
    if [row['id'] for row in rows] != [row['id'] for row in pairs]:
        raise ValueError('CSV target inventory differs')
    for row, pair in zip(rows, pairs):
        for side in ('baseline','surface'):
            for metric, value in pair[side].items():
                actual = row[side+'_'+metric]
                if value is None:
                    if actual:
                        raise ValueError('CSV invents a missing metric')
                elif float(actual) != value:
                    raise ValueError('CSV metric differs')
        if not np.isclose(float(row['lpips_gain']), pair['baseline']['LPIPS']-pair['surface']['LPIPS'], atol=1e-15, rtol=0):
            raise ValueError('CSV paired gain differs')
    ranker = json.loads((design/'method-freeze.json').read_text())['ranker']
    with (report/'ranker-parameters.csv').open(encoding='utf-8',newline='') as stream:
        coefficients = list(csv.DictReader(stream))
    if len(coefficients) != 22 or [row['feature'] for row in coefficients] != ranker['feature_names']:
        raise ValueError('Parameter inventory differs')
    for key, field in (('coefficients','coefficient'),('medians','median_imputation'),('center','center'),('scale','scale')):
        if [float(row[field]) for row in coefficients] != ranker[key]:
            raise ValueError('Reported ranker parameters differ')
    pdf = PdfReader(report/'REPORT.pdf')
    texts = [page.extract_text() or '' for page in pdf.pages]
    if len(texts) < 5 or any(len(text.strip()) < 60 for text in texts):
        raise ValueError('Empty or incomplete PDF')
    if '204' not in texts[0] or '95%' not in texts[0]:
        raise ValueError('Missing primary scope or uncertainty in PDF')
    for metric_row in content['metric_rows']:
        for value in metric_row[1:5]:
            if str(value) not in texts[0]:
                raise ValueError('PDF metric value missing: '+str(value))
    # Render every page as a standalone visual review artifact.
    rendered = pdfium.PdfDocument(str(report/'REPORT.pdf'))
    review = report/'pdf-review'
    review.mkdir(exist_ok=True)
    tiles = []
    for i in range(len(rendered)):
        page = rendered[i]
        image = page.render(scale=1.25).to_pil().convert('RGB')
        image.save(review/f'page-{i+1:02d}.png')
        thumb = image.copy()
        thumb.thumbnail((300,430))
        tile = Image.new('RGB',(320,460),'#e6ebef')
        tile.paste(thumb,((320-thumb.width)//2,10))
        ImageDraw.Draw(tile).text((12,440),f'Page {i+1}',fill='black')
        tiles.append(tile)
    cols = 3
    sheet = Image.new('RGB',(cols*320,((len(tiles)+cols-1)//cols)*460),'white')
    for i,tile in enumerate(tiles):
        sheet.paste(tile,((i%cols)*320,(i//cols)*460))
    sheet.save(review/'contact-sheet.png')
    source_paths = [ROOT/'scripts'/name for name in ('build_annotated_surface_report.py',
        'validate_annotated_surface_report.py','audit_annotated_surface.py','audit_annotated_surface_pools.py')]
    result = dict(passed=True,at=datetime.now(timezone.utc).isoformat(),targets=len(rows),
        gallery_panels=len(gallery['panels']),ranker_parameters=22,pdf_pages=len(pdf.pages),
        rendered_pages=len(tiles),all_links_exist=True,all_csv_values_equal_audit=True,
        source_sha256={str(path.relative_to(ROOT)):sha256(path) for path in source_paths},
        human_or_browser_visual_review='Separate from automated checks; see visual-review.json if present.')
    write_json(report/'validation.json',result)
    print(json.dumps(result,indent=2))


if __name__ == '__main__':
    main()
