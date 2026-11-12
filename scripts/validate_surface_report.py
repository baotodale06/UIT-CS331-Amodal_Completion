#!/usr/bin/env python3
"""Verify report arithmetic, perceptual input identities, PDFs and gallery files."""
import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import sys

import numpy as np
from PIL import Image
from pypdf import PdfReader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from amodal_benchmark.dataset import sha256, write_json
from amodal_benchmark.surface_evaluation import summarize_cases
from scripts.build_surface_report import attach_perceptual, perceptual_summary
from scripts.validate_surface_experiment import assert_equivalent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pilot', type=Path, required=True)
    parser.add_argument('--replication', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    rows, cohort_rows = [], {}
    perceptual_checks = 0
    for cohort, experiment in [('Pilot test', args.pilot), ('Replication', args.replication)]:
        validation = json.loads((experiment / 'validation.json').read_text())
        transport = json.loads((experiment / 'transport_verification.json').read_text())
        if validation['status'] != 'passed' or transport['status'] != 'passed':
            raise ValueError('Experiment validation incomplete')
        cases = json.loads((experiment / 'cases.json').read_text())
        seeds = json.loads((experiment / 'protocol.json').read_text())['candidate_seeds']
        for case in cases:
            perceptual = json.loads((experiment / 'perceptual' / (case['case_id'] + '.json')).read_text())
            if perceptual['case_id'] != case['case_id'] or perceptual['truth_hashes'] != case['evaluation_hashes']:
                raise ValueError('Perceptual truth identity mismatch')
            if [v['seed'] for v in perceptual['values']] != seeds:
                raise ValueError('Perceptual candidate pool changed')
            for value in perceptual['values']:
                candidate = json.loads((experiment / 'full/candidates' / case['case_id'] / f"seed-{value['seed']}.json").read_text())
                digest = candidate['artifacts'].get(f"seed-{value['seed']}.png")
                if digest:
                    if value['candidate_sha256'] != digest or value['missing_generation']:
                        raise ValueError('Perceptual candidate identity mismatch')
                    if not np.isfinite(value['masked_spatial_lpips']) or value['masked_spatial_lpips'] < 0:
                        raise ValueError('Invalid learned-metric value')
                elif value['masked_spatial_lpips'] is not None or not value['missing_generation']:
                    raise ValueError('Missing perceptual input was silently scored')
                perceptual_checks += 1
        group = json.loads((experiment / 'analysis/test_cases.json').read_text())
        attach_perceptual(experiment, group)
        rows.extend(group)
        cohort_rows[cohort] = group
    saved = json.loads((args.report / 'pooled_test_summary.json').read_text())
    assert_equivalent(summarize_cases(rows, 'test'), saved['methods'], 'pooled methods')
    for cohort, group in cohort_rows.items():
        assert_equivalent(summarize_cases(group, 'test'), saved['cohorts'][cohort], cohort)
        assert_equivalent(perceptual_summary(group), saved['perceptual_cohorts'][cohort], cohort + ' perceptual')
    assert_equivalent(perceptual_summary(rows), saved['perceptual'], 'perceptual arithmetic')
    with (args.report / 'tables/candidates.csv').open(newline='', encoding='utf-8') as stream:
        candidates = list(csv.DictReader(stream))
    expected = {(row['case_id'], metric['seed']): metric for row in rows for metric in row['metrics']}
    if len(candidates) != len(expected) or {(r['case_id'], int(r['seed'])) for r in candidates} != expected.keys():
        raise ValueError('Candidate table coverage changed')
    for candidate in candidates:
        actual = expected[(candidate['case_id'], int(candidate['seed']))]
        for key in ('hidden_core_mae', 'visible_mae', 'hidden_mask_recall', 'masked_spatial_lpips'):
            if actual.get(key) is not None:
                assert_equivalent(float(candidate[key]), actual[key], candidate['case_id'] + '/' + key)
    gallery = json.loads((args.report / 'gallery/cases.json').read_text())
    if len(gallery) != len(rows) or {r['case_id'] for r in gallery} != {r['case_id'] for r in rows}:
        raise ValueError('Gallery coverage changed')
    for item in gallery:
        for key, size in [('image', (1536, 418)), ('masks', (1536, 340))]:
            with Image.open(args.report / 'gallery' / item[key]) as image:
                image.load()
                if image.size != size:
                    raise ValueError('Gallery dimensions changed')
    markdown = (args.report / 'REPORT.md').read_text(encoding='utf-8')
    links = re.findall(r'\]\(([^)]+)\)', markdown)
    for link in links:
        if not link.startswith(('https://', 'http://', '#')) and not (args.report / link).exists():
            raise ValueError('Broken report link: ' + link)
    pdf = PdfReader(args.report / 'REPORT.pdf')
    page_text = [page.extract_text() or '' for page in pdf.pages]
    if any(len(text.strip()) < 35 for text in page_text) or len(page_text) < 8:
        raise ValueError('PDF has missing or unexpectedly empty pages')
    build = json.loads((args.report / 'build.json').read_text())
    if build['pdf_sha256'] != sha256(args.report / 'REPORT.pdf') or build['report_sha256'] != sha256(args.report / 'REPORT.md'):
        raise ValueError('Rendered report changed after building')
    result = dict(status='passed', verified_at=datetime.now(timezone.utc).isoformat(), test_cases=len(rows),
        candidate_table_rows=len(candidates), perceptual_input_identity_checks=perceptual_checks,
        gallery_image_checks=2 * len(gallery), report_link_checks=len(links), pdf_pages=len(pdf.pages),
        numerical_summaries='Pooled and separate cohort methods, intervals and perceptual summaries recomputed',
        learned_metric_limit='Input identities and saved values checked; the neural model is not recomputed by this CPU validator',
        report_sha256=sha256(args.report / 'REPORT.md'), pdf_sha256=sha256(args.report / 'REPORT.pdf'),
        validator_sha256=sha256(Path(__file__)))
    write_json(args.report / 'validation.json', result)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
