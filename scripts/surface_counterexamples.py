#!/usr/bin/env python3
"""Analytical stress fixtures, not PMC generations or natural-image statistics."""
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from amodal_benchmark.dataset import write_json
from amodal_benchmark.surface import METHODS, score_surface, select_candidate


def build_fixtures(size=128):
    yy, xx = np.indices((size, size))
    full = (xx >= 8) & (xx < size - 8) & (yy >= 8) & (yy < size - 8)
    occlusion = xx >= size // 2
    visible, hidden = full & ~occlusion, full & occlusion
    solid = np.full((size, size, 3), (185, 55, 45), np.uint8)
    wrong = solid.copy()
    wrong[hidden] = (30, 65, 175)
    stripe = np.where(((yy // 8) % 2)[..., None], np.array([205, 60, 50]), np.array([50, 45, 35]))
    stripe = np.broadcast_to(stripe, (size, size, 3)).astype(np.uint8).copy()
    phase = stripe.copy()
    phase[hidden] = np.roll(stripe, 8, axis=0)[hidden]
    material = solid.copy()
    material[hidden] = (30, 65, 175)
    texture = np.where((((yy // 4 + xx // 4) % 2))[..., None],
                       np.array([180, 145, 110]), np.array([85, 65, 45])).astype(np.uint8)
    blurred = np.array(Image.fromarray(texture).filter(ImageFilter.GaussianBlur(3)))
    smooth = texture.copy()
    smooth[hidden] = blurred[hidden]
    gradient = np.broadcast_to(np.linspace(190, 65, size)[None, :, None], (size, size, 3)).astype(np.uint8).copy()
    uniform = gradient.copy()
    uniform[hidden] = gradient[:, size // 2 - 1][0]
    return full, visible, occlusion, [
        ('wrong_color', solid, [wrong, solid], ['wrong blue fill', 'correct red surface']),
        ('stripe_phase', stripe, [phase, stripe], ['shifted stripe phase', 'correct stripe continuation']),
        ('material_transition', material, [solid, material], ['wrong uniform red', 'correct hidden blue material']),
        ('texture_smoothing', texture, [smooth, texture], ['blurred hidden texture', 'correct texture']),
        ('legitimate_shading', gradient, [uniform, gradient], ['incorrect flat continuation', 'correct changing illumination']),
    ]


def main():
    output = Path(sys.argv[1] if len(sys.argv) > 1 else 'output/surface-consistency-20261108/counterexamples')
    output.mkdir(parents=True, exist_ok=True)
    full, visible, occlusion, fixtures = build_fixtures()
    rows = []
    panel = Image.new('RGB', (1040, 235 * len(fixtures)), 'white')
    draw = ImageDraw.Draw(panel)
    for row_index, (name, truth, candidates, labels) in enumerate(fixtures):
        observed = truth.copy()
        observed[occlusion] = 127
        scores = [score_surface(observed, visible, occlusion, candidate, full) for candidate in candidates]
        errors = [float(np.abs(candidate[full & occlusion].astype(float) - truth[full & occlusion]).mean() / 255)
                  for candidate in candidates]
        selected = {method: select_candidate(scores, weights) for method, weights in METHODS.items()}
        row = dict(name=name, hidden_mae=errors, candidates=labels, scores=scores,
                   selections=selected, score_scales='analytic raw units; not development-normalized')
        rows.append(row)
        top = row_index * 235
        for i, (label, image) in enumerate(zip(['Observed', 'Truth'] + labels, [observed, truth] + candidates)):
            panel.paste(Image.fromarray(image).resize((180, 180), Image.Resampling.NEAREST), (i * 255 + 10, top + 30))
            draw.text((i * 255 + 10, top + 12), label, fill='black')
        draw.text((10, top + 216), f"{name}: full chooses {selected['surface_full']['index']}; local chooses "
                  f"{selected['local_surface']['index']}; true hidden MAE {errors[0]:.3f} / {errors[1]:.3f}", fill='black')
    panel.save(output / 'counterexamples.png')
    write_json(output / 'results.json', rows)
    print(json.dumps([{k: row[k] for k in ('name', 'hidden_mae')} | {
        'full': row['selections']['surface_full']['index'], 'local': row['selections']['local_surface']['index']}
        for row in rows], indent=2))


if __name__ == '__main__':
    main()
