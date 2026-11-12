#!/usr/bin/env python3
"""Print a saved candidate's primitive-score differences without changing evidence."""
import json
import os
from pathlib import Path
import platform
import sys

import numpy as np
import PIL
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from amodal_benchmark.surface import score_surface

experiment = ROOT / 'output/surface-followup-20261110'
case = 'VG-000-01154e3322'
inputs = experiment / 'inputs' / case
folder = experiment / 'full/candidates' / case
rgb = lambda path: np.array(Image.open(path).convert('RGB'))
mask = lambda path: np.array(Image.open(path).convert('L')) > 0
actual = score_surface(rgb(inputs / 'observed.png'), mask(inputs / 'visible.png'),
    mask(inputs / 'occlusion.png'), rgb(folder / 'seed-11.png'), mask(folder / 'seed-11-mask.png'))
expected = json.loads((folder / 'seed-11.json').read_text())['surface']
print(json.dumps(dict(host=platform.node(), python=sys.version, numpy=np.__version__,
    pillow=PIL.__version__, numpy_path=np.__file__, core=os.environ.get('OPENBLAS_CORETYPE'),
    differences={key:dict(expected=value, actual=actual[key], difference=actual[key]-value)
                 for key,value in expected.items() if isinstance(value, float)},
    cpu_features=np.core._multiarray_umath.__cpu_features__), indent=2))
np.show_config()
