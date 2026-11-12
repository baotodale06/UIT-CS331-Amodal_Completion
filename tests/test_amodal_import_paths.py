"""Reproduce editable SAM's namespace collision without Torch or a GPU."""
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


PROBE = r'''
import importlib.abc
import importlib.util
from pathlib import Path
import sys

package = Path(sys.argv[1])
# Reproduce the flat-layout editable finder appended by setuptools. The outer
# checkout namespace can be returned by PathFinder before this finder runs.
class EditableFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "segment_anything":
            return importlib.util.spec_from_file_location(fullname, package)
sys.meta_path.append(EditableFinder())
import segment_anything
if sys.argv[2] == "broken":
    assert segment_anything.__file__ is None
    try:
        from segment_anything import SamPredictor
    except ImportError as error:
        assert "unknown location" in str(error), str(error)
        print("Original SamPredictor namespace error reproduced")
    else:
        raise AssertionError("Expected the logged ImportError")
else:
    from segment_anything import build_sam, SamPredictor, SamAutomaticMaskGenerator
    assert Path(segment_anything.__file__).resolve() == package.resolve()
    assert callable(build_sam)
    assert isinstance(SamPredictor, type)
    assert isinstance(SamAutomaticMaskGenerator, type)
    # main.py appends the outer directory later. It must retain the real package.
    sys.path.append(str(package.parents[2]))
    from segment_anything import SamPredictor
    import diffusers
    assert diffusers.MIXED_CONTEXT_FORK
    print("SAM APIs and the Diffusers source package resolve correctly")
'''


@unittest.skipUnless(os.name == "posix", "Bash import-path integration requires Linux/WSL")
class ImportPathTests(unittest.TestCase):
    def test_editable_namespace_collision_and_shared_path_fix(self):
        with tempfile.TemporaryDirectory(prefix="amodal-import-paths-") as temporary:
            repo = Path(temporary) / "amodal checkout with spaces"
            gsa = repo / "Grounded-Segment-Anything"
            package = gsa / "segment_anything/segment_anything/__init__.py"
            package.parent.mkdir(parents=True)
            package.write_text(
                "def build_sam(): pass\n"
                "class SamPredictor: pass\n"
                "class SamAutomaticMaskGenerator: pass\n"
            )
            diffusers = repo / "diffusers/src/diffusers/__init__.py"
            diffusers.parent.mkdir(parents=True)
            diffusers.write_text("MIXED_CONTEXT_FORK = True\n")
            environment = dict(os.environ, PYTHONNOUSERSITE="1")
            environment.pop("PYTHONHOME", None)
            environment["PYTHONPATH"] = os.pathsep.join((str(repo), str(gsa)))
            broken = subprocess.run(
                [sys.executable, "-c", PROBE, str(package), "broken"],
                cwd=repo, env=environment, capture_output=True, text=True,
            )
            self.assertEqual(broken.returncode, 0, broken.stderr)
            self.assertIn("Original SamPredictor namespace error reproduced", broken.stdout)

            helper = Path(__file__).with_name("test_run_amodal_slurm.sh")
            path = subprocess.run(
                ["bash", str(helper), "--print-pythonpath", str(repo)],
                capture_output=True, text=True, check=True,
            ).stdout.strip()
            self.assertEqual(path.split(os.pathsep)[0], str(gsa / "segment_anything"))
            environment["PYTHONPATH"] = path
            fixed = subprocess.run(
                [sys.executable, "-c", PROBE, str(package), "fixed"],
                cwd=repo, env=environment, capture_output=True, text=True,
            )
            self.assertEqual(fixed.returncode, 0, fixed.stderr)
            self.assertIn("resolve correctly", fixed.stdout)


if __name__ == "__main__":
    unittest.main(verbosity=2)
