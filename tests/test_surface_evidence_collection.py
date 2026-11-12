"""Evidence recovery rejects corruption and unsafe archives before extraction."""
import hashlib
import io
from pathlib import Path
import sys
import tarfile
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from collect_surface_job_evidence import recover_archive, status_program, successful


class SurfaceEvidenceCollectionTests(unittest.TestCase):
    def archive(self, root, files):
        path = root / 'evidence.tar.gz'
        with tarfile.open(path, 'w:gz') as bundle:
            for name, content in files.items():
                member = tarfile.TarInfo(name)
                member.size = len(content)
                bundle.addfile(member, io.BytesIO(content))
        return path

    def test_remote_manifest_checked_and_existing_evidence_preserved(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            files = {'output/study/report.json': b'{"passed":true}'}
            expected = {name:hashlib.sha256(data).hexdigest() for name,data in files.items()}
            archive = self.archive(root, files)
            destination = root / 'recovered'
            for _ in range(2):
                self.assertEqual(recover_archive(archive, destination, ['output/study'], expected), expected)
            target = destination / 'output/study/report.json'
            target.write_bytes(b'pre-existing different evidence')
            with self.assertRaisesRegex(ValueError, 'Existing recovery evidence differs'):
                recover_archive(archive, destination, ['output/study'], expected)
            self.assertEqual(target.read_bytes(), b'pre-existing different evidence')

    def test_checksum_mismatch_prevents_all_extraction(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            archive = self.archive(root, {'output/study/a.json': b'first', 'output/study/b.json': b'changed'})
            expected = {'output/study/a.json':hashlib.sha256(b'first').hexdigest(),
                        'output/study/b.json':hashlib.sha256(b'original').hexdigest()}
            destination = root / 'recovered'
            with self.assertRaisesRegex(ValueError, 'Remote member checksum mismatch'):
                recover_archive(archive, destination, ['output/study'], expected)
            self.assertFalse(destination.exists())

    def test_path_traversal_prevents_all_extraction(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            archive = self.archive(root, {'output/study/valid.json': b'valid', '../outside': b'escape'})
            with self.assertRaisesRegex(ValueError, 'Unsafe archive path'):
                recover_archive(archive, root / 'recovered', ['output/study'])
            self.assertFalse((root / 'recovered').exists())
            self.assertFalse((root / 'outside').exists())

    def test_running_or_failed_jobs_never_authorize_collection(self):
        for state, code in [('RUNNING', '0:0'), ('PENDING', '0:0'), ('FAILED', '1:0'), ('COMPLETED', '1:0')]:
            self.assertFalse(successful({'1':dict(state=state, exit_code=code)}, '1'))
        self.assertFalse(successful({}, '1'))
        self.assertTrue(successful({'1':dict(state='COMPLETED', exit_code='0:0')}, '1'))

    def test_resume_monitor_watches_replacement_not_timed_out_parent(self):
        program = status_program(39387, 37814, 37843)
        self.assertIn('39387,37814,37843', program)
        self.assertNotIn('37812', program)
        compile(program, '<remote-status>', 'exec')
        for invalid in (0, -1, '39387; false'):
            with self.assertRaises(ValueError):
                status_program(invalid, 37814, 37843)


if __name__ == '__main__':
    unittest.main()
