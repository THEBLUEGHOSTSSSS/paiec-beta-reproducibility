"""Local additional checks; protected Commander tests remain unmodified."""
import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import safety


class SafetyContract(unittest.TestCase):
    def test_help_without_site_packages(self):
        result = subprocess.run([sys.executable, '-S', str(safety.PACKAGE/'reproduce.py'), '--help'], capture_output=True, text=True, timeout=20)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_missing_path_without_site_packages(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            result = subprocess.run([sys.executable, '-S', str(safety.PACKAGE/'reproduce.py'), '--mode', 'train', '--data-root', str(root/'absent'), '--output', str(root/'out')], capture_output=True, text=True, timeout=20)
            self.assertNotEqual(result.returncode, 0)
            self.assertNotIn('ModuleNotFoundError', result.stderr)
            self.assertFalse((root/'out').exists())

    def fixture(self, root):
        data = root/'data'
        data.mkdir()
        (data/'allowed').mkdir()
        raw = data/'allowed'/'responses.parquet'
        raw.write_bytes(b'not parsed until provenance checks complete')
        package = root/'package'
        package.mkdir()
        (package/'package_manifest.json').write_text(json.dumps({'files': {}}))
        manifest = {'raw_files': {'allowed/responses.parquet': {'sha256': hashlib.sha256(raw.read_bytes()).hexdigest()}}, 'reference_files': {}}
        return data, raw, package, manifest

    def test_hash_mismatch_does_not_create_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            data, raw, package, manifest = self.fixture(root)
            raw.write_bytes(b'changed')
            with patch.object(safety, 'MANIFEST', manifest), patch.object(safety, 'PACKAGE', package):
                with self.assertRaisesRegex(ValueError, 'hash-mismatched'):
                    safety.validate(data, None, root/'out')
            self.assertFalse((root/'out').exists())

    def test_symlink_input_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            data, raw, package, manifest = self.fixture(root)
            target = root/'real'
            raw.rename(target)
            raw.symlink_to(target)
            with patch.object(safety, 'MANIFEST', manifest), patch.object(safety, 'PACKAGE', package):
                with self.assertRaisesRegex(ValueError, 'symlink'):
                    safety.validate(data, None, root/'out')

    def test_traversal_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            with self.assertRaisesRegex(ValueError, 'traversal'):
                safety.safe_path(root/'a'/'..'/'out')
            for name in ('../escape', '/absolute', 'a/../../escape', 'a\\escape'):
                with self.assertRaises(ValueError):
                    safety.relative(root, name)

    def test_both_overlap_directions_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            data, raw, package, manifest = self.fixture(root)
            with patch.object(safety, 'MANIFEST', manifest), patch.object(safety, 'PACKAGE', package):
                for output in (data/'out', root, package/'out'):
                    with self.assertRaises(ValueError):
                        safety.validate(data, None, output)

    def test_nonempty_output_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            output = root/'out'
            output.mkdir()
            keep = output/'keep'
            keep.write_text('unchanged')
            with self.assertRaises(ValueError):
                safety.validate(root/'absent', None, output)
            self.assertEqual(keep.read_text(), 'unchanged')


if __name__ == '__main__':
    unittest.main()
