"""Standard-library-only input validation, performed before numerical imports."""
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import resource
import shutil
import sys

PACKAGE = Path(__file__).resolve().parent
REFERENCE = None
MANIFEST = json.loads((PACKAGE / 'input_manifest.json').read_text())

def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()

def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)+'\n')

def safe_path(value, directory=False):
    raw = Path(value).expanduser()
    if '..' in raw.parts:
        raise ValueError('path traversal refused')
    path = raw if raw.is_absolute() else Path.cwd()/raw
    for part in (path, *path.parents):
        if part.is_symlink():
            raise ValueError('symlink refused: '+str(part))
    path = path.resolve()
    if directory and not path.is_dir():
        raise ValueError('input directory missing: '+str(path))
    return path

def relative(root, name):
    member = PurePosixPath(name)
    if member.is_absolute() or '..' in member.parts or not member.parts or '\\' in name:
        raise ValueError('unsafe manifest member')
    path = safe_path(root / str(member))
    if not path.is_relative_to(root):
        raise ValueError('escaped input')
    return path

def reference_path(name):
    if REFERENCE is None:
        raise ValueError('prepared reference bundle required')
    return relative(REFERENCE, name)

def resources(output=None):
    usage = resource.getrusage(resource.RUSAGE_SELF)
    rss = usage.ru_maxrss * (1 if sys.platform == 'darwin' else 1024)
    if rss >= 4*1024**3:
        raise RuntimeError('RSS reached 4 GiB')
    location = output or PACKAGE
    while not location.exists():
        location = location.parent
    free = shutil.disk_usage(location).free
    if free < 1024**3:
        raise RuntimeError('free disk below 1 GiB')
    size = sum(member.stat().st_size for member in output.rglob('*') if member.is_file()) if output and output.exists() else 0
    if size >= 1024**3:
        raise RuntimeError('output reached 1 GiB')
    return dict(peak_rss_bytes=rss, output_bytes=size, free_disk_bytes=free, cpu_threads=2)

def validate(data_root, reference_root, output, encoder_root=None):
    out = safe_path(output)
    if out.exists() and (not out.is_dir() or any(out.iterdir())):
        raise ValueError('output must be fresh and empty')
    data = safe_path(data_root, directory=True)
    ref = safe_path(reference_root, directory=True) if reference_root else None
    enc = safe_path(encoder_root, directory=True) if encoder_root else None
    for root in [data, ref, enc, PACKAGE]:
        if root and (out.is_relative_to(root) or root.is_relative_to(out)):
            raise ValueError('input/output overlap refused')
    receipts = {'raw': {}, 'reference': {}, 'encoder': {}}
    for name, meta in MANIFEST['raw_files'].items():
        member = relative(data, name)
        if not member.is_file() or sha(member) != meta['sha256']:
            raise ValueError('missing or hash-mismatched allowed raw input: '+name)
        receipts['raw'][name] = meta['sha256']
    if ref:
        for name, meta in MANIFEST['reference_files'].items():
            member = relative(ref, name)
            if not member.is_file() or sha(member) != meta['sha256']:
                raise ValueError('missing or hash-mismatched reference: '+name)
            receipts['reference'][name] = meta['sha256']
        bundle = json.loads(relative(ref, 'bundle_manifest.json').read_text())
        if bundle['input_manifest_sha256'] != sha(PACKAGE/'input_manifest.json') or bundle['files'] != receipts['reference']:
            raise ValueError('reference-bundle manifest mismatch')
    if enc:
        for name, meta in MANIFEST['encoder']['files'].items():
            member = relative(enc, name)
            if not member.is_file() or sha(member) != meta['sha256']:
                raise ValueError('missing or hash-mismatched encoder asset: '+name)
            receipts['encoder'][name] = meta['sha256']
    for name, expected in json.loads((PACKAGE/'package_manifest.json').read_text())['files'].items():
        if sha(relative(PACKAGE, name)) != expected:
            raise ValueError('packaged source drift: '+name)
    resources(out)
    global REFERENCE
    REFERENCE = ref
    return data, ref, out, enc, receipts
