#!/usr/bin/env python3
"""Validate and extract the published, data-minimized reference ZIP offline."""
import argparse
import hashlib
import json
import stat
import zipfile
from pathlib import Path
from safety import MANIFEST, PACKAGE, dump, relative, resources, safe_path, sha

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle-archive',required=True,help='downloaded reference-20261031.zip')
    parser.add_argument('--output',required=True,help='fresh reference directory')
    args=parser.parse_args()
    archive=safe_path(args.bundle_archive)
    output=safe_path(args.output)
    if not archive.is_file():raise ValueError('reference archive missing')
    if output.exists() and (not output.is_dir() or any(output.iterdir())):raise ValueError('output must be fresh')
    if archive.is_relative_to(output) or output.is_relative_to(PACKAGE) or PACKAGE.is_relative_to(output):raise ValueError('input/output overlap')
    resources(output)
    expected=MANIFEST['reference_files']
    names=set(expected)|{'bundle_manifest.json'}
    with zipfile.ZipFile(archive) as bundle:
        infos=bundle.infolist()
        if len(infos)!=len(names) or {x.filename for x in infos}!=names:raise ValueError('archive inventory differs')
        if sum(x.file_size for x in infos)>=1024**3:raise ValueError('reference exceeds1GiB')
        for entry in infos:
            relative(output,entry.filename)
            if entry.is_dir() or stat.S_ISLNK(entry.external_attr>>16):raise ValueError('archive link/directory refused')
            ceiling=expected[entry.filename]['bytes'] if entry.filename in expected else 200000
            if entry.file_size>ceiling:raise ValueError('archive member size differs')
            digest=hashlib.sha256()
            with bundle.open(entry) as stream:
                for chunk in iter(lambda:stream.read(1024*1024),b''):digest.update(chunk)
            if entry.filename in expected and digest.hexdigest()!=expected[entry.filename]['sha256']:raise ValueError('archive member hash differs')
        metadata=json.loads(bundle.read('bundle_manifest.json'))
        if metadata['input_manifest_sha256']!=sha(PACKAGE/'input_manifest.json') or metadata['files']!={n:m['sha256'] for n,m in expected.items()}:
            raise ValueError('bundle metadata differs')
        output.mkdir(parents=True,exist_ok=True)
        for entry in infos:
            dest=relative(output,entry.filename)
            dest.parent.mkdir(parents=True,exist_ok=True)
            with bundle.open(entry) as src,dest.open('xb') as dst:
                for chunk in iter(lambda:src.read(1024*1024),b''):dst.write(chunk)
    print(json.dumps({'status':'verified','reference_files':len(expected),'archive_sha256':sha(archive),'raw_data_included':False}))

if __name__=='__main__':
    try:main()
    except Exception as exc:raise SystemExit(type(exc).__name__+': '+str(exc))
