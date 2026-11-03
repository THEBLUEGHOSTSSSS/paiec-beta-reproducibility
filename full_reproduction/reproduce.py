#!/usr/bin/env python3
"""Five fixed development scopes. No network, production guards or retries."""
import argparse
import os
import sys

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', required=True, choices=('verify', 'replay', 'train'))
    parser.add_argument('--data-root', required=True, help='directory containing only the benchmark subdirectories needed here')
    parser.add_argument('--reference-root', help='explicit prepared immutable derived reference bundle')
    parser.add_argument('--encoder-root', help='pinned local E5 assets; with train, freshly encode head and tail vectors')
    parser.add_argument('--output', required=True, help='fresh explicit output directory')
    args = parser.parse_args(argv)
    out = None
    try:
        from safety import validate, dump, sha, PACKAGE, resources
        data, ref, out, encoder, receipts = validate(args.data_root, args.reference_root, args.output, args.encoder_root)
        if args.mode != 'verify' and ref is None:
            raise ValueError('replay/train requires --reference-root; prepare it with build_reference.py')
        if args.encoder_root and args.mode != 'train':
            raise ValueError('--encoder-root is supported only in train mode')
        for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
            os.environ[key] = '2'
        os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', HF_HUB_DISABLE_IMPLICIT_TOKEN='1', TOKENIZERS_PARALLELISM='false', PYTHONDONTWRITEBYTECODE='1')
        sys.dont_write_bytecode = True
        out.mkdir(parents=True, exist_ok=True)
        dump(out/'input_receipt.json', dict(mode=args.mode, hashes=receipts, package_manifest_sha256=sha(PACKAGE/'package_manifest.json'), scope='fixed public development only; confirmation labels excluded'))
        if args.mode == 'verify':
            dump(out/'verification.json', dict(status='passed', verified_raw_tables=len(receipts['raw']), verified_reference_files=len(receipts['reference']), numerical_execution=False, resources=resources(out)))
            return 0
        from engine import run
        run(args.mode, data, ref, out, encoder)
        return 0
    except Exception as exc:
        if out is not None and out.is_dir() and (out/'input_receipt.json').is_file():
            dump(out/'failure_receipt.json', dict(status='failed_or_uncertain', error=type(exc).__name__+': '+str(exc), mode=args.mode, retries=0, dependent_execution_stopped=True))
        print(type(exc).__name__+': '+str(exc), file=sys.stderr)
        return 2

if __name__ == '__main__':
    raise SystemExit(main())
