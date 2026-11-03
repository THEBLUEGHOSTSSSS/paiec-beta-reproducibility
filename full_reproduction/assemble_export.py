#!/usr/bin/env python3
"""Assemble the original inference layout locally. No publication or network."""
import argparse
import shutil
from safety import MANIFEST, PACKAGE, safe_path, relative, sha, dump, resources

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--heads-root',required=True)
    parser.add_argument('--encoder-root',required=True)
    parser.add_argument('--output',required=True)
    args=parser.parse_args();heads=safe_path(args.heads_root,True);encoder=safe_path(args.encoder_root,True);out=safe_path(args.output)
    if out.exists() and (not out.is_dir() or any(out.iterdir())):raise ValueError('fresh output required')
    for root in [heads,encoder,PACKAGE]:
        if out.is_relative_to(root) or root.is_relative_to(out):raise ValueError('input/output overlap')
    names=['config.json','model.safetensors','sentencepiece.bpe.model','special_tokens_map.json','tokenizer.json','tokenizer_config.json']
    for name in names:
        if sha(relative(encoder,name))!=MANIFEST['encoder']['files'][name]['sha256']:raise ValueError('encoder asset hash mismatch')
    for seed in range(5):
        if not relative(heads,f'head_seed{seed}.npz').is_file():raise ValueError('five head files required')
    import importlib.util
    import numpy as np
    spec=importlib.util.spec_from_file_location('frozen_inference',PACKAGE/'inference_model.py')
    frozen=importlib.util.module_from_spec(spec);spec.loader.exec_module(frozen)
    for seed in range(5):
        with np.load(relative(heads,f'head_seed{seed}.npz'),allow_pickle=False) as arrays:
            frozen._validate_state({key:arrays[key].copy() for key in arrays.files})
    resources(out);out.mkdir(parents=True,exist_ok=True);assets=out/'assets';assets.mkdir()
    shutil.copyfile(PACKAGE/'inference_model.py',out/'model.py')
    shutil.copyfile(PACKAGE/'prediction_config.json',out/'prediction_config.json')
    shutil.copyfile(PACKAGE/'requirements.txt',out/'requirements.txt')
    inventory={}
    for name in names:
        shutil.copyfile(relative(encoder,name),assets/name);inventory[name]=sha(assets/name)
    for seed in range(5):
        name=f'head_seed{seed}.npz';shutil.copyfile(relative(heads,name),assets/name);inventory[name]=sha(assets/name)
    dump(assets/'manifest.json',inventory)
    dump(out/'assembly_receipt.json',dict(external_encoder_revision=MANIFEST['encoder']['revision'],asset_sha256=inventory,model_source_sha256=sha(out/'model.py'),publication=False,hosted_compatibility='unexecuted',resources=resources(out)))
if __name__=='__main__':
    try:main()
    except Exception as error:raise SystemExit(type(error).__name__+': '+str(error))
