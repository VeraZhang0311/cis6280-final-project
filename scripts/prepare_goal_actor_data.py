#!/usr/bin/env python3
"""Download the released offline actor cache and frozen encoder, without raw videos."""
import argparse
import json
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download, snapshot_download


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--checkpoints', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    args = parser.parse_args()
    api = HfApi()
    dataset_repo = 'YongYong/cis6280-stack-datasets-v2'
    model_repo = 'YongYong/cis6280-stack-world-models-v2'
    dataset = api.repo_info(dataset_repo, repo_type='dataset', files_metadata=True)
    models = api.repo_info(model_repo, files_metadata=True)
    prefixes = ('derived/jepa-cache-4cube-20k/', 'derived/latents-smwm-c3/')
    files = [f for f in dataset.siblings if f.rfilename.startswith(prefixes)]
    required = {prefixes[0]+'index.json', prefixes[0]+'actions.f32',
                prefixes[1]+'latents.json', prefixes[1]+'latents.f16'}
    if not required <= {f.rfilename for f in files}:
        raise RuntimeError('Expected released cache files are missing')
    manifest = dict(dataset_repo=dataset_repo, dataset_revision=dataset.sha,
                    model_repo=model_repo, model_revision=models.sha,
                    files=[dict(path=f.rfilename, bytes=f.size,
                                sha256=f.lfs.sha256 if f.lfs else None) for f in files])
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2)+'\n')
    print(json.dumps(dict(stage='download', **manifest)), flush=True)
    snapshot_download(dataset_repo, repo_type='dataset', revision=dataset.sha,
                      allow_patterns=sorted(required), local_dir=args.dataset, max_workers=4)
    for name in ('jepa/smwm-c3/pretrain.pt', 'policy/df-smwm-c1-sigma04/policy.pt'):
        hf_hub_download(model_repo, name, revision=models.sha, local_dir=args.checkpoints)
    print(json.dumps(dict(stage='ready', dataset=str(args.dataset), checkpoints=str(args.checkpoints))), flush=True)


if __name__ == '__main__':
    main()
