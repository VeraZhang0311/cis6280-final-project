#!/usr/bin/env python3
"""Lock A5 model identities, scene order, equal behavior budgets and independent RNGs."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from assignment.training.data.signals import scene_split
from assignment.training.training.onpolicy_a5 import SCHEMA, sha256, validate_scene_plan, verify_models, write_json


def scenes(path):
    value = json.loads(path.read_text(encoding='utf-8-sig'))
    if isinstance(value, dict):
        value = value.get('scene_seeds', value.get('scenes'))
    if not isinstance(value, list) or any(type(s) is not int or s < 0 for s in value):
        raise ValueError('Expected an explicit scene seed list')
    return value


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--a4-run', type=Path, required=True)
    p.add_argument('--policy', type=Path, required=True)
    p.add_argument('--world-model', type=Path, required=True)
    p.add_argument('--development-scenes', type=Path, required=True)
    p.add_argument('--reserved-scenes', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--train-scenes', type=int, default=128)
    p.add_argument('--calibration-scenes', type=int, default=32)
    p.add_argument('--train-anchors', type=int, default=5000)
    p.add_argument('--calibration-anchors', type=int, default=500)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--keep', type=float, default=.25)
    args = p.parse_args()
    if args.out.exists() or min(args.train_scenes, args.calibration_scenes) < 1:
        p.error('Use a new plan path and positive scene counts')
    reference = json.loads((args.a4_run/'run.json').read_text(encoding='utf-8'))
    if (sha256(args.policy) != reference['models']['goal_actor']['sha256']
            or sha256(args.world_model) != reference['models']['world_model']['sha256']):
        p.error('Actor/WM must match the actual A4 evaluation manifest')
    development, reserved = scenes(args.development_scenes), scenes(args.reserved_scenes)
    if set(reference['scene_seeds']) != set(development) or set(development) & set(reserved) or not reserved:
        p.error('Lock disjoint development and nonempty formal reserved scene lists')
    order, seed = [], 0
    while len(order) < args.train_scenes+args.calibration_scenes:
        if scene_split(seed) == 'train' and seed not in development+reserved:
            order.append(seed)
        seed += 1
    plan = dict(schema=SCHEMA,
                models=dict(actor=dict(path=str(args.policy.resolve()), sha256=sha256(args.policy)),
                            world_model=dict(path=str(args.world_model.resolve()), sha256=sha256(args.world_model))),
                a4_manifest_sha256=sha256(args.a4_run/'run.json'),
                scenes=dict(train=order[:args.train_scenes], calibration=order[args.train_scenes:]),
                excluded=dict(development=development, reserved=reserved),
                exclusion_files=dict(development=sha256(args.development_scenes), reserved=sha256(args.reserved_scenes)),
                anchors=dict(train=args.train_anchors, calibration=args.calibration_anchors),
                candidates=dict(train=16, calibration=32), behavior_candidates=[16, 32],
                sample_seeds=[0, 2, 3], seed=args.seed, keep=args.keep, horizon=1795,
                stride=5, blocks=16, goal_every=80, reference_source='scripted nominal + recovery',
                order='fixed scene order then sample seed order; no success filtering',
                rng=dict(main='scene + 1000000 * sample_seed',
                         branch='SeedSequence([seed, scene, sample_seed, behavior, control_step, 5105])',
                         anchor='SeedSequence([seed, scene, sample_seed, behavior, control_step, 5106])'))
    validate_scene_plan(plan)
    verify_models(args.policy, args.world_model, plan)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.out, plan)
    print(json.dumps(dict(plan=str(args.out), sha256=sha256(args.out))))


if __name__ == '__main__':
    main()
