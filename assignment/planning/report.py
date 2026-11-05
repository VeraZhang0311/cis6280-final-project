"""Compare evaluator reports with scene-clustered paired bootstrap intervals."""
import argparse
import json
from pathlib import Path
import random
import statistics


def summarize_episodes(episodes):
    rows = list(episodes.values())
    return dict(mean_boxes=statistics.mean(e['max_placed'] for e in rows),
                reached={str(k): statistics.mean(e['max_placed'] >= k for e in rows) for k in range(1, 5)},
                ground_fall_rate=statistics.mean(e['ground_falls'] > 0 for e in rows),
                agent_s_per_block=statistics.mean(e['agent_s']/((e['steps']+4)//5) for e in rows))


def read_reports(paths):
    episodes, signatures, configs = {}, set(), set()
    for path in paths:
        report = json.loads(Path(path).read_text())
        signatures.add(json.dumps({key: report.get(key) for key in
                                   ('policy', 'world_model', 'split', 'task', 'goals')}, sort_keys=True))
        configs.add(json.dumps(dict(kind=report['kind'], config=report['config']), sort_keys=True))
        for episode in report['episodes']:
            key = (episode['scene_seed'], report['sample_seed'])
            if key in episodes:
                raise ValueError(f'duplicate scene/sample pair {key}')
            episodes[key] = episode
    if len(signatures) != 1 or len(configs) != 1 or not episodes:
        raise ValueError('reports must contain one experiment configuration and compatible task metadata')
    return episodes, signatures.pop()


def compare(baseline_paths, candidate_paths, repeats=10000):
    baseline, signature = read_reports(baseline_paths)
    candidate, other_signature = read_reports(candidate_paths)
    if signature != other_signature or baseline.keys() != candidate.keys():
        raise ValueError('comparisons require identical artifacts, tasks and scene/sample pairs')
    scenes = sorted({key[0] for key in baseline})
    sample_sets = [{key[1] for key in baseline if key[0] == scene} for scene in scenes]
    if any(samples != sample_sets[0] for samples in sample_sets):
        raise ValueError('each scene must have the same policy seeds')
    values = []
    for scene in scenes:
        values.append(statistics.mean(candidate[scene, sample]['max_placed']-baseline[scene, sample]['max_placed']
                                      for sample in sample_sets[0]))
    rng = random.Random(6280)
    bootstrap = sorted(statistics.mean(rng.choices(values, k=len(values))) for _ in range(repeats))

    return dict(scenes=len(scenes), episodes=len(baseline), sample_seeds=sorted(sample_sets[0]),
                baseline=summarize_episodes(baseline), candidate=summarize_episodes(candidate),
                paired_delta=statistics.mean(values),
                scene_bootstrap_95ci=[bootstrap[int(.025*repeats)], bootstrap[min(repeats-1, int(.975*repeats))]],
                bootstrap_repeats=repeats, bootstrap_seed=6280)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--baseline', type=Path, nargs='+', required=True)
    p.add_argument('--candidate', type=Path, nargs='+', required=True)
    p.add_argument('--out', type=Path)
    args = p.parse_args()
    report = compare(args.baseline, args.candidate)
    text = json.dumps(report, indent=2)+'\n'
    print(text, end='')
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text)


if __name__ == '__main__':
    main()
