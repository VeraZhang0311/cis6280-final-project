import unittest
import json
from pathlib import Path
import tempfile
from assignment.training.planning.v11 import choose_g5, paired_comparison, variants, compatible_reuse, TASK, SEEDS


class PilotAndPairs(unittest.TestCase):
    def test_precommitted_tie_prefers_weighted_and_clear_gain_selects_min(self):
        for g1, g2, selected in [(1., 1., 'G2'), (13/12, 1., 'G2'), (14/12, 1., 'G1'), (1., 2., 'G2')]:
            self.assertEqual(choose_g5({'mean_max_placed': g1}, {'mean_max_placed': g2}, 12)['selected_source'], selected)

    def test_new_g4_and_selected_g5_are_bidirectional(self):
        for scoring in ('tail_min','tail_weighted'):
            groups = variants(scoring)
            self.assertEqual(groups['G4']['scoring'], 'terminal')
            self.assertEqual(groups['G5']['scoring'], scoring)
            for g in ('G4','G5'):
                self.assertEqual(groups[g]['realignment_mode'], 'bidirectional')
            self.assertNotIn('realignment_mode', groups['G3'])

    def test_reuse_excludes_old_g4_and_rejects_timeout_config(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            inputs = {k:dict(sha256=k) for k in ('policy','world_model','goals')}
            parent = dict(protocol=dict(workers=8,gpus='0,1,2,3',task=TASK,sample_seeds=SEEDS,
                          inputs=inputs,initial_variants={g:c for g,c in variants().items() if g!='G4'},
                          full_scenes=[100038]), execution_signature={},runtime=[])
            (root/'run.json').write_text(json.dumps(parent))
            for g in ('G0','G4'):
                group=root/'pilot'/g
                episode=group/'episodes/scene-100038.sample-0'
                episode.mkdir(parents=True)
                (group/'run.json').write_text(json.dumps(dict(config=variants()['G0'],protocol_id='original')))
                (episode/'complete.json').write_text(json.dumps(dict(protocol_id='original')))
                (episode/'summary.json').write_text(json.dumps(dict(scene_seed=100038,sample_seed=0,variant=g)))
                for f in ('steps.jsonl','events.jsonl','video.mp4'):(episode/f).write_text('fixture')
            reused=compatible_reuse(root,inputs,8,'0,1,2,3')
            self.assertEqual(list(reused['episodes']), ['pilot/G0/scene-100038.sample-0'])
            parent['protocol']['initial_variants']['G3']['timeout_advance']=True
            (root/'run.json').write_text(json.dumps(parent))
            with self.assertRaises(ValueError):compatible_reuse(root,inputs,8,'0,1,2,3')

    def test_pairs_match_identity_and_bootstrap_whole_scene(self):
        base = [dict(scene_seed=s, sample_seed=k, max_placed=1, final_count=1)
                for s in (1, 2) for k in (0, 2, 3)]
        treatment = [dict(r, max_placed=2 if r['scene_seed']==1 else 0) for r in reversed(base)]
        pair = paired_comparison(base, treatment)
        self.assertEqual((pair['improved'], pair['tied'], pair['regressed']), (3, 0, 3))
        self.assertEqual(pair['mean_peak_delta'], 0)
        self.assertEqual(pair['mean_peak_delta_ci95_scene_bootstrap'], [-1, 1])
        with self.assertRaises(ValueError):
            paired_comparison(base, treatment[:-1])


if __name__ == '__main__':
    unittest.main()
