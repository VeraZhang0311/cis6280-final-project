import unittest
from assignment.training.planning.v11_search import choose_new_method, search_variants, compare


class SearchPipeline(unittest.TestCase):
    def test_precommitted_winner_hierarchy(self):
        full={g:dict(mean_max_placed=1.,mean_final_count=1.,decision_s_mean=.2) for g in search_variants()}
        self.assertEqual(choose_new_method(full)['variant'],'G8-32')
        full['G9']['decision_s_mean']=.1
        self.assertEqual(choose_new_method(full)['variant'],'G9')
        full['G8-96']['mean_final_count']=1.1
        self.assertEqual(choose_new_method(full)['variant'],'G8-96')
        full['G10']['mean_max_placed']=1.1
        chosen=choose_new_method(full)
        self.assertEqual(chosen['variant'],'G10')
        self.assertEqual(chosen['config']['search_mode'],'icem')

    def test_hyphenated_group_has_explicit_comparison_roles(self):
        case=dict(scene_seed=1,sample_seed=0,max_placed=1,final_count=1)
        pair=compare([case],[dict(case,max_placed=2)],'G8-96','G10')
        self.assertEqual((pair['baseline'],pair['treatment']),('G8-96','G10'))
        self.assertEqual(pair['mean_peak_delta'],1.)


if __name__=='__main__':unittest.main()
