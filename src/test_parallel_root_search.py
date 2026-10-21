import os
os.environ.setdefault('WOTR_ACT_GEN','v52')
os.environ.setdefault('WOTR_NATIVE','0')
import copy
import unittest
from types import SimpleNamespace
import numpy as np
from parallel_root_search import aggregate,split_budget

def action(target):
    return SimpleNamespace(action_type='combat_effect_choice',die_idx=-1,params={'target':target})

def record(actions,visits,q):
    n=len(actions)
    return dict(root_actions=actions,visit_counts=np.array(visits),q_values=np.array(q),
        q_var=np.zeros(n),term_frac=np.zeros(n),priors_clean=np.full(n,1/n),proven=np.zeros(n),
        proven_value=np.zeros(n),forced=np.zeros(n),conf=np.ones(n),phi=np.zeros(n),
        action_encs=np.zeros((n,133)),region_indices=np.full(n,-1))

class RootAggregation(unittest.TestCase):
    def test_budget_exact_and_nonzero(self):
        self.assertEqual(split_budget(65,2),[33,32])
        self.assertEqual(split_budget(5,5),[1]*5)
        for args in [(1,2),(0,1),(3,0),(True,1),(3,True)]:
            with self.assertRaises(ValueError):split_budget(*args)

    def test_semantic_aliases_stay_distinct_and_q_is_visit_weighted(self):
        a,b=action('nazgul'),action('Witch King')
        first=record([a,b],[9,1],[1,-1]);second=record([b,a],[3,1],[1,-1])
        result=aggregate([first,second],[a,b])
        self.assertEqual(result['visit_counts'].tolist(),[10,4])
        np.testing.assert_allclose(result['q_values'],[.8,.5])
        np.testing.assert_allclose(result['q_var'],[.36,.75])
        self.assertEqual(len(result['root_actions']),2) # Encodings intentionally alias.

    def test_missing_or_conflicting_proof_never_becomes_proven(self):
        a,b=action('nazgul'),action('Witch King')
        first=record([a,b],[2,2],[1,1]);second=record([a],[2],[1])
        first['proven'][:]=1;first['proven_value'][:]=1
        second['proven'][:]=1;second['proven_value'][:]=1
        result=aggregate([first,second],[a,b])
        self.assertEqual(result['proven'].tolist(),[1,0])
        second['proven'][:]=-1
        self.assertEqual(aggregate([first,second],[a,b])['proven'].tolist(),[0,0])

    def test_pooled_population_variance_matches_literal_backups_not_sem(self):
        a=action('nazgul')
        samples=[np.array([-.5,.25,1.0]),np.array([-.25,-.25,.75,1.0,1.0])]
        records=[]
        for values in samples:
            item=record([a],[len(values)],[values.mean()])
            item['q_var'][0]=np.mean(values*values)-values.mean()**2
            records.append(item)
        result=aggregate(records,[a]);all_values=np.concatenate(samples)
        self.assertAlmostEqual(result['q_values'][0],all_values.mean())
        self.assertAlmostEqual(result['q_var'][0],np.var(all_values,ddof=0))
        self.assertNotAlmostEqual(result['q_var'][0],np.var(all_values,ddof=0)/len(all_values))

    def test_illegal_duplicate_nonfinite_zero_visit_and_mismatched_encoding_refused(self):
        a,b=action('nazgul'),action('Witch King');base=record([a],[2],[.5])
        cases=[]
        cases.append(record([b],[2],[1]))
        cases.append(record([a,a],[1,1],[1,1]))
        bad=copy.deepcopy(base);bad['q_values'][0]=np.nan;cases.append(bad)
        bad=copy.deepcopy(base);bad['visit_counts'][0]=-1;cases.append(bad)
        cases.append(record([a],[0],[0]))
        for bad in cases:
            with self.assertRaises(ValueError):aggregate([bad],[a])
        bad=copy.deepcopy(base);bad['action_encs'][0,0]=1
        with self.assertRaises(ValueError):aggregate([base,bad],[a])

if __name__=='__main__':unittest.main()
