"""Independent counterexamples; core audit never imports original model code."""
import copy
import importlib.util
import unittest
from pathlib import Path
from reproduce_beta import simulate, score_labels, aggregate, BUDGETS, partition_and_acquire


def fixture():
    return [{'item_id':f'i{i:03}', 'response_id':f'i{i:03}-r{j}', 'response':(i+j)%2}
            for i in range(81) for j in range(1+i%3)]


class Adversarial(unittest.TestCase):
    def test_repeat_trials_never_cross_pools(self):
        rows=fixture(); s,_=simulate(rows,11,'bench:subject')
        a=set(s['acquisition_items']);e=set(s['evaluation_items'])
        self.assertFalse(a&e);self.assertEqual(len(a),40);self.assertEqual(len(e),41)
        self.assertEqual(a|e,{r['item_id'] for r in rows})
        lookup={r['response_id']:r for r in rows}
        chosen=[lookup[k] for k in s['selected_response_ids']]
        self.assertEqual(len({r['item_id'] for r in chosen}),31)
        self.assertTrue(all(r['item_id'] in a for r in chosen))

    def test_outcome_leakage_and_future_labels(self):
        rows=fixture();s,m=simulate(rows,4,'b:s');evaluation=set(s['evaluation_items'])
        for mode in ['eval','future']:
            changed=copy.deepcopy(rows)
            for row in changed:
                if (mode=='eval' and row['item_id'] in evaluation) or (mode=='future' and row['response_id'] in s['selected_response_ids'][3:]):
                    row['response']=1-row['response']
            t,n=simulate(changed,4,'b:s')
            self.assertEqual(s['selected_response_ids'],t['selected_response_ids'])
            for left,right in zip(m,n):
                if mode=='eval' or left['budget']<=3:
                    self.assertEqual(left['prediction'],right['prediction'])

    def test_nested_budget_prefixes(self):
        s,m=simulate(fixture(),0,'b:s')
        for r in m:
            self.assertEqual(r['label_response_ids'],s['selected_response_ids'][:r['budget']])
            self.assertEqual(r['n_eval_responses'],len(s['eval_outcomes']))

    def test_unequal_trials_are_response_weighted(self):
        value=score_labels([1],[0,0,0,1])
        self.assertAlmostEqual(value['official_mean']['brier'],.75)
        self.assertNotEqual(value['official_mean']['brier'],.5)
        self.assertAlmostEqual(value['beta11']['brier'],13/36)

    def test_equal_pair_weight_not_response_pool(self):
        records=[]
        for pair,error,count in [('a',.1,1000),('b',.3,1)]:
            for n in BUDGETS:
                records.append(dict(pair_id=pair,seed=0,method='m',budget=n,brier=error,n_eval_responses=count))
        self.assertAlmostEqual(aggregate(records)['m']['alc_mean'],.2)

    def test_invalid_data_and_duplicate_ids_fail(self):
        for invalid in [.5,float('nan'),2,None]:
            with self.assertRaises(ValueError):score_labels([invalid],[0])
            with self.assertRaises(ValueError):score_labels([],[invalid])
        with self.assertRaises(ValueError):score_labels([],[])
        rows=fixture()
        with self.assertRaises(ValueError):partition_and_acquire(rows+[rows[0]],0,'x')
        with self.assertRaises(ValueError):partition_and_acquire([r for r in rows if r['item_id']<'i079'],0,'x')

    def test_row_order_invariance(self):
        rows=fixture();before=copy.deepcopy(rows)
        self.assertEqual(simulate(rows,19,'x'),simulate(list(reversed(rows)),19,'x'))
        self.assertEqual(rows,before)

    def test_group_identity_is_part_of_random_stream(self):
        rows=fixture()
        a=partition_and_acquire(rows,0,'benchmark_a:subject_a')
        b=partition_and_acquire(rows,0,'benchmark_a:subject_b')
        c=partition_and_acquire(rows,0,'benchmark_b:subject_a')
        self.assertNotEqual(a['selected_response_ids'],b['selected_response_ids'])
        self.assertNotEqual(a['selected_response_ids'],c['selected_response_ids'])

    def test_exact_runtime_grouping(self):
        from model import predict
        subject = {'normalized_name': 'm', 'harness': 'a'}
        target = [subject, {'benchmark_id': 'b'}]
        self.assertEqual(predict(target, []), .5)
        label = [[subject, {'benchmark_id': 'b'}], 1]
        noise = [[dict(subject, harness='other'), {'benchmark_id': 'b'}], 0]
        self.assertEqual(predict(target, [label, noise]), 2/3)
        with self.assertRaises(ValueError):
            predict(target, [[[subject, {'benchmark_id': 'b'}], .5]])

    def test_actual_runtime_matches_formula(self):
        rows = fixture()
        items = {r['item_id']: {'benchmark_id': 'b', 'item_content': r['item_id']} for r in rows}
        _, actual = simulate(rows, 7, 'b:m', {'normalized_name':'m'}, items)
        _, formula = simulate(rows, 7, 'b:m')
        for a, b in zip(actual, formula):
            self.assertAlmostEqual(a['prediction'], b['prediction'], delta=1e-15)
            self.assertAlmostEqual(a['brier'], b['brier'], delta=1e-15)

if __name__ == '__main__':
    unittest.main()
