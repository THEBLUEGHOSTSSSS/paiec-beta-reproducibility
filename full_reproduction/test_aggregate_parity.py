"""Numerical comparison regressions, using the declared scientific dependencies."""
import unittest
import numpy as np
from engine import array_error, compare_tree


class AggregateParityTests(unittest.TestCase):
    def test_nested_nonfinite_values_on_either_side_are_rejected(self):
        for value in (float('nan'), float('inf'), -float('inf')):
            for left, right in ((value, 1.), (1., value), (value, value)):
                with self.subTest(left=left, right=right):
                    with self.assertRaisesRegex(ValueError, 'non-finite'):
                        compare_tree({'outer':[0., {'loss':left}]},
                                     {'outer':[0., {'loss':right}]})

    def test_finite_nested_maximum_is_retained(self):
        self.assertEqual(compare_tree({'scores':[0., .375]}, {'scores':[0., .25]}), .125)

    def test_overflowing_difference_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'non-finite'):
            compare_tree(1.7e308, -1.7e308)

    def test_array_reduction_rejects_an_invalid_cell_on_either_side(self):
        baseline = np.array([0., 1.])
        for value in (float('nan'), float('inf'), -float('inf')):
            changed = np.array([0., value])
            for left, right in ((changed, baseline), (baseline, changed)):
                with self.subTest(value=value):
                    with self.assertRaisesRegex(ValueError, 'non-finite'):
                        array_error(left, right)
        self.assertEqual(array_error(baseline, np.array([0.25, 1.])), .25)

    def test_array_shape_mismatch_cannot_broadcast(self):
        with self.assertRaisesRegex(ValueError, 'shapes'):
            array_error(np.array([0., 1.]), np.array([0.]))
