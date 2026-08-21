import unittest

from preprocess import _summarize_samples


class SummarizeSamplesTests(unittest.TestCase):
    def test_seq_len_dist_and_failure_counts(self):
        samples = [
            {'digit_sequence': ['1'] * 8},
            {'digit_sequence': ['2'] * 7},
            {'digit_sequence': ['3'] * 7},
        ]
        failed = [('a', 'video not found'), ('b', 'video not found'), ('c', 'empty alignment')]
        summary = _summarize_samples(samples, failed)
        self.assertEqual(summary['seq_len_dist'], {'7': 2, '8': 1})
        self.assertEqual(summary['failure_counts'],
                         {'video not found': 2, 'empty alignment': 1})

    def test_empty_inputs(self):
        self.assertEqual(_summarize_samples([], []),
                         {'seq_len_dist': {}, 'failure_counts': {}})


if __name__ == '__main__':
    unittest.main()
