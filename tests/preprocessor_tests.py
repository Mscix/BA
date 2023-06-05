import unittest
import sys
import os
import pandas as pd


project_root = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(project_root)

from preprocessor import Preprocessor  # replace with the actual module name


class TestPreprocessor(unittest.TestCase):

    def setUp(self):
        self.preprocessor = Preprocessor('/Users/misha/Desktop/Bachelor-Thesis/BA/data_sets/the_one/small_t.csv', 'cpu')

    def test_init(self):
        # Test if initial data frame is loaded properly
        self.assertIsInstance(self.preprocessor.df, pd.DataFrame)
        self.assertEqual(self.preprocessor.df['Class Index'].min(), 0)
        self.assertEqual(self.preprocessor.df['Class Index'].max(), 3)
        # Check that the Test, Validation split works, 80%, 20%
        total_len = len(self.preprocessor.train_data) + len(self.preprocessor.eval_data)
        self.assertEqual(total_len * 0.8, len(self.preprocessor.train_data))
