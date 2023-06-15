import unittest
import torch
import sys
import os
import pandas as pd

project_root = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(project_root)

from labeler import StrongLabeller, WeaklyLabeller, CustomLabeller
from preprocessor import Preprocessor


class TestWeakLabeller(unittest.TestCase):

    def setUp(self):
        self.control = pd.DataFrame({
            'Index': ['1', '2', '3', '4', '5'],
            'Class Index': ['A', 'B', 'C', 'D', 'E']
        })
        self.subset = pd.DataFrame({
            'Index': ['1', '2', '3', '4', '5'],
            'Class Index': ['A', 'B', 'C', 'X', 'X']
        })
        self.error_calculator = WeaklyLabeller()

    def test_calc_error(self):
        error = self.error_calculator.calc_error(self.control, self.subset)
        # There are two differing 'Class Index' values in 'subset' compared to 'control'
        self.assertEqual(error, 2, "How many classes mismatch")

    def test_calc_error_with_none(self):
        error = self.error_calculator.calc_error(self.control, None)
        # If 'subset' is None the error should be 0
        self.assertEqual(error, 0, "Check if works with None")

    def test_custom_labelling(self):
        error_rate = 0.25
        error_rate_2 = 0.3
        # Will break if this test is run on another system
        data = Preprocessor('AG_NEWS_KAGGLE/small.csv', 'AG_NEWS_KAGGLE/small_test.csv', 'cpu')
        w = CustomLabeller(error_rate, data.control)
        w2 = CustomLabeller(error_rate, data.control)
        labelled_25 = w.label(data.control)
        labelled_30 = w2.label(data.control)
        # the method calc_error is tested in the test above :)
        self.assertTrue(WeaklyLabeller.calc_error(data.control, labelled_25), error_rate)
        self.assertTrue(WeaklyLabeller.calc_error(data.control, labelled_30), error_rate_2)


class TestStrongLabeler(unittest.TestCase):

    def setUp(self):
        self.control = pd.DataFrame({
            'Value': ['A', 'B', 'C', 'D', 'E'],
            'Class Index': ['0', '1', '2', '3', '0']
        }, index=['0', '1', '2', '3', '4'])

        self.to_label = pd.DataFrame({
            'Value': ['D', 'E'],
            'Class Index': ['1', '1']
        }, index=['3', '4'])

        self.strong_labeller = StrongLabeller(self.control)

    def test_label(self):
        labelled_data = self.strong_labeller.label(self.to_label)
        # The values at indices '3', '4', '5' should be replaced with 'C', 'D', 'E' respectively
        # The values at indices '6', '7' should remain None
        expected_result = pd.DataFrame({
            'Value': ['D', 'E'],
            'Class Index': ['3', '0']
        }, index=['3', '4'])

        pd.testing.assert_frame_equal(labelled_data, expected_result)

