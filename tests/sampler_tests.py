import unittest
import torch
import sys
import os
import pandas as pd

project_root = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(project_root)

from sampler import Sampler


class TestUncertaintyMethods(unittest.TestCase):

    def test_sample_by_value(self):
        s = Sampler('cpu', 'AL')  # 1
        s_plus = Sampler('cpu', 'AL+')  # 1 - 0.4 = 0.6, all values below 0.6 are pseudo labels

        data = pd.DataFrame({"col1": range(10)}, index=range(10))

        sample_size = 5

        values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        predictions = [[0.17, 0.21, 0.25, 0.37],  # 3
                       [0.06, 0.34, 0.41, 0.19],  # 2
                       [0.3, 0.22, 0.29, 0.19],   # 0
                       [0.15, 0.13, 0.25, 0.47],  # 3
                       [0.18, 0.14, 0.35, 0.33],  # 2
                       [0.3, 0.11, 0.18, 0.41],   # 3
                       [0.23, 0.29, 0.18, 0.3],   # 3
                       [0.08, 0.35, 0.4, 0.17],   # 1
                       [0.32, 0.22, 0.18, 0.28],  # 0
                       [0.1, 0.29, 0.32, 0.29]]   # 2

        to_label, remaining, pseudo_labels = s.sample_by_value(data, sample_size, values, 1 - 0, predictions)
        assert len(to_label) == sample_size
        assert len(remaining) == len(data) - sample_size
        assert to_label.index.tolist() == [5, 6, 7, 8, 9]

    def test_generate_pseudo_labels(self):
        your_class_instance = Sampler('cpu', 'AL+')

        df1 = pd.DataFrame({'value': [0.6, 0.8, 0.9], 'prediction': [[0.1, 0.2, 0.3, 0.4], [0.1, 0.2, 0.3, 0.4],
                                                                     [0.1, 0.2, 0.3, 0.4]]})
        remaining1 = pd.DataFrame({'Class Index': ['a', 'b', 'c']}, index=[0, 1, 2])
        u_cap1 = 0.5
        result1 = your_class_instance.generate_pseudo_labels(df1, remaining1, u_cap1)
        assert result1.empty

        df2 = pd.DataFrame({'value': [0.2, 0.3, 0.4], 'prediction': [[0.1, 0.2, 0.3, 0.4],
                                                                     [0.1, 0.2, 0.3, 0.4],
                                                                     [0.1, 0.2, 0.3, 0.4]]})
        remaining2 = pd.DataFrame({'Class Index': [3, 3, 3]}, index=[0, 1, 2])
        u_cap2 = 0.5
        result2 = your_class_instance.generate_pseudo_labels(df2, remaining2, u_cap2)
        assert result2.equals(
            remaining2)

        df3 = pd.DataFrame({'value': [0.2, 0.6, 0.4], 'prediction': [[0.4, 0.2, 0.3, 0.1],
                                                                     [0.1, 0.2, 0.3, 0.4],
                                                                     [0.1, 0.2, 0.3, 0.4]]})

        remaining3 = pd.DataFrame({'Class Index': [0, 3, 2]}, index=[0, 1, 2])
        u_cap3 = 0.5
        result3 = your_class_instance.generate_pseudo_labels(df3, remaining3, u_cap3)
        expected3 = pd.DataFrame({'Class Index': [0]}, index=[0])
        assert result3.equals(expected3)

    def test_entropy(self):
        # Prepare input tensor
        input_probs = torch.tensor([[0.2, 0.3, 0.1, 0.4], [0.1, 0.5, 0.2, 0.2]])
        # Expected output
        expected_result = torch.tensor([0.9232, 0.8804])  # approximation
        # Compute the result
        result = Sampler.entropy(input_probs)
        # Check the result
        self.assertTrue(torch.allclose(result, expected_result, atol=0.01))

    def test_least(self):
        input_probs = torch.tensor([[0.2, 0.3, 0.1, 0.4], [0.1, 0.7, 0.1, 0.1]])
        expected_result = torch.tensor([0.8, 0.4])
        result = Sampler.least(input_probs)
        self.assertTrue(torch.allclose(result, expected_result))

    def test_margin(self):
        input_probs = torch.tensor([[0.2, 0.3, 0.1, 0.4], [0.1, 0.5, 0.2, 0.2]])
        expected_result = torch.tensor([0.9, 0.7])
        result = Sampler.margin(input_probs)
        self.assertTrue(torch.allclose(result, expected_result))
        print(result, expected_result)

    def test_ratio(self):
        input_probs = torch.tensor([[0.2, 0.3, 0.1, 0.4], [0.1, 0.5, 0.2, 0.2]])
        expected_result = torch.tensor([0.75, 0.4])
        result = Sampler.ratio(input_probs)
        self.assertTrue(torch.allclose(result, expected_result))
        print(result, expected_result)
