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
        s = Sampler('cpu', 'AL', 1)
        s_plus = Sampler('cpu', 'AL+', 0.4)  # 1 - 0.4 = 0.6, all values below 0.6 are pseudo labels

        data = pd.DataFrame({"col1": range(10)}, index=range(10))

        sample_size = 5

        values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

        to_label, remaining, pseudo_labels = s.sample_by_value(data, sample_size, values)
        to_label_plus, remaining_plus, pseudo_labels_plus = s_plus.sample_by_value(data, sample_size, values)

        assert len(to_label) == sample_size
        assert len(to_label_plus) == sample_size

        # Ensure the remaining DataFrame is the correct size
        assert len(remaining) == len(data) - sample_size
        assert len(remaining_plus) == len(data) - sample_size

        # Ensure that the to_label DataFrame contains the instances with the highest values
        assert to_label.index.tolist() == [5, 6, 7, 8, 9]
        assert to_label_plus.index.tolist() == [5, 6, 7, 8, 9]

        # Check pseudo_labels for the case of 'AL+' mode
        assert len(pseudo_labels) == 0  # In the mode 'AL', no pseudo labels should be generated.
        print(pseudo_labels_plus.index.tolist())
        assert pseudo_labels_plus.index.tolist() == [0, 1, 2, 3, 4]




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
        # Prepare input tensor
        input_probs = torch.tensor([[0.2, 0.3, 0.1, 0.4], [0.1, 0.7, 0.1, 0.1]])
        # Expected output
        expected_result = torch.tensor([0.8, 0.4])
        # Compute the result
        result = Sampler.least(input_probs)
        # Check the result
        self.assertTrue(torch.allclose(result, expected_result))


    def test_margin(self):
        # Prepare input tensor
        input_probs = torch.tensor([[0.2, 0.3, 0.1, 0.4], [0.1, 0.5, 0.2, 0.2]])
        # Expected output
        expected_result = torch.tensor([0.9, 0.7])
        # Compute the result
        result = Sampler.margin(input_probs)
        # Check the result
        self.assertTrue(torch.allclose(result, expected_result))
        print(result, expected_result)


    def test_ratio(self):
        # Prepare input tensor
        input_probs = torch.tensor([[0.2, 0.3, 0.1, 0.4], [0.1, 0.5, 0.2, 0.2]])
        # Expected output
        expected_result = torch.tensor([0.75, 0.4])
        # Compute the result
        result = Sampler.ratio(input_probs)
        # Check the result
        self.assertTrue(torch.allclose(result, expected_result))
        print(result, expected_result)
