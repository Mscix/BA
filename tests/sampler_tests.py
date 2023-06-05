import unittest
import torch
import sys
import os

project_root = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(project_root)

from sampler import Sampler

class TestUncertaintyMethods(unittest.TestCase):


    def test_sample_by_value(self):
        s = Sampler('cpu', 'AL', 1)
        s_plus = Sampler('cpu', 'AL+', 0.4)




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
