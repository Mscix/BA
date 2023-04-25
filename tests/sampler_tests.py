import pytest
import unittest
import sampler
import pandas as pd


def ec_test():
    s = sampler.Sampler.sample_by_value
    df = pd.DataFrame({})

