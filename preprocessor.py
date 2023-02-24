# Data processing
import pandas as pd
from torch.utils.data import DataLoader

from transformers import AutoTokenizer

# Hugging Face Dataset
from datasets import Dataset
from data_sets.data_enum import DataPath


class Preprocessor:
    def __init__(self, path: DataPath):
        # Don't forget, that still have the test.csv
        df = pd.read_csv(f'/Users/misha/Desktop/Bachelor-Thesis/BA/{path.value}')
        df = df[['Class Index', 'Description']]  # only these two columns
        df['Class Index'] = df['Class Index'] - 1  # Cross Class entropy expects [0,3] instead of [1,4]
        #  Defines a datatype together with instructions for converting to Tensor.
        train_data = df.sample(frac=0.8, random_state=42)  # random_state is a seed aka pseudo random num generator
        test_data = df.drop(train_data.index)
        # Transform pandas frame into Huggingface Dataset
        hg_train_data = Dataset.from_pandas(train_data)
        hg_test_data = Dataset.from_pandas(test_data)
        # Tokenizer from a Bert model
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
        dataset_train = hg_train_data.map(self.tokenize)
        dataset_test = hg_test_data.map(self.tokenize)
        # Remove unused columns from Data set
        dataset_train = dataset_train.remove_columns(['Description', '__index_level_0__'])
        dataset_test = dataset_test.remove_columns(['Description', '__index_level_0__'])
        # Change name Class Index -> labels because the model expects name labels
        dataset_train = dataset_train.rename_column("Class Index", "labels")
        dataset_test = dataset_test.rename_column("Class Index", "labels")
        # Reformat to PyTorch tensors
        dataset_train.set_format('torch')
        dataset_test.set_format('torch')
        self.train_data = dataset_train
        self.test_data = dataset_test

    # Train on Description first Description
    def tokenize(self, data):
        return self.tokenizer(data["Description"], padding='max_length', truncation=True, max_length=32)

    def get_train_data(self):
        return self.train_data

    def get_test_data(self):
        return self.test_data

    #def get_random_sample(self):
