# Data processing
import pandas as pd
import logging

from transformers import AutoTokenizer

# Hugging Face Dataset
from datasets import Dataset
from data_sets.data_enum import DataPath


class Preprocessor:
    def __init__(self, path: DataPath, logger):  # pass mode?
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
        self.train_data = dataset_train  # <class 'datasets.arrow_dataset.Dataset'>
        self.test_data = dataset_test

    def tokenize(self, data):
        # TODO: apply to Title as well later on and adjust max length maybe? Or create another tokenizer for Title.
        return self.tokenizer(data["Description"], padding='max_length', truncation=True, max_length=32)

    def get_train_data(self):
        return self.train_data

    def get_test_data(self):
        return self.test_data

    def get_random_sample(self, sample_size):
        # if sample_size float then % of data set that should be test set!
        # if int then absolute number of test samples
        # Shuffle the dataset
        # TODO: throw exception when larger or return remaining instances and that's it
        # data set is shuffled by default
        ds_dict = self.train_data.train_test_split(test_size=sample_size)
        remaining, sampled = ds_dict['train'], ds_dict['test']
        print('SAMPLING FROM ' + str(len(self.train_data)))
        print('Remaining ' + str(len(remaining)))
        print('Sampled ' + str(len(sampled)))
        self.train_data = remaining
        return sampled

    def uncertainty_sampling(self):
        # gotta work together with evaluator
        print('scurrrrrr')

    def least_confidence_sampling(self):
        print('I am a very confident person')  # check Robert munroe,
        # My idea let model predict on unlabeled labels and the pick ls

    def diversity_sampling(self):
        print('yeeeeeee boiiii')

    """
    def set_to_labelled(self, indices: [int]):
        for i in indices:
            # find by index but change the Labelled value to True
            self.df.loc[self.df['Index'] == i, 'Labelled'] = True

    def get_labelled_instances(self):
        return self.df[self.df['Labelled']]


    def add_column(self, column_name, init_val):
        self.df.insert(1, column_name, [init_val for n in range(len(self.df))])  # not sure why warning is thrown

    def remove_column(self, cols):
        self.df = list(set(self.df.columns) - set(cols))
    """
