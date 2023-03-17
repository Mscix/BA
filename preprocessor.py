# Data processing
import numpy as np
import pandas as pd
import logging

from transformers import AutoTokenizer

# Hugging Face Dataset
from datasets import Dataset
from data_sets.data_enum import DataPath
import csv
# Label Type encoding:
# 0: Unlabelled
# 1: Weakly Labeled (K-Means for ex.)
# 2: Human Label
# 3: Random Label


class Preprocessor:
    def __init__(self, path: DataPath, logger):  # pass mode?
        # Don't forget, that still have the test.csv
        df = pd.read_csv(f'/Users/misha/Desktop/Bachelor-Thesis/BA/{path.value}')
        # df = df[['Class Index', 'Description']]  # only these two columns
        df['Class Index'] = df['Class Index'] - 1  # Cross Class entropy expects [0,3] instead of [1,4]
        df['Label Type'] = 0
        self.df = df
        self.embeddings = self.df_col_to_embeddings()  # resulting in list of np.arrays each is a vector/embedding
        # Split data set into Test and Train data
        self.train_data = df.sample(frac=0.8, random_state=42)
        self.test_data = df.drop(self.train_data.index)  # TODO change name to eval!
        # Do I need test/train Embeddings?
        # Load Tokenizer from a Bert model
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

    def df_col_to_embeddings(self):
        result = []
        for i in range(len(self.df)):
            result.append(self.get_embedding(i))
        return result

    def get_embeddings(self):
        return self.embeddings

    def get_embedding(self, index):
        return np.array(eval(self.df.iloc[index]['Embedding']), np.float64)  # TODO CAREFULL EVAL
    # len(ddf.loc[ddf['Index'] == 9, 'Embedding'].values[0]) == 34393 ???

    def to_arrow_data(self):
        # This method transforms df so that Dataloader can take as input
        # TODO: possibly will need to add option to take a slice of the dataset, then provide index list?

        # Transform pandas frame into Huggingface Dataset
        hg_train_data = Dataset.from_pandas(self.train_data)
        hg_test_data = Dataset.from_pandas(self.test_data)

        dataset_train = hg_train_data.map(self.tokenize)
        dataset_test = hg_test_data.map(self.tokenize)
        # Remove unused columns from Data set
        # TODO maybe need to remove more columns
        dataset_train = dataset_train.remove_columns(['Description', '__index_level_0__', 'Label Type', 'Title'])
        dataset_test = dataset_test.remove_columns(['Description', '__index_level_0__', 'Label Type', 'Title'])
        # Change name Class Index -> labels because the model expects name labels
        dataset_train = dataset_train.rename_column("Class Index", "labels")
        dataset_test = dataset_test.rename_column("Class Index", "labels")
        # Reformat to PyTorch tensors
        dataset_train.set_format('torch')
        dataset_test.set_format('torch')

        #self.train_data = dataset_train  # <class 'datasets.arrow_dataset.Dataset'>
        #self.test_data = dataset_test
        return dataset_train, dataset_test

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

    def set_labels(self, indices: [int], labels: [int]):
        # has to know which data set is it working on
        # One column Label one column Weak/Annotator/Unlabelled (random)
        for index, value in enumerate(indices):
            # Maybe better way:
            self.df.iloc[value, self.df.columns.get_loc('Label Type')] = 1
            self.df.iloc[value, self.df.columns.get_loc('Class Index')] = labels[index]

    def get_df(self):
        return self.df

    def uncertainty_sampling(self):
        # gotta work together with evaluator
        print('scurrrrrr')

    def least_confidence_sampling(self):
        print('I am a very confident person')  # check Robert munroe,
        # My idea let model predict on unlabeled labels and the pick ls

    def diversity_sampling(self):
        print('yeeeeeee boiiii')
