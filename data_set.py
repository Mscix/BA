from torch.utils.data.dataset import T_co

from data_sets.data_enum import DataPath
import numpy as np
import pandas as pd
import spacy
nlp = spacy.load('en_core_web_sm')  # Loads small NLP spacy model
from torchtext.vocab import build_vocab_from_iterator
from torch.utils.data import Dataset, DataLoader
import torch

# PREPROCESSING STATIC METHODS

def vectorize(tokens):
    """"""
    return [nlp(token).vector for token in tokens]


def tokenize(text):
    """"""
    return [t.text for t in nlp(text)]


class DataSet(Dataset):

    def __init__(self, d_set: DataPath = DataPath.SMALL):
        self.df = pd.read_csv(d_set.value)

    def __getitem__(self, index):
        # convert to np array
        # np array to pytorch tensor
        # adjusted for my csv file
        # Index, Class Index, Title, Description
        x = self.df.loc[index, 'Title':'Description'].to_numpy()
        x = torch.tensor(x, dtype=torch.float)
        y = torch.tensor(self.df.loc[index, 'Class Index'])
        return x, y

    def __len__(self):
        len(self.df)

    def add_column(self, column_name, init_val):
        self.df.insert(1, column_name, [init_val for n in range(len(self.df))])  # not sure why warning is thrown

    def remove_column(self, cols):
        """Takes list of columns that should be removed."""
        self.df = list(set(self.df.columns) - set(cols))

    def random_sample(self,  n: int = 10):
        # TODO: make more efficient
        # Pick the ones where unlabelled
        res = self.df[~self.df['Labelled']]
        # indices = res.sample(frac=1)['Index'][:n]  deprecated
        indices = res.sample(frac=1)['Index'].iloc[:n]  # shuffle and take n first instances
        if n > len(res):  # N too big
            print("N bigger then unlabelled instances length.")
            return []
        self.set_to_labelled(indices)

    # HELPER/ANNOTATOR
    def set_to_labelled(self, indices: [int]):
        for i in indices:
            # find by index but change the Labelled value to True
            self.df.loc[self.df['Index'] == i, 'Labelled'] = True

    def get_labelled_instances(self):
        return self.df[self.df['Labelled']]

