# Data processing
import numpy as np
import pandas as pd
from transformers import AutoTokenizer
# Hugging Face Dataset
from datasets import Dataset
from torch.utils.data import DataLoader
from ast import literal_eval


def to_data_loader(df, device):
    data = Dataset.from_pandas(df)

    data = data.remove_columns(['Embedding', 'Index'])

    # Change name Class Index -> labels because the model expects name labels
    data = data.rename_column("Class Index", "labels")

    # Reformat to PyTorch tensors
    data.set_format('torch')
    if device == 'cuda':
        # batch_size = 256
        # batch_size = 128
        data = DataLoader(dataset=data, batch_size=64)
    elif device == 'prediction':
        data = DataLoader(dataset=data, batch_size=1, shuffle=False)
    else:
        data = DataLoader(dataset=data, batch_size=2)

    if device == 'cuda':
        for batch_id, sample in enumerate(data):
            labels = sample['labels'].to('cuda')
            input_ids = sample['input_ids'].to('cuda')
            token_type_ids = sample['token_type_ids'].to('cuda')
            attention_mask = sample['attention_mask'].to('cuda')
    return data


def get_first_reps_4_class(df, keep=True):
    # TODO: the Problem I discussed with Fabian can happen here lol add Exception handler or something
    # maybe for now set manually ...
    # This is not a random method but after fixed labelling when used it should be, because of random sampling
    centroids = []
    to_drop = []
    for class_id in range(4):
        row = get_first_by_class(df, class_id)
        centroids.append(np.array(eval(row['Embedding'])))
        to_drop.append(row.name)  # gets the index of specified row
    if keep:
        return centroids
    else:
        _df = df.drop(to_drop)
        return centroids, _df


def get_first_by_class(df, class_id):
    # Gets first instance that is assigned to the given class
    return df.loc[df['Class Index'] == class_id].iloc[0]


def get_embeddings_from_df(df):
    # Takes the Embedding column makes a list evaluates each and then transforms each vector into np.array
    return list(map(lambda n: np.array(eval(n)), df['Embedding'].tolist()))


def get_embedding(df, index):
    # return np.array(eval(df.iloc[index]['Embedding']), np.float64) # TODO CARE EVAL
    return np.array(literal_eval(df.iloc[index]['Embedding']), np.float64)


class Preprocessor:
    def __init__(self, path: str, device):  # pass mode?
        # Don't forget, that still have the test.csv
        # set the index_col as Index, which is the custom index in relation to the whole set
        self.device = device
        df = pd.read_csv(path, index_col='Index')

        # df = df[['Class Index', 'Description']]  # only these two columns
        df['Class Index'] = df['Class Index'] - 1  # Cross Class entropy expects [0,3] instead of [1,4]
        df['input_ids'] = df['input_ids'].apply(literal_eval)
        df['attention_mask'] = df['attention_mask'].apply(literal_eval)
        df['token_type_ids'] = df['token_type_ids'].apply(literal_eval)
        self.control = df
        self.df = df
        # Split Training set 80%,  Validation set 20% (Only Two split)
        self.train_data = df.sample(frac=0.8)
        self.eval_data = df.drop(self.train_data.index)

        # Load Tokenizer from a Bert model
        model = "bert-base-cased"
        self.tokenizer = AutoTokenizer.from_pretrained(model)

        # This is how the data is split later on
        self.labelled = None
        self.partial = self.train_data

    def get_df(self):
        return self.df
