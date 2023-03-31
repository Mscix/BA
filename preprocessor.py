# Data processing
import numpy as np
import pandas as pd
import logging
from transformers import AutoTokenizer
# Hugging Face Dataset
from datasets import Dataset
from enums import DataPath
import random
# Label Type encoding:
# 0: Unlabelled
# 1: Weakly Labeled (K-Means for ex.)
# 2: Human Label
# 3: Random Label


class Preprocessor:
    def __init__(self, path: DataPath):  # pass mode?
        self.logger = logging.getLogger(__name__)
        # Don't forget, that still have the test.csv
        # set the index_col as Index, which is the custom index in relation to the whole set
        df = pd.read_csv(f'/Users/misha/Desktop/Bachelor-Thesis/BA/{path.value}', index_col='Index')
        self.control = df
        # df = df[['Class Index', 'Description']]  # only these two columns
        df['Class Index'] = df['Class Index'] - 1  # Cross Class entropy expects [0,3] instead of [1,4]
        self.df = df
        # Split Training set 80%, Test set 10%, Validation set 10%
        self.train_data = df.sample(frac=0.8, random_state=42)
        temp = df.drop(self.train_data.index)
        self.test_data = temp.sample(frac=0.5, random_state=42)
        self.eval_data = temp.drop(self.test_data.index)

        # Load Tokenizer from a Bert model
        model = "bert-base-cased"
        self.tokenizer = AutoTokenizer.from_pretrained(model)

        # This is how the data is split later on
        self.labelled = None
        self.weakly_labelled = None
        self.unlabelled = self.train_data

        self.logger.info('DataSet size used: ' + path.value)
        self.logger.info('Tokenizer model used: ' + model)

    def to_arrow_data(self, df):
        # This method transforms df so that Dataloader can take as input
        # Transform pandas frame into Huggingface Dataset
        hg_train_data = Dataset.from_pandas(df)
        dataset_train = hg_train_data.map(self.tokenize)
        # Remove unused columns from Data set
        dataset_train = dataset_train.remove_columns(['Description', 'Embedding', 'Index', 'Title'])
        # Change name Class Index -> labels because the model expects name labels
        dataset_train = dataset_train.rename_column("Class Index", "labels")
        # Reformat to PyTorch tensors
        dataset_train.set_format('torch')
        # Returns <class 'datasets.arrow_dataset.Dataset'>
        return dataset_train

    def tokenize(self, data):
        return self.tokenizer(data["Description"], padding='max_length', truncation=True, max_length=32)

    def get_random_sample(self, sample_size):
        # if sample_size float then % of data set that should be test set!
        # if int then absolute number of test samples
        # Shuffle the dataset
        # TODO: throw exception when larger or return remaining instances and that's it
        # data set is shuffled by default
        ds_dict = self.train_data.train_test_split(test_size=sample_size)
        remaining, sampled = ds_dict['train'], ds_dict['test']

        self.train_data = remaining
        logging.info('SAMPLING FROM: ' + str(len(self.train_data)) + '\nRemaining: ' + str(len(remaining))
                     + '\nSampled: ' + str(len(sampled)))
        return sampled, remaining

    @staticmethod
    def get_first_by_class(df, class_id):
        # Gets first instance that is assigned to the given class
        return df.loc[df['Class Index'] == class_id].iloc[0]

    @staticmethod
    def get_first_reps_4_class(df, keep=True):
        # TODO: maybe dangerous to return 2 different types... however convenient
        # TODO: the Problem I discussed with Fabian can happen here lol add Exception handler or something
        # maybe for now set manually ...
        # This is not a random method but after fixed labelling when used it should be, because of random sampling
        centroids = []
        to_drop = []
        logging.info('get_random_centroids with keep=' + str(keep))
        for class_id in range(4):
            # row = df.loc[df['Class Index'] == class_id].iloc[0]  # is this reliable?
            row = Preprocessor.get_first_by_class(df, class_id)
            centroids.append(np.array(eval(row['Embedding'])))
            to_drop.append(row.name)  # gets the index of specified row
            logging.info('Centroid for Class ID: ' + str(class_id) + '\n' + row.to_string())
        if keep:
            return centroids
        else:
            _df = df.drop(to_drop)
            return centroids, _df

    @staticmethod
    def get_embeddings_from_df(df):
        # Takes the Embedding column makes a list evaluates each and then transforms each vector into np.array
        return list(map(lambda n: np.array(eval(n)), df['Embedding'].tolist()))

    @staticmethod
    def get_embedding(df, index):
        return np.array(eval(df.iloc[index]['Embedding']), np.float64)  # TODO CARE EVAL

    def get_df(self):
        return self.df

    def uncertainty_sampling(self):
        pass

    def least_confidence_sampling(self):
        pass

    def diversity_sampling(self):
        pass
