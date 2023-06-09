from transformers import AutoTokenizer
import pandas as pd
from datasets import load_dataset
from sklearn.datasets import fetch_20newsgroups
import numpy as np


def transform_data(df):
    tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

    # print(df["Description"].tolist()[0])
    tokenized_texts = tokenizer(df["Description"].tolist(), padding='max_length', truncation=True, max_length=32)
    df['input_ids'] = tokenized_texts['input_ids']
    df['attention_mask'] = tokenized_texts['attention_mask']
    df['token_type_ids'] = tokenized_texts['token_type_ids']
    df = df.drop('Description', axis=1)
    df = df.drop('Title', axis=1)
    return df


if __name__ == "__main__":

    # Read files
    # df = pd.read_csv('/Users/misha/Desktop/Bachelor-Thesis/BA/AG_NEWS_KAGGLE/test.csv')
    # df = df.sort_values(by='Class Index')
    #print(df.head())
    #print(len(df))

    #classes = df['Class Index'].unique()
    #balanced_subset = pd.DataFrame()

    #for class_label in classes:
        # Filter the dataset to extract samples from the current class
    #    class_samples = df[df['Class Index'] == class_label].sample(n=200, random_state=42)

        # Append the extracted samples to the balanced subset
    #    balanced_subset = pd.concat([balanced_subset, class_samples])

    # Reset the index of the balanced subset
    #balanced_subset = balanced_subset.reset_index(drop=True)
    #balanced_subset.index.name = 'Index'
    #balanced_subset = transform_data(balanced_subset)

    #balanced_subset.to_csv('/Users/misha/Desktop/Bachelor-Thesis/BA/data_sets/the_one/big_t_test.csv')

    #dataset = load_dataset('glue', 'sst2')
    # Write fiels
    # data = pd.read_csv('/Users/misha/Desktop/Bachelor-Thesis/BA/data_sets/the_one/very_big.csv', index_col='Index')
    #data = transform_data(data)
    # data.to_csv('/Users/misha/Desktop/Bachelor-Thesis/BA/data_sets/the_one/final_train.csv')
    data = pd.read_csv('/Users/misha/Desktop/Bachelor-Thesis/BA/data_sets/the_one/test_f.csv')
    data = transform_data(data)
    data.index.name = 'Index'
    data.to_csv('/Users/misha/Desktop/Bachelor-Thesis/BA/data_sets/the_one/final_test.csv')
