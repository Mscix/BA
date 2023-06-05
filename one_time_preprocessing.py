from transformers import AutoTokenizer
import pandas as pd


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

    # Write fiels
    data = pd.read_csv('/Users/misha/Desktop/Bachelor-Thesis/BA/data_sets/the_one/medium.csv', index_col='Index')
    data = transform_data(data)
    data.to_csv('/Users/misha/Desktop/Bachelor-Thesis/BA/data_sets/the_one/medium_t.csv')

    # data = pd.read_csv('/Users/misha/Desktop/Bachelor-Thesis/BA/data_sets/the_one/big_e.csv', index_col='Index')
    # data = transform_data(data)
    # data.to_csv('/Users/misha/Desktop/Bachelor-Thesis/BA/data_sets/the_one/big_t.csv')
