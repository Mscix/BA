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


def create_subset(df, size_per_class, file_name):
    subset_df = df.groupby('Class Index').head(size_per_class)

    subset_df = subset_df.reset_index().rename(columns={'index': 'Index'})

    subset_df = transform_data(subset_df)

    subset_df.to_csv('AG_NEWS_KAGGLE/' + file_name, index=False)


if __name__ == "__main__":
    train_df = pd.read_csv('AG_NEWS_KAGGLE/train.csv')

    df = train_df.sort_values('Class Index')

    create_subset(df, 10, 'small.csv')
    create_subset(df, 1000, 'medium.csv')
    create_subset(df, 10000, 'large.csv')
