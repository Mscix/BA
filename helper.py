# Transform DataFrame
# TODO: Useless file?

def set_to_labelled(df, indices: [int]):
    for i in indices:
        # find by index but change the Labelled value to True
        df.loc[df['Index'] == i, 'Labelled'] = True
    return df


def get_labelled_instances(df):
    return df[df['Labelled']]


def add_column(df, column_name, init_val):
    df.insert(1, column_name, [init_val for n in range(len(df))])  # not sure why warning is thrown
    return df


def remove_column(df, cols):
    df = list(set(df.columns) - set(cols))
    return df