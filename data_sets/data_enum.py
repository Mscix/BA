from enum import Enum



class DataPath(Enum):
    # Should not only Decide on the csv file but also embeddings file mabe pass a tuple?
    pre = 'data_sets/the_one/'
    SMALL = pre + 'small_e.csv'
    # TODO: do for the rest, including downloading/creating the embeddings
    MEDIUM = pre + 'medium.csv'
    BIG = pre + 'big.csv'
    VER_BIG = pre + 'very_big.csv'
    FULL = pre + 'full.csv'
    CUSTOM = pre + 'custom.csv'




