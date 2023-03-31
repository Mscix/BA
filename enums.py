from enum import Enum

path_prefix = 'data_sets/the_one/'


class DataPath(Enum):
    SMALL = path_prefix + 'small_e.csv'
    # TODO: do for the rest, including downloading/creating the embeddings
    MEDIUM = path_prefix + 'medium.csv'
    BIG = path_prefix + 'big_e.csv'
    VER_BIG = path_prefix + 'very_big.csv'
    FULL = path_prefix + 'full.csv'
    CUSTOM = path_prefix + 'custom.csv'


class Mode(Enum):
    STANDARD = 0
    AL = 1
    AL_PLUS = 2
    AL_PLUS_DEV = 3
    TEST = 4




