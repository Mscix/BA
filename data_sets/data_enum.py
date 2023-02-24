from enum import Enum

# class syntax

class DataPath(Enum):
    pre = 'data_sets/the_one/'
    SMALL = pre + 'small.csv'
    MEDIUM = pre + 'medium.csv'
    BIG = pre + 'big.csv'
    VER_BIG = pre + 'very_big.csv'
    FULL = pre + 'full.csv'
    CUSTOM = pre + 'custom.csv'




