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
    STANDARD = 'Standard Machine Learning'
    AL = 'Active Learning'
    AL_PLUS = 'Active Learning Plus'
    TEST = 'Testing Grounds'
    AL_PLUS_DEV = 'Active Learning Plus in Development'


class EvalSet(Enum):
    # What set should the Model be evaluated on.
    TRAINING_SET = 'Training Set'  # Used for training the model
    EVALUATION_SET = 'Evaluation Set'  # The complete Evaluation set should be untouched until after model training
    TEST_SET = 'Test Set'  # Subset of Evaluation set
    VALIDATION_SET = 'Validation Set'  # Used for training and fine-tuning the model




