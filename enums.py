from enum import Enum

path_prefix = 'data_sets/the_one/'


class Data(Enum):
    SMALL = {'path': path_prefix + 'small_e.csv', 'size': 'small=40'}
    # TODO: do for the rest, including downloading/creating the embeddings
    MEDIUM = {'path': path_prefix + 'medium.csv', 'size': 'medium=400'}
    BIG = {'path': path_prefix + 'big_e.csv', 'size': 'big=4000'}
    VER_BIG = {'path': path_prefix + 'very_big.csv', 'size': 'very big=40000'}
    FULL = {'path': path_prefix + 'full.csv', 'size': 'full=120000'}
    # CUSTOM = path_prefix + 'custom.csv'


class Mode(Enum):
    STANDARD = 'Standard Machine Learning'
    AL = 'Active Learning'
    AL_PLUS = 'Active Learning Plus'
    TEST = 'Testing Grounds'


class EvalSet(Enum):
    # What set should the Model be evaluated on.
    TRAINING_SET = 'Training Set'  # Used for training the model
    EVALUATION_SET = 'Evaluation Set'  # The complete Evaluation set should be untouched until after model training
    TEST_SET = 'Test Set'  # Subset of Evaluation set
    VALIDATION_SET = 'Validation Set'  # Used for training and fine-tuning the model

class SamplingMethod(Enum):
    RANDOM = 'Random'
    LC = 'Least Confidence'
    # TODO add more
