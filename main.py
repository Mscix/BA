#!/usr/bin/env python
# -*- coding: utf-8 -*-
from trainer import Trainer
from preprocessor import Preprocessor, to_data_loader
from evaluator import Evaluator
from transformers import AutoModelForSequenceClassification
import torch
from labeler import KMeansLabeller, StrongLabeller, CustomLabeller, WeaklyLabeller
from sampler import Sampler
import pandas as pd
import wandb
import argparse
import copy
import warnings
import logging


class Main:
    def __init__(self,
                 path_train,
                 path_test,
                 mode,
                 sampling_method,
                 weakly_error=0.25,
                 al_iterations=3,
                 init_sample_size=0.05,
                 n_sample_size=0.01,
                 resetting_model=False,
                 patience=3,
                 delta=1,
                 delta_rate=0.0
                 ):

        logging.getLogger('transformers.modeling_utils').setLevel(logging.ERROR)
        # Load model
        model_name = "bert-base-cased"
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=4)
        if torch.cuda.is_available():
            print('CUDA is used.')
            self.device = torch.device('cuda')
        else:
            print('CPU is used')
            self.device = torch.device('cpu')
        # Empty cache
        torch.cuda.empty_cache()
        self.data = Preprocessor(path_train, path_test, self.device.type)

        self.mode = mode
        # weak_labeler = 'K-Means'

        self.fixed_centroids = True
        # Following if-statement is only important if using K-Means
        #   If Data sample too small there exists a possibility that there might not be a representative of
        #   each class. This will lead to problems when picking centroids. Therefor make the choice fixed.
        if len(self.data.train_data) > 3000:
            self.fixed_centroids = False
            # Set data Labels to ambiguous number but only if fixed_centroids = false

        self.strong_labeler = StrongLabeller(self.data.control)
        self.weak_labeler = CustomLabeller(weakly_error, self.data.control)

        self.model.to(self.device)

        self.valid_dataloader = to_data_loader(self.data.eval_data, self.device.type)
        self.test_dataloader = to_data_loader(self.data.test_data, self.device.type)
        self.evaluator = Evaluator(self.device, self.valid_dataloader, self.test_dataloader)
        self.trainer = Trainer(self.model, self.device, self.evaluator, resetting_model,
                               copy.deepcopy(self.model.state_dict()), patience)
        self.sampler = Sampler(self.device, mode)
        self.sampling_method = sampling_method
        self.mode = mode
        self.weakly_error = weakly_error
        self.delta = delta
        self.delta_rate = delta_rate

        self.hyperparameters = {
            'Mode': mode,
            'Weakly Error': 0 if mode != 'AL+' and mode != 'ALI' else weakly_error,
            'Data Set': 'AG_NEWS',
            'Train Set': len(self.data.train_data),
            'Sampling Method': sampling_method,
            'Init Sample Size': init_sample_size,
            'N-Sample': n_sample_size,
            'Reset Model': resetting_model,
            'AL Iterations': al_iterations,
            'Patience': patience,
            'P. Label Conf.': delta,
            'Delta Change': delta_rate
        }

    def run(self):
        if self.mode == 'Standard':
            self.standard_ml(self.hyperparameters)
        elif self.mode == 'Dev':
            print('Testing grounds')
        else:
            # Default
            self.al(self.hyperparameters)

    def standard_ml(self, hyperparameters):
        with wandb.init(project='active-learning-plus', config=hyperparameters):
            init_sample_size = hyperparameters['Init Sample Size']
            init_sample, _, _ = self.sampler.sample(self.data, self.data.partial, init_sample_size)
            self.data.labelled = self.strong_labeler.label(init_sample)
            train_dataloader = to_data_loader(self.data.labelled, self.device.type)
            self.trainer.train(train_dataloader, self.data, None, 0)

    def al(self, hyperparameters):
        with wandb.init(project='active-learning-plus', config=hyperparameters):
            init_sample_size = hyperparameters['Init Sample Size']
            sample_size = hyperparameters['N-Sample']
            al_iterations = hyperparameters['AL Iterations']
            pseudo_labels_len = 0
            print('AL Iteration: 0')
            init_sample, self.data.partial, _ = self.sampler.sample(self.data, self.data.partial, init_sample_size)
            self.data.labelled = self.strong_labeler.label(init_sample)
            # Part of initial implementation:
            # self.weak_labeler = KMeansLabeller(self.data, self.fixed_centroids)
            # --------------- AL PLUS --------------- #
            if self.mode == 'AL+':
                # Initially trains on all Samples
                self.data.partial = self.weak_labeler.label(self.data.partial)
                train_set = self.data.labelled
            elif self.mode == 'ALI':
                self.data.partial = self.weak_labeler.label(self.data.partial)
                train_set = pd.concat([self.data.labelled, self.data.partial])
            else:
                train_set = self.data.labelled
            # --------------- AL PLUS --------------- #
            train_dataloader = to_data_loader(train_set, self.device.type)
            self.trainer.train(train_dataloader, self.data, pd.DataFrame(), 0)

            for i in range(al_iterations):
                print(f'AL Iteration: {i + 1}')
                sample, self.data.partial, pseudo_labels = self.sampler.sample(preprocessor=self.data,
                                                                               data=self.data.partial,
                                                                               sample_size=sample_size,
                                                                               sampling_method=self.sampling_method,
                                                                               model=self.trainer.model,
                                                                               u_cap=1 - self.delta
                                                                               )
                self.data.labelled = pd.concat([self.data.labelled, self.strong_labeler.label(sample)])
                # --------------- AL PLUS --------------- #
                if self.mode == 'AL+':
                    train_set = pd.concat([self.data.labelled, pseudo_labels])
                    if len(pseudo_labels) > pseudo_labels_len:
                        self.delta += self.delta_rate
                        print(f'Current Delta: {self.delta}')
                        pseudo_labels_len = len(pseudo_labels)
                else:
                    train_set = self.data.labelled
                    pseudo_labels = pd.DataFrame()
                # --------------- AL PLUS --------------- #
                train_dataloader = to_data_loader(train_set, self.device.type)
                self.trainer.train(train_dataloader, self.data, pseudo_labels, i + 1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process passed Hyperparameters.')

    parser.add_argument('-p', '--path', type=str, help='Path to the csv file with the data set.',
                        default='AG_NEWS_KAGGLE/small.csv')

    parser.add_argument('-tp', '--test_path', type=str, help='Path to the csv file with the test set.',
                        default='AG_NEWS_KAGGLE/small_test.csv')

    parser.add_argument('-m', '--mode', type=str, choices=['AL', 'AL+', 'ALI', 'Standard', 'Dev'], default='AL+',
                        help='The Learning mode.')

    parser.add_argument('-sm', '--sampling_method', type=str, choices=['Random', 'EC', 'LC', 'MC', 'RC', 'Diversity',
                                                                       'SD', 'DD'],
                        default='Random', help='Sampling method for active learning.')

    # parser.add_argument('-ep', '--epochs', type=int, default=1,
    #                    help='How many epochs.')

    parser.add_argument('-iss', '--init_sample_size', type=float, default=0.05,
                        help='Initial random sample size.')

    parser.add_argument('-ns', '--n_sample_size', type=float, default=0.01,
                        help='Sample size in the subsequents AL iterations.')

    parser.add_argument('-ait', '--al_iterations', type=int, default=3, help='Number of AL iterations.')

    parser.add_argument('-err', '--weakly_error', type=float, default=0.25,
                        help='The error rate of the custom WeaklyLabeller')

    parser.add_argument('-r', '--resetting_model', action='store_true',
                        help='This argument determines whether the model is reset each active learning (AL) iteration.')

    parser.add_argument('-pat', '--patience', type=int, default=3,
                        help='This the tolerance parameter for Early stopping.')

    parser.add_argument('-d', '--delta', type=float, default=0,
                        help='Confidence from which the pseudo labels are accepted as Pseudo Labels.')

    parser.add_argument('-dr', '--delta_rate', type=float, default=0.0,
                        help='Delta adjustment if the Pseudo Label amount increases.')

    args = parser.parse_args()

    data_path = args.path
    test_path = args.test_path
    pipeline_mode = args.mode
    _sampling_method = args.sampling_method
    # _epochs = args.epochs
    _init_sample_size = args.init_sample_size
    _n_sample_size = args.n_sample_size
    if _n_sample_size >= 1:
        try:
            _n_sample_size = int(_n_sample_size)
        except ValueError:
            raise argparse.ArgumentTypeError('Argument parser failed.')

    _al_iterations = args.al_iterations
    _weakly_error = args.weakly_error
    _resetting_model = args.resetting_model
    _patience = args.patience
    _delta = args.delta
    _delta_rate = args.delta_rate

    m = Main(data_path,
             test_path,
             pipeline_mode,
             _sampling_method,
             weakly_error=_weakly_error,
             al_iterations=_al_iterations,
             n_sample_size=_n_sample_size,
             init_sample_size=_init_sample_size,
             resetting_model=_resetting_model,
             patience=_patience,
             delta=_delta,
             delta_rate=_delta_rate
             )
    m.run()
