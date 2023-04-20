#!/usr/bin/env python
# -*- coding: utf-8 -*-
from trainer import Trainer
from preprocessor import Preprocessor, transform_data
from evaluator import Evaluator
from transformers import AutoModelForSequenceClassification
import torch
from labeler import KMeansLabeller, StrongLabeller
from sampler import Sampler
import pandas as pd
import wandb
import plotter as p
import argparse


class Main:
    def __init__(self,
                 path,
                 mode,
                 sampling_method,
                 epochs=1,
                 al_iterations=3,
                 init_sample_size=0.2,
                 n_sample_size=0.1):

        self.data = Preprocessor(path)
        self.mode = mode

        model_name = "bert-base-cased"
        weak_labeler = 'K-Means'

        self.fixed_centroids = True
        # If Data sample too small there exists a possibility that there might not be a representative of
        # each class. This will lead to problems when picking centroids. Therefor make the choice fixed.
        if len(self.data.train_data) > 3000:
            self.fixed_centroids = False
            # Set data Labels to ambiguous number but only if fixed_centroids = false

        self.strong_labeler = StrongLabeller(self.data.control)
        # Load model
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=4)
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")

        self.model.to(device)
        self.trainer = Trainer(self.model, device)
        self.evaluator = Evaluator(device)
        self.sampler = Sampler(device)
        self.sampling_method = sampling_method
        self.mode = mode
        # Empty cache
        torch.cuda.empty_cache()

        self.hyperparameters = {
            'Mode': mode,
            'Classes': 4,
            'Model Name': model_name,
            'Weak Labeler': weak_labeler,
            'Fixed Weak Labelling': True,
            'Data Set': 'AG_NEWS',
            'Train Set': len(self.data.train_data),
            'Batch Size': 4,
            'Epochs': epochs,
            'AL Iterations': al_iterations,
            'AL Batch Size': 4,
            'Sampling Method': sampling_method,
            'Init Sample Size': init_sample_size,
            'N-Sample': [n_sample_size] * al_iterations
        }

    def run(self):
        if self.mode == 'Standard':
            self.standard_ml(self.hyperparameters)
        elif self.mode == 'Test':
            self.proto(self.hyperparameters)
        else:
            # Default
            self.al(self.hyperparameters)

    def standard_ml(self, hyperparameters):
        # Just for understanding, could have just pasted unlabelled for now
        self.data.labelled = self.strong_labeler.label(self.data.partial)

        train_dataloader = transform_data(self.data.labelled)

        trained_model = self.trainer.train(train_dataloader, 0)

        eval_dataloader = transform_data(self.data.eval_data)
        self.evaluator.eval(trained_model, eval_dataloader)

    def al(self, hyperparameters):
        loss = []
        with wandb.init(project='active-learning-plus', config=hyperparameters):
            init_sample_size = hyperparameters['Init Sample Size']
            sample_size = hyperparameters['N-Sample']
            al_iterations = hyperparameters['AL Iterations']

            eval_dataloader = transform_data(self.data.eval_data)
            init_sample, self.data.partial = self.sampler.sample(self.data.partial, init_sample_size)
            self.data.labelled = self.strong_labeler.label(init_sample)

            # --------------- AL PLUS --------------- #
            if self.mode == 'AL+':
                weak_labeler = KMeansLabeller(self.data, self.fixed_centroids)
                self.data.partial = weak_labeler.label(self.data.partial)
                train_set = pd.concat([self.data.labelled, self.data.partial])
            else:
                train_set = self.data.labelled
            # --------------- AL PLUS --------------- #

            train_dataloader = transform_data(train_set)
            self.trainer.train(train_dataloader, 0)
            self.evaluator.eval(self.trainer.model, eval_dataloader)

            loss.append(wandb.run.summary['loss'])

            for i in range(al_iterations):
                sample, self.data.partial = self.sampler.sample(data=self.data.partial,
                                                                sample_size=sample_size[i],
                                                                sampling_method=self.sampling_method,
                                                                model=self.trainer.model
                                                                )
                self.data.labelled = pd.concat([self.data.labelled, self.strong_labeler.label(sample)])

                # --------------- AL PLUS --------------- #
                train_set = pd.concat([self.data.labelled, self.data.partial]) if self.mode == 'AL+' else \
                    self.data.labelled
                # --------------- AL PLUS --------------- #

                train_dataloader = transform_data(train_set)
                self.trainer.train(train_dataloader, i+1)
                self.evaluator.eval(self.trainer.model, eval_dataloader)
                loss.append(wandb.run.summary['loss'])

            p.standard_chart(y=loss, x_label='AL iteration', y_label='Loss',
                             title='Loss - Training cycles')

    def proto(self, hyperparameters):
        pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process passed Hyperparameters.')

    parser.add_argument('-p', '--path', type=str, help='Path to the csv file with the data set.',
                        default='/Users/misha/Desktop/Bachelor-Thesis/BA/data_sets/the_one/small_e.csv')

    parser.add_argument('-m', '--mode', type=str, choices=['AL', 'AL+', 'Standard'], default='AL+',
                        help='The Learning mode.')

    parser.add_argument('--sm', '--sampling_method', type=str, choices=['Random', 'LC'], default='Random',
                        help='Sampling method for active learning.')

    parser.add_argument('--e', '--epochs', type=int, default=1,
                        help='How many epochs.')

    parser.add_argument('--iss', '--init_sample_size', type=float, default=0.2,
                        help='Initial random sample size.')

    parser.add_argument('--ns', '--n_sample_size', type=float, default=0.1,
                        help='Sample size in the subsequents AL iterations.')

    parser.add_argument('--ait', '--al_iterations', type=int, default=3, help='Number of AL iterations.')

    args = parser.parse_args()

    data_path = args.path
    pipeline_mode = args.mode
    _sampling_method = args.sm
    _epochs = args.e
    _init_sample_size = args.iss
    _n_sample_size = args.ns
    _al_iterations = args.ait

    m = Main(data_path,
             pipeline_mode,
             _sampling_method,
             epochs=_epochs,
             al_iterations=_al_iterations,
             n_sample_size=_n_sample_size,
             init_sample_size=_init_sample_size)
    m.run()
