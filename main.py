#!/usr/bin/env python
# -*- coding: utf-8 -*-
from trainer import Trainer
from preprocessor import Preprocessor, to_data_loader
from evaluator import Evaluator
from transformers import AutoModelForSequenceClassification
import torch
from labeler import KMeansLabeller, StrongLabeller, CustomLabeller
from sampler import Sampler
import pandas as pd
import wandb
# import plotter as p
import argparse


class Main:
    def __init__(self,
                 path,
                 mode,
                 sampling_method,
                 weakly_error=0.25,
                 epochs=1,
                 al_iterations=3,
                 init_sample_size=0.2,
                 n_sample_size=0.1):

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
        self.data = Preprocessor(path, self.device.type)

        self.mode = mode
        weak_labeler = 'K-Means'

        self.fixed_centroids = True
        # If Data sample too small there exists a possibility that there might not be a representative of
        # each class. This will lead to problems when picking centroids. Therefor make the choice fixed.
        if len(self.data.train_data) > 3000:
            self.fixed_centroids = False
            # Set data Labels to ambiguous number but only if fixed_centroids = false

        self.strong_labeler = StrongLabeller(self.data.control)

        self.model.to(self.device)

        self.eval_dataloader = to_data_loader(self.data.eval_data, self.device.type)
        self.evaluator = Evaluator(self.device, self.eval_dataloader)
        self.trainer = Trainer(self.model, self.device, self.evaluator)
        self.sampler = Sampler(self.device)
        self.sampling_method = sampling_method
        self.mode = mode
        self.weakly_error = weakly_error

        # pass the actual error rate also? because if insufficient data size
        self.hyperparameters = {
            'Mode': mode,
            # 'Classes': 4,
            # 'Model Name': model_name,
            'Weak Labeler': weak_labeler,
            'Weakly Error': weakly_error,
            'Data Set': 'AG_NEWS',
            'Train Set': len(self.data.train_data),
            # 'Batch Size': 2 if self.device.type == 'CPU' else 256,
            # 'Epochs': epochs,
            # 'AL Iterations': al_iterations,
            'Sampling Method': sampling_method,
            'Init Sample Size': init_sample_size,
            # 'N-Sample': [n_sample_size] * al_iterations
        }

    def run(self):
        if self.mode == 'Standard':
            self.standard_ml(self.hyperparameters)
        elif self.mode == 'Dev':
            self.proto(self.hyperparameters)
        else:
            # Default
            self.al(self.hyperparameters)

    def standard_ml(self, hyperparameters):
        self.data.labelled = self.strong_labeler.label(self.data.partial)
        train_dataloader = to_data_loader(self.data.labelled, self.device.type)
        trained_model = self.trainer.train(train_dataloader, 0)
        self.evaluator.eval(trained_model)

    # make absolute number of samples and approximate
    def al(self, hyperparameters):
        # loss = []
        with wandb.init(project='active-learning-plus', config=hyperparameters):
            init_sample_size = hyperparameters['Init Sample Size']
            # sample_size = hyperparameters['N-Sample']
            # al_iterations = hyperparameters['AL Iterations']
            print('AL Iteration: 0')

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
            train_dataloader = to_data_loader(train_set, self.device.type)
            self.trainer.train(train_dataloader, 0)

            # Here early stoppage argument if accuracy did not improve from al iteration to al iteration
            current_accuracy = self.trainer.current_accuracy
            # for i in range(al_iterations):
            i = 0
            epsilon = 0.02
            counter = 0
            while True:
                print(f'AL Iteration: {i+1}')
                sample, self.data.partial = self.sampler.sample(data=self.data.partial,
                                                                # sample_size=sample_size[i],
                                                                sample_size=0.01,
                                                                sampling_method=self.sampling_method,
                                                                model=self.trainer.model
                                                                )
                self.data.labelled = pd.concat([self.data.labelled, self.strong_labeler.label(sample)])

                # --------------- AL PLUS --------------- #
                train_set = pd.concat([self.data.labelled, self.data.partial]) if self.mode == 'AL+' else \
                    self.data.labelled
                # --------------- AL PLUS --------------- #
                train_dataloader = to_data_loader(train_set, self.device.type)
                self.trainer.train(train_dataloader, i+1)
                # the accuracy did not increase from previous iteration return
                # next step du some epsilon which allows for some decrease
                # Is this achievable?
                # need some other stopping condition...and or propage the model with the said accuracy like in trainer
                # Model did not surpass accuracy of 0.8925
                # print total samples..
                # retrun if no improvement here...
                # log sample size and make it fixed?
                # Tweak how good Weakly labelelr is
                # Parameter übergabe weakly labeller / pseudo labelling
                if counter >= 2:
                    return
                if not self.trainer.current_accuracy > current_accuracy:
                    counter += 1
                else:
                    current_accuracy = self.trainer.current_accuracy
                    counter = 0
                i += 1


    def proto(self, hyperparameters):
        with wandb.init(project='active-learning-plus', config=hyperparameters):
            init_sample_size = hyperparameters['Init Sample Size']
            sample_size = hyperparameters['N-Sample']
            al_iterations = hyperparameters['AL Iterations']
            print('AL Iteration: 0')
            init_sample, self.data.partial = self.sampler.sample(self.data.partial, init_sample_size)
            self.data.labelled = self.strong_labeler.label(init_sample)

            # --------------- AL PLUS --------------- #
            if self.mode == 'Dev':
                # print(self.weakly_error)
                weak_labeler = CustomLabeller(self.weakly_error, self.data.control)
                print(len(self.data.partial))
                self.data.partial = weak_labeler.label(self.data.partial)
                train_set = pd.concat([self.data.labelled, self.data.partial])
            else:
                train_set = self.data.labelled
            # --------------- AL PLUS --------------- #

            # train_dataloader = transform_data(train_set, self.device.type)
            train_dataloader = to_data_loader(train_set, self.device.type)
            self.trainer.train(train_dataloader, 0)
            self.evaluator.eval(self.trainer.model)

            # loss.append(wandb.run.summary['loss'])

            for i in range(al_iterations):
                print(f'AL Iteration: {i + 1}')
                sample, self.data.partial = self.sampler.sample(data=self.data.partial,
                                                                sample_size=sample_size[i],
                                                                sampling_method=self.sampling_method,
                                                                model=self.trainer.model
                                                                )
                self.data.labelled = pd.concat([self.data.labelled, self.strong_labeler.label(sample)])

                # --------------- AL PLUS --------------- #
                train_set = pd.concat([self.data.labelled, self.data.partial]) if self.mode == 'Dev' else \
                    self.data.labelled
                # --------------- AL PLUS --------------- #

                # train_dataloader = transform_data(train_set, self.device.type)
                train_dataloader = to_data_loader(train_set, self.device.type)

                self.trainer.train(train_dataloader, i + 1)
                self.evaluator.eval(self.trainer.model)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process passed Hyperparameters.')

    parser.add_argument('-p', '--path', type=str, help='Path to the csv file with the data set.',
                        default='/Users/misha/Desktop/Bachelor-Thesis/BA/data_sets/the_one/small_t.csv')

    parser.add_argument('-m', '--mode', type=str, choices=['AL', 'AL+', 'Dev', 'Standard'], default='AL+',
                        help='The Learning mode.')

    parser.add_argument('-sm', '--sampling_method', type=str, choices=['Random', 'EC', 'LC', 'MC', 'RC'],
                        default='Random', help='Sampling method for active learning.')

    parser.add_argument('-ep', '--epochs', type=int, default=1,
                        help='How many epochs.')

    parser.add_argument('-iss', '--init_sample_size', type=float, default=0.2,
                        help='Initial random sample size.')

    parser.add_argument('-ns', '--n_sample_size', type=float, default=0.1,
                        help='Sample size in the subsequents AL iterations.')

    parser.add_argument('-ait', '--al_iterations', type=int, default=3, help='Number of AL iterations.')

    parser.add_argument('-err', '--weakly_error', type=float, default=0.25,
                        help='The error rate of the custom WeaklyLabeller')

    args = parser.parse_args()

    data_path = args.path
    pipeline_mode = args.mode
    _sampling_method = args.sampling_method
    _epochs = args.epochs
    _init_sample_size = args.init_sample_size
    _n_sample_size = args.n_sample_size
    _al_iterations = args.al_iterations
    _weakly_error = args.weakly_error

    m = Main(data_path,
             pipeline_mode,
             _sampling_method,
             weakly_error=_weakly_error,
             epochs=_epochs,
             al_iterations=_al_iterations,
             n_sample_size=_n_sample_size,
             init_sample_size=_init_sample_size)
    m.run()
