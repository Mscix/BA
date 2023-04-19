#!/usr/bin/env python
# -*- coding: utf-8 -*-
from enums import Data, Mode, EvalSet, SamplingMethod
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
import matplotlib.pyplot as plt
import sys



class Main:
    def __init__(self, remote=False, remote_path='', data_type=Data.SMALL, eval_set_type=EvalSet.EVALUATION_SET,
                 mode=Mode.AL, model_name="bert-base-cased", weak_labeler='K-Means', default_epochs=1, al_iterations=3,
                 sampling_method=SamplingMethod.RANDOM, init_sample_size=0.2, n_sample_size=0.1):
        if remote:
            self.data = Preprocessor(remote_path)
        else:
            self.data = Preprocessor(data_type.value['path'])

        # Set-Up
        # eval_set_type
        # Alternative split but measure accuracy on both Validation and training set
        if eval_set_type == EvalSet.TRAINING_SET:
            pass
        elif eval_set_type == EvalSet.EVALUATION_SET:
            pass
        elif eval_set_type == EvalSet.TEST_SET:
            pass
        else:
            pass

        self.fixed_centroids = True
        # If Data sample too small there exists a possibility that there might not be a representative of
        # each class. This will lead to problems when picking centroids. Therefor make the choice fixed.
        if len(self.data.train_data) > 3000:
            self.fixed_centroids = False
            # Set data Labels to ambiguous number but only if fixed_centroids = false

        self.strong_labeler = StrongLabeller(self.data.control)
        # Load model
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=4)

        # device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        # device = "mps" if torch.backends.mps.is_available() else "cpu"
        device = "cpu"
        # device = "mps"
        self.model.to(device)
        self.trainer = Trainer(self.model, device)
        self.evaluator = Evaluator(device)
        self.sampler = Sampler(device)
        self.sampling_method = sampling_method
        self.mode = mode
        # Empty cache
        torch.cuda.empty_cache()

        # Set up Hyper parameters
        # General: Mode, Model, Weak Labeler, Training Set, Evaluation Set, Epochs, AL Iterations, Sampling Method,
        # Metrics: Accuracy, Recall, Precision, F1
        self.hyperparameters = {
            'Mode': mode.value,
            'Classes': 4,
            'Model Name': model_name,
            'Weak Labeler': weak_labeler,
            'Fixed Weak Labelling': True,  # TODO: false is the case when the weakly model is updated
            'Data Set': 'AG_NEWS',
            'Train Set': data_type.value['size'],
            'Batch Size': 4,
            'Eval Set': eval_set_type.value,
            'Epochs': default_epochs,
            'AL Iterations': al_iterations,
            'AL Batch Size': 4,  # TODO: think maybe make it an array?
            'Sampling Method': sampling_method.value,
            'Init Sample Size': init_sample_size,
            'N-Sample': [n_sample_size] * al_iterations
        }

    def run(self):
        if self.mode == Mode.STANDARD:
            self.standard_ml(self.hyperparameters)
        elif self.mode == Mode.TEST:
            self.proto(self.hyperparameters)
        else:
            # raise Exception('Invalid Mode')
            # Default
            self.al(self.hyperparameters)

    def standard_ml(self, hyperparameters):
        # Just for understanding, could have just pasted unlabelled for now
        self.data.labelled = self.strong_labeler.label(self.data.partial)

        train_dataloader = transform_data(self.data.labelled)

        trained_model = self.trainer.train(train_dataloader, 0)

        eval_dataloader = transform_data(self.data.eval_data)
        self.evaluator.eval(trained_model, eval_dataloader)

    # Model Performance on Training set
    # Model Performance on Evaluation set
    # log the iteration
    def al(self, hyperparameters):
        loss = []
        with wandb.init(project='active-learning-plus', config=hyperparameters):
            init_sample_size = hyperparameters['Init Sample Size']
            sample_size = hyperparameters['N-Sample']
            al_iterations = hyperparameters['AL Iterations']

            eval_dataloader = transform_data(self.data.eval_data)
            init_sample, self.data.partial = self.sampler.sample(self.data.partial, init_sample_size)
            self.data.labelled = self.strong_labeler.label(init_sample)

            # --------------- AL PLUS ---------------
            if self.mode == Mode.AL_PLUS:
                weak_labeler = KMeansLabeller(self.data, self.fixed_centroids)
                self.data.partial = weak_labeler.label(self.data.partial)
                train_set = pd.concat([self.data.labelled, self.data.partial])
            else:
                train_set = self.data.labelled

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

                # --------------- AL PLUS ---------------
                if self.mode == Mode.AL_PLUS:
                    train_set = pd.concat([self.data.labelled, self.data.partial])
                else:
                    train_set = self.data.labelled

                train_dataloader = transform_data(train_set)
                self.trainer.train(train_dataloader, i+1)
                self.evaluator.eval(self.trainer.model, eval_dataloader)
                loss.append(wandb.run.summary['loss'])

            p.standard_chart(y=loss, x_label='AL iteration', y_label='Loss',
                             title='Loss - Training cycles')


    def proto(self, hyperparameters):
        pass


if __name__ == "__main__":
    # m = Main(data_type=Data.SMALL, mode=Mode.AL_PLUS)
    args = sys.argv[1:]
    if not args:
        m = Main(data_type=Data.SMALL, mode=Mode.AL_PLUS)
    else:
        m = Main(remote=True, remote_path=args[0])
    m.run()
