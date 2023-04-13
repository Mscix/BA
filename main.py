#!/usr/bin/env python
# -*- coding: utf-8 -*-
from enums import Data, Mode, EvalSet, SamplingMethod
from trainer import Trainer
from preprocessor import Preprocessor
from evaluator import Evaluator

from transformers import AutoModelForSequenceClassification
import torch
from labeler import KMeansLabeller, StrongLabeller
from sampler import Sampler
import pandas as pd
import wandb



class Main:
    def __init__(self, data_type=Data.SMALL, eval_set_type=EvalSet.EVALUATION_SET, mode=Mode.AL,
                 model_name="bert-base-cased", weak_labeler='K-Means', default_epochs=1, al_iterations=3,
                 sampling_method=SamplingMethod.RANDOM):
        # Set-Up
        self.data = Preprocessor(data_type.value['path'])
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

        # Model Performance on Training set
        # Model Performance on Evaluation set

        self.fixed_centroids = True
        # If Data sample too small there exists a possibility that there might not be a representative of
        # each class. This will lead to problems when picking centroids. Therefor make the choice fixed.
        if len(self.data.train_data) > 3000:
            self.fixed_centroids = False
            # Set data Labels to ambiguous number but only if fixed_centroids = false

        self.strong_labeler = StrongLabeller(self.data.control)
        # Load model
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=4)

        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.model.to(device)
        self.trainer = Trainer(self.model, device)
        self.evaluator = Evaluator(device)
        self.sampler = Sampler()
        # Empty cache
        torch.cuda.empty_cache()

        # Set up Hyper parameters
        # General: Mode, Model, Weak Labeler, Training Set, Evaluation Set, Epochs, AL Iterations, Sampling Method,
        # Metrics: Accuracy, Recall, Precision, F1
        hyperparameters = {
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
        }

        if mode == Mode.STANDARD:
            self.standard_ml(hyperparameters)
        elif mode == Mode.AL:
            self.al(hyperparameters)
        elif mode == Mode.AL_PLUS:
            self.al_plus(hyperparameters)
        else:
            raise Exception('Invalid Mode')

    def standard_ml(self, hyperparameters):
        # Just for understanding, could have just pasted unlabelled for now
        self.data.labelled = self.strong_labeler.label(self.data.unlabelled)

        train_dataloader = self.data.transform_data(self.data.labelled)

        trained_model = self.trainer.train(train_dataloader)

        eval_dataloader = self.data.transform_data(self.data.eval_data)
        self.evaluator.eval(trained_model, eval_dataloader)

    def al(self, hyperparameters):
        with wandb.init(project='active-learning-plus', config=hyperparameters):
            # Difference between epoch and
            # Too small for small data set
            # init = 0.05
            # samples = [0.01] * 3
            init = 0.2
            samples = [0.2] * 3

            # init 0.05 sample
            # 1-sample 0.01
            # 2-sample 0.01
            # 3-sample 0.01
            # Get init sample
            init_sample, self.data.unlabelled = self.sampler.sample(self.data.unlabelled, init)
            self.data.labelled = self.strong_labeler.label(init_sample)

            train_dataloader = self.data.transform_data(self.data.labelled)
            self.trainer.train(train_dataloader)

            for s in samples:
                # PROBLEM HERE
                train_sample, self.data.unlabelled = self.sampler.sample(self.data.unlabelled, s)
                # Combine/Concat new labelled data with old labelled data
                self.data.labelled = pd.concat([self.data.labelled, self.strong_labeler.label(train_sample)])

                train_dataloader = self.data.transform_data(self.data.labelled)
                self.trainer.train(train_dataloader)
            eval_dataloader = self.data.transform_data(self.data.eval_data)
            self.evaluator.eval(self.trainer.model, eval_dataloader)

    # TODO at some point weakly labelled and strong labelled are interchangeble because all instances are
    # TODO chanage name...
    #  always weakly labelled
    def al_plus(self, hyperparameters):
        with wandb.init(project='active-learning-plus', config=hyperparameters):
            # Too small for small data set
            # init = 0.05
            # samples = [0.01] * 3
            init = 0.2
            samples = [0.2] * 3

            config = wandb.config
            init_sample, self.data.unlabelled = self.sampler.sample(self.data.unlabelled, init)
            self.data.labelled = self.strong_labeler.label(init_sample)

            # --------------- AL PLUS ---------------
            weak_labeler = KMeansLabeller(self.data, self.fixed_centroids)
            self.data.weakly_labelled = weak_labeler.label(self.data.unlabelled)
            train_set = pd.concat([self.data.labelled, self.data.weakly_labelled])
            # --------------- AL PLUS ---------------

            train_dataloader = self.data.transform_data(train_set)
            # zwischen jeden training schauen wie gut es läuft
            self.trainer.train(train_dataloader)
            for s in samples:
                sample, self.data.weakly_labelled = self.sampler.sample(self.data.weakly_labelled, s)
                # Combine/Concat new labelled data with old labelled data
                self.data.labelled = pd.concat([self.data.labelled, self.strong_labeler.label(sample)])
                train_set = pd.concat([self.data.labelled, self.data.weakly_labelled])

                train_dataloader = self.data.transform_data(train_set)
                self.trainer.train(train_dataloader)
            eval_dataloader = self.data.transform_data(self.data.eval_data)
            self.evaluator.eval(self.trainer.model, eval_dataloader)

    def proto(self):
        pass

    def make(self, config):
        # Unpack Config here return data loader, Eval Loader trainer and so on the logic should not be in the init
        # Returns DataLoader, E
        pass


# TODO add the ability to log after each iteration???
if __name__ == "__main__":

    m = Main(data_type=Data.BIG, mode=Mode.AL_PLUS)
