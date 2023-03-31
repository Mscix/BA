#!/usr/bin/env python
# -*- coding: utf-8 -*-

from enums import DataPath, Mode
from trainer import Trainer
from preprocessor import Preprocessor
from evaluator import Evaluator
import logging
from logging_config import configure_logging
from transformers import AutoModelForSequenceClassification
import torch
from torch.utils.data import DataLoader
from labeler import KMeansLabeller, StrongLabeller
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np

# Call to Logger config
configure_logging()

logger = logging.getLogger(__name__)


class Main:
    def __init__(self, path=DataPath.SMALL, mode=Mode.AL, model_name="bert-base-cased",
                 n_epochs=2):
        # Set-Up
        self.data = Preprocessor(path)

        self.fixed_centroids = True
        # If Data sample too small there exists a possibility that there might not be a representative of
        # each class. This will lead to problems when picking centroids. Therefor make the choice fixed.
        if len(self.data.test_data) > 4000:
            self.fixed_centroids = False
            # Set data Labels to ambiguous number but only if fixed_centroids = false

        self.strong_labeler = StrongLabeller(self.data.control)
        # Load model
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=4)

        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.model.to(device)
        self.trainer = Trainer(self.model, device, n_epochs)
        self.evaluator = Evaluator(device)
        # Empty cache
        torch.cuda.empty_cache()

        if mode.value == 0:
            self.standard_ml()
        elif mode.value == 1:
            self.al()
        elif mode.value == 2:
            self.al_plus()
        else:
            self.testing_grounds()

    def standard_ml(self):
        # Just for understanding, could have just pasted unlabelled for now
        self.data.labelled = self.strong_labeler.label(self.data.unlabelled)
        train_dataloader = DataLoader(dataset=self.data.to_arrow_data(self.data.labelled),
                                      shuffle=True, batch_size=4)

        eval_dataloader = DataLoader(dataset=self.data.to_arrow_data(self.data.eval_data), batch_size=4)
        self.evaluator.set_eval_loader(eval_dataloader)

        self.trainer.set_train_loader(train_dataloader)
        self.trainer.train()

        trained_model = self.trainer.get_model()
        self.evaluator.set_model(trained_model)
        self.evaluator.eval()

    def al(self):
        samples = [0.3] * 3  # samples = [0.3, 0.3, 0.3]

        eval_dataloader = DataLoader(dataset=self.data.to_arrow_data(self.data.eval_data))
        self.evaluator.set_eval_loader(eval_dataloader)

        self.data.unlabelled, init_sample = train_test_split(self.data.unlabelled, test_size=0.3, random_state=42)
        self.data.labelled = self.strong_labeler.label(init_sample)
        train_set = self.data.to_arrow_data(self.data.labelled)
        train_dataloader = DataLoader(dataset=train_set, shuffle=True, batch_size=4)

        logger.info('TRAINING')
        self.trainer.set_train_loader(train_dataloader)
        self.trainer.train()
        for s in samples:
            self.data.unlabelled, train_sample = train_test_split(self.data.unlabelled, test_size=s, random_state=42)
            self.data.labelled = pd.concat([self.data.labelled, self.strong_labeler.label(train_sample)])
            train_sample = self.data.to_arrow_data(self.data.labelled)
            train_dataloader = DataLoader(dataset=train_sample, shuffle=True, batch_size=4)
            self.trainer.set_train_loader(train_dataloader)
            self.trainer.train()

        logger.info('EVALUATING')
        trained_model = self.trainer.get_model()
        self.evaluator.set_model(trained_model)
        self.evaluator.eval()
        # Accuracy is sometimes 0 lol probably because the eval set is very smol

    def al_plus(self):
        # TODO: No iteration yet as uncertainty sampling not implemented in any way...
        # Load evaluation and test data into Data loader, not touched by Weakly labeller
        eval_dataloader = DataLoader(dataset=self.data.to_arrow_data(self.data.eval_data), batch_size=4)
        self.evaluator.set_eval_loader(eval_dataloader)
        test_dataloader = DataLoader(dataset=self.data.to_arrow_data(self.data.test_data), batch_size=4)
        # Split unlabelled data into 2% that will be manually annotated and 98% that will be processed further
        remaining, init_sample = train_test_split(self.data.unlabelled, test_size=0.02, random_state=42)
        # Now label the init sample manually
        self.data.labelled = self.strong_labeler.label(init_sample)

        # --------------- AL PLUS ---------------
        # now pick 4 centroids, later randomly?
        if self.fixed_centroids:
            # Reasoning behind this in the __init__ method.
            # Centroids are only picked for K-Means, nothing else changes, they might be part of Eval, Test or Train set
            centroids = self.data.get_first_reps_4_class(self.data.control, keep=True)
        else:
            centroids = self.data.get_first_reps_4_class(self.data.labelled, keep=True)

        # Now label the remaining data with weakly labeller
        embeddings = self.data.get_embeddings_from_df(remaining)
        # Centroids passed do KMeansLabeler are not numbers
        wl = KMeansLabeller(embeddings, centroids, num_clusters=4,  dims=1536)
        logger.info('K-Means prediction')
        y_predict = wl.get_fit_predict()  # returns a list of Class indexes
        # Set labels of data points, the order of the rows is the same
        # TODO: Check that I can just assign this way and do not have to check for Index
        remaining['Class Index'] = y_predict
        self.data.weakly_labelled = remaining
        self.data.unlabelled = None  # unset attribute as all instances are at least weakly labelled now
        # ---- TRAINING ----
        # Train model on the combined data set of labelled and weakly labelled combined
        train_set = pd.concat([self.data.labelled, self.data.weakly_labelled])  # just different attributes concat
        # --------------- AL PLUS ---------------

        train_set = self.data.to_arrow_data(train_set)
        train_dataloader = DataLoader(dataset=train_set, shuffle=True, batch_size=4)
        logger.info('TRAINING')
        self.trainer.set_train_loader(train_dataloader)
        self.trainer.train()

        logger.info('EVALUATING')
        trained_model = self.trainer.get_model()
        self.evaluator.set_model(trained_model)
        self.evaluator.eval()

    @staticmethod
    def testing_grounds():
        data = Preprocessor(DataPath.BIG)
        print(list(map(lambda n: np.array(eval(n)), data.unlabelled['Embedding'].tolist())))


if __name__ == "__main__":
    m = Main(mode=Mode.AL_PLUS)
