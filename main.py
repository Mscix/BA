#!/usr/bin/env python
# -*- coding: utf-8 -*-

from data_sets.data_enum import DataPath
from trainer import Trainer
from preprocessor import Preprocessor
from evaluator import Evaluator
import logging
from transformers import AutoModelForSequenceClassification, pipeline
import torch
from torch.utils.data import DataLoader

LOGGER = None

# TODO: Change this structure to a class based one and not method based!


def standard_ml():
    """ Main program """
    data = Preprocessor(DataPath.SMALL, LOGGER)

    # Empty cache
    torch.cuda.empty_cache()
    # DataLoader, shuffle: data reshuffled each epoch, batch: 4 instances, processed at once
    train_set, test_set = data.to_arrow_data()
    train_dataloader = DataLoader(dataset=train_set, shuffle=True, batch_size=4)
    eval_dataloader = DataLoader(dataset=test_set, batch_size=4)

    # Load model
    model = AutoModelForSequenceClassification.from_pretrained("bert-base-cased", num_labels=4)
    # Use GPU if it is available
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)

    print('TRAINING')
    trainer = Trainer(train_dataloader, model, device, 2, LOGGER)
    trainer.train()
    trained_model = trainer.get_model()
    print('EVALUATING')
    evaluator = Evaluator(eval_dataloader, trained_model, device, LOGGER)
    evaluator.eval()
    print('DONE')


def al_ml():
    # iterations = 5
    samples = [0.3, 0.3, 0.3]  # relative sample size to the data set
    # first with random sampling
    data = Preprocessor(DataPath.MEDIUM, LOGGER)
    # Load evaluation data into Data loader
    eval_dataloader = DataLoader(dataset=data.get_test_data(), batch_size=4)
    # Empty cache
    torch.cuda.empty_cache()
    # get a initial sample in accordance to dataset size
    init_sample = data.get_random_sample(0.3)
    # init sample size?
    train_dataloader = DataLoader(dataset=init_sample, shuffle=True, batch_size=4)
    # Load model
    model = AutoModelForSequenceClassification.from_pretrained("bert-base-cased", num_labels=4)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)
    print('INIT TRAINING')
    trainer = Trainer(train_dataloader, model, device, 2, LOGGER)
    trainer.train()
    # for i in range(iterations):
    for s in samples:
        i = 1
        print('TRAINING ' + str(i))
        train_sample = data.get_random_sample(s)
        train_dataloader = DataLoader(dataset=train_sample, shuffle=True, batch_size=4)
        trainer.set_train_loader(train_dataloader)
        trainer.train()

    trained_model = trainer.get_model()
    print('EVALUATING')
    evaluator = Evaluator(eval_dataloader, trained_model, device, LOGGER)
    evaluator.eval()


def al_plus():
    # first find out if the model was trained on this data set
    # get embeddings (text) from Chatgpt
    # Actually same workflow but when sampling for the annotator distribute some of the instances to the Large model
    # and some to the actual annotator, don't care first which give to which
    # => read from embeddings
    # compare with cosine similarity
    # don't want to use pinecone because it consts as well although it should be a good scalabel db

    # first step embeddings für alle descriptions zu ziehen nimm
    # in einer JSON file speichern, diese ins gitignore packen und schauen ob bedarf für SQL ist...
    # DB bedarf sollte erst da sein wenn ganzer daten satz genutzt wird

    # weakly label all
    # compare uncertainty = distance?
    print('Nothing')

def set_up_logger():
    logging.basicConfig(filename='logs.log', filemode='w', format='%(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    return logger


if __name__ == "__main__":
    LOGGER = set_up_logger()
    standard_ml()
    # al_ml()
    #al_plus()
