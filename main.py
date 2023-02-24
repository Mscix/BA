#!/usr/bin/env python
# -*- coding: utf-8 -*-

from data_sets.data_enum import DataPath
from trainer import Trainer
from preprocessor import Preprocessor
from evaluator import Evaluator
from transformers import AutoModelForSequenceClassification
import torch
from torch.utils.data import DataLoader


# THIS CLASS IS THE COORDINATOR PIPOLINE
def standart_ml():
    """ Main program """
    data = Preprocessor(DataPath.SMALL)

    # Empty cache
    torch.cuda.empty_cache()
    # DataLoader, shuffle: data reshuffled each epoch, batch: 4 instances, processed at once
    train_dataloader = DataLoader(dataset=data.get_train_data(), shuffle=True, batch_size=4)
    eval_dataloader = DataLoader(dataset=data.get_test_data(), batch_size=4)

    # Load model
    model = AutoModelForSequenceClassification.from_pretrained("bert-base-cased", num_labels=4)
    # Use GPU if it is available
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)

    print('TRAINING')
    trainer = Trainer(train_dataloader, model, device)
    trainer.train()
    trained_model = trainer.get_model()
    print('EVALUATING')
    evaluator = Evaluator(eval_dataloader, trained_model, device)
    evaluator.eval()
    print('DONE')

def al_ml():
    data = Preprocessor(DataPath.SMALL)


if __name__ == "__main__":
    al_ml()
