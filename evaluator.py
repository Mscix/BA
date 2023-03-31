
# class for graphs and plots and so on
import torch
# Model performance evaluation
import evaluate

import logging


class Evaluator:
    def __init__(self, device, model=None, eval_dataloader=None):
        self.eval_dataloader = eval_dataloader
        self.model = model
        self.device = device

        acc_metric = evaluate.load("accuracy")
        self.metric = acc_metric  # temporary
        # insert all metrics I want to use
        self.metrics = [acc_metric]  # currently only one metric used

        #  A list for all logits
        self.logits_all = []
        # A list for all predicted probabilities
        self.predicted_prob_all = []
        # A list for all predicted labels
        self.predictions_all = []

    def eval(self):
        if not self.eval_dataloader:
            raise Exception('Before evaluating, Evaluator requires DataLoader to be set. Please use '
                            '\'set_eval_loader(DataLoader)\'')
        if not self.model:
            raise Exception('Before evaluating, please set the model that should be evaluated.')
        # Tells the model that we are evaluting the model performance
        self.model.eval()

        for batch in self.eval_dataloader:
            # Get the batch
            batch = {k: v.to(self.device) for k, v in batch.items()}
            # Disable the gradient calculation
            with torch.no_grad():
                # Compute the model output
                outputs = self.model(**batch)
            # Get the logits
            logits = outputs.logits
            # Append the logits batch to the list
            self.logits_all.append(logits)
            # Get the predicted probabilities for the batch
            predicted_prob = torch.softmax(logits, dim=1)
            # Append the predicted probabilities for the batch to all the predicted probabilities
            self.predicted_prob_all.append(predicted_prob)
            # Get the predicted labels for the batch
            predictions = torch.argmax(logits, dim=-1)
            # Append the predicted labels for the batch to all the predictions
            self.predictions_all.append(predictions)
            # Add the prediction batch to the evaluation metric
            self.metric.add_batch(predictions=predictions, references=batch["labels"])

        # Compute the metric
        print(self.metric.compute())

    def get_predictions(self):
        return self.predictions_all

    def set_eval_loader(self, eval_loader):
        self.eval_dataloader = eval_loader

    def set_model(self, trained_model):
        self.model = trained_model
