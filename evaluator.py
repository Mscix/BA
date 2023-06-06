
# class for graphs and plots and so on
import torch
# Model performance evaluation
import evaluate

import wandb


class Evaluator:
    def __init__(self, device, valid_dataloader, test_dataloader):
        self.device = device
        self.accuracy = evaluate.load('accuracy')
        self.recall = evaluate.load('recall')
        self.precision = evaluate.load('precision')
        self.f1 = evaluate.load('f1')

        # insert all metrics I want to use
        self.metrics = [self.accuracy, self.recall, self.precision, self.f1]
        self.metrics_results = None
        #  A list for all logits
        self.logits_all = []
        # A list for all predicted probabilities
        self.predicted_prob_all = []
        # A list for all predicted labels
        self.predictions_all = []

        self.valid_dataloader = valid_dataloader
        self.test_dataloader = test_dataloader

        self.evaluation_results = {}

    def eval(self, model, valid):
        if model.training:
            model.eval()
        # PUT THE EVAL data permanently on the GPU
        if valid:
            dataloader = self.valid_dataloader
        else:
            dataloader = self.test_dataloader

        loss_accumulator = 0.0
        for batch in dataloader:
            # Get the batch
            batch = {k: v.to(self.device) for k, v in batch.items()}
            # Disable the gradient calculation
            with torch.no_grad():
                # Compute the model output
                outputs = model(**batch)

            loss_accumulator += outputs.loss.item()
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
            for metric in self.metrics:
                metric.add_batch(predictions=predictions, references=batch["labels"])

        # Compute the metrics
        # zero_division=0, If there is no predicted Label for some class the score will be set to 0
        # This should only be a problem if the data set is small
        self.evaluation_results = {
            **self.accuracy.compute(),
            **self.recall.compute(average='macro', zero_division=0),
            **self.precision.compute(average='macro', zero_division=0),
            **self.f1.compute(average='macro'),
            'avg Loss': loss_accumulator / len(dataloader)
        }
        return self.get_final_results(self.evaluation_results, valid)

    def get_predictions(self):
        return self.predictions_all

    @staticmethod
    def get_final_results(evaluation_results, valid):
        if valid:
            prefix = 'validation'
        else:
            prefix = 'test'
        results = {f'{prefix} accuracy': evaluation_results['accuracy'],
                   f'avg {prefix} Loss': evaluation_results['avg Loss'],
                   f'{prefix} f1': evaluation_results['f1'],
                   f'{prefix} precision': evaluation_results['precision'],
                   f'{prefix} recall': evaluation_results['recall']}
        return results
