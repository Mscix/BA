
# class for graphs and plots and so on
import torch
# Model performance evaluation
import evaluate

import wandb


class Evaluator:
    def __init__(self, device):
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

    def eval(self, model, eval_dataloader):
        if not eval_dataloader:
            raise Exception('Before evaluating, Evaluator requires DataLoader to be set. Please use '
                            '\'set_eval_loader(DataLoader)\'')
        if not model:
            raise Exception('Before evaluating, please set the model that should be evaluated.')
        # Tells the model that we are evaluating the model performance
        model.eval()

        for batch in eval_dataloader:
            # Get the batch
            batch = {k: v.to(self.device) for k, v in batch.items()}
            # Disable the gradient calculation
            with torch.no_grad():
                # Compute the model output
                outputs = model(**batch)
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
        metrics_results = {
            **self.accuracy.compute(),
            **self.recall.compute(average='macro', zero_division=0),
            **self.precision.compute(average='macro', zero_division=0),
            **self.f1.compute(average='macro')
        }
        print(metrics_results['accuracy'])
        wandb.log(metrics_results)

        # TODO: save and export model
        # torch.onnx.export(model, texts, 'model.onnx)
        # wandb.save('model.onnx')  # Check how this works

    def get_predictions(self):
        return self.predictions_all
