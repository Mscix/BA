import wandb
import torch
from torch.optim import AdamW
# from transformers import get_scheduler
from torch.utils.data import DataLoader
from preprocessor import Preprocessor
from transformers import AutoModelForSequenceClassification



class Trainer:
    def __init__(self,  model, device, evaluator, resetting_model, initial_weights):
        self.model = model
        self.device = device
        self.optimizer = AdamW(params=self.model.parameters(), lr=5e-6)  # check if the optimizer ok like this
        self.al_iteration = 0
        self.current_accuracy = 0
        self.evaluator = evaluator
        self.resetting_model = resetting_model
        self.initial_weights = initial_weights

    def train(self, train_dataloader: DataLoader, data: Preprocessor, al_iteration=0):
        # need criterion?
        wandb.watch(self.model, log='all', log_freq=10)
        self.reset_model()
        self.model.train()
        epoch = 0
        # while True:
        for i in range(5):
            # Loop through the batches
            for batch in train_dataloader:
                # Get the batch
                batch = {k: v.to(self.device) for k, v in batch.items()}
                # Compute the model output for the batch
                outputs = self.model(**batch)
                # Loss computed by the model
                loss = outputs.loss
                # backpropagates the error to calculate gradients
                loss.backward()
                # Update the model weights
                self.optimizer.step()
                # Clear the gradients
                self.optimizer.zero_grad()

                self.log_training(al_iteration, loss, epoch, len(data.labelled))

            print(f'Epoch {epoch}')
            eval_obj = {
                "AL Iteration": al_iteration,
                'epoch': epoch + i * 3,
                "Strong Labels": len(data.labelled)
            }
            self.evaluator.eval(self.model, eval_obj)
            self.current_accuracy = self.evaluator.metrics_results['accuracy']
            print(str(self.current_accuracy))
            """
            # Stops if accuracy got worse and returns model from the iteration before
            if self.current_accuracy <= self.evaluator.metrics_results['accuracy']:
                print(str(self.current_accuracy) + ' <= ' + str(self.evaluator.metrics_results['accuracy']))
                self.current_accuracy = self.evaluator.metrics_results['accuracy']
            else:
                print(str(self.current_accuracy) + ' > ' + str(self.evaluator.metrics_results['accuracy']))
                torch.cuda.empty_cache()
                return
            """
            epoch += 1
            torch.cuda.empty_cache()


    def reset_model(self):
        if self.resetting_model:
            self.model = self.model.to("cpu")
            torch.cuda.empty_cache()
            model = AutoModelForSequenceClassification.from_pretrained("bert-base-cased", num_labels=4)
            model.load_state_dict(self.initial_weights)
            # Move the model and its tensors back to the GPU
            self.model = model.to(self.device)
            self.optimizer = AdamW(params=self.model.parameters(), lr=5e-6)
            self.current_accuracy = 0
            torch.cuda.empty_cache()

    @staticmethod
    def log_training(al_iteration, loss, epoch, strong_labels):
        wandb.log({"AL Iteration": al_iteration, "epoch": epoch, "loss": loss, "Strong Labels": strong_labels})
