import wandb
import torch
from torch.optim import AdamW
# from transformers import get_scheduler
from torch.utils.data import DataLoader
from preprocessor import Preprocessor
from transformers import AutoModelForSequenceClassification
import copy


class Trainer:
    def __init__(self,  model, device, v_eval, resetting_model, initial_weights, patience=3):
        self.model = model
        self.device = device
        self.optimizer = AdamW(params=self.model.parameters(), lr=5e-6)  # check if the optimizer ok like this
        self.al_iteration = 0
        self.best_val_loss = None
        self.best_model = None
        self.v_eval = v_eval
        self.delta = 0
        # self.t_eval = t_eval
        self.resetting_model = resetting_model
        self.initial_weights = initial_weights
        self.patience = patience
        self.patience_counter = 0
        # Deep copy the model to use as temporary storage model for early stopping
        # Do not place on GPU otherwise there will be storage problems
        self.best_model = copy.deepcopy(self.model)
        self.al_results = {}

    def train(self, train_dataloader: DataLoader, data: Preprocessor, al_iteration=0):
        # need criterion?
        wandb.watch(self.model, log='all', log_freq=10)
        self.reset_model()
        self.model.train()

        # Set up training evaluator, as each iteration the train set changes?
        epoch = 0
        # not 'while True:' just in case
        while epoch < 500:
            loss_accumulator = 0.0
            print(f'Epoch {epoch}')
            # Loop through the batches
            for batch in train_dataloader:
                # Get the batch
                batch = {k: v.to(self.device) for k, v in batch.items()}
                # Compute the model output for the batch
                outputs = self.model(**batch)
                # Loss computed by the model
                loss = outputs.loss

                loss_accumulator += loss.item()
                # backpropagates the error to calculate gradients
                loss.backward()
                # Update the model weights
                self.optimizer.step()
                # Clear the gradients
                self.optimizer.zero_grad()
            # LOG best accuracy for this AL Iteration
            # For every AL iteration log new 'Run'
            # Log for both layers AL and epoch layer
            results = self.v_eval.eval(self.model)
            results['AL Iteration'] = al_iteration
            results['Strong Labels'] = len(data.labelled)
            results['avg Training Loss'] = loss_accumulator / len(train_dataloader)
            wandb.log(results)
            epoch += 1
            if self.early_stopping(results):
                wandb.log(self.al_results)
                return
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
            # self.current_accuracy = 0
            torch.cuda.empty_cache()

    def early_stopping(self, results):
        print('Measured Loss: ' + str(results['avg Validation Loss']) +
              ', Current Best Loss: ' + str(self.best_val_loss))
        # Lower loss obviously better
        if self.best_val_loss is None:
            # self.best_val_loss = results['avg Validation Loss']
            self.set_metrics(results)
        elif self.best_val_loss - results['avg Validation Loss'] > self.delta:
            self.set_metrics(results)
            self.patience_counter = 0
            # Loads the current state of self.model into self.best_model as the loss is bette
            self.best_model.load_state_dict(self.model.state_dict())
        elif self.best_val_loss - results['avg Validation Loss'] < self.delta:
            self.patience_counter += 1
            if self.patience_counter >= self.patience:
                print(f'Early stopping {self.patience_counter}')
                # Loads the best state into current model
                self.model.load_state_dict(self.best_model.state_dict())
                # Rest values
                self.patience_counter = 0
                self.best_val_loss = None
                # return True to stop the loop
                return True
        print(f'Current Patience {self.patience_counter}/{self.patience}')
        return False

    def set_metrics(self, results):
        self.al_results['*avg Validation Loss'] = self.best_val_loss = results['avg Validation Loss']
        self.al_results['*accuracy'] = results['accuracy']
        self.al_results['*f1'] = results['f1']
        self.al_results['*precision'] = results['precision']
        self.al_results['*recall'] = results['recall']
        self.al_results['AL Iteration'] = results['AL Iteration']
        self.al_results['Strong Labels'] = results['Strong Labels']

