import wandb
import torch
from torch.optim import AdamW
from transformers import get_scheduler
from torch.utils.data import DataLoader
from preprocessor import Preprocessor
# Progress bar
# from tqdm.auto import tqdm


class Trainer:
    def __init__(self,  model, device, evaluator):
        self.model = model
        self.device = device
        # Number of training steps
        self.num_training_steps = 0
        self.optimizer = AdamW(params=self.model.parameters(), lr=5e-6)  # check if the optimizer ok like this
        self.al_iteration = 0
        self.current_accuracy = 0
        self.evaluator = evaluator
        self.step = 0

    # TODO early stopping
    # Calc Accuracy in the training step
    # check woth the currect accuracy if gets lower return
    def train(self, train_dataloader: DataLoader, data: Preprocessor, al_iteration=0 ):
        # need criterion?
        wandb.watch(self.model, log='all', log_freq=10)
        # scheduler = get_scheduler(name="linear", optimizer=self.optimizer, num_warmup_steps=0)

        self.model.train()
        temp_model = self.model
        epoch = 0
        while True:

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
                # Learning rate sheduler
                #scheduler.step()
                # Clear the gradients
                self.optimizer.zero_grad()

                self.log_training(al_iteration, loss, epoch, self.step, len(data.labelled))

                self.step += 1
            print(f'Epoch {epoch}')
            size_labelled = len(data.labelled)
            percent_labelled = (len(data.labelled) / len(data.train_data)) * 100

            eval_obj = {"AL Iteration": al_iteration, 'epoch': epoch, "Strong Labels": size_labelled,
                        'Percent Labelled': percent_labelled}
            self.evaluator.eval(self.model, eval_obj)

            # Stops if accuracy got worse and returns model from the iteration before
            if self.current_accuracy <= self.evaluator.metrics_results['accuracy']:
                print(str(self.current_accuracy) + ' <= ' + str(self.evaluator.metrics_results['accuracy']))
                self.current_accuracy = self.evaluator.metrics_results['accuracy']
            else:
                print(str(self.current_accuracy) + ' > ' + str(self.evaluator.metrics_results['accuracy']))
                torch.cuda.empty_cache()
                # does this have any effect? or the newer model is returned?
                return temp_model
            epoch += 1

    @staticmethod
    def log_training(al_iteration, loss, epoch, step, strong_labels):
        wandb.log({"AL Iteration": al_iteration, "epoch": epoch, "loss": loss, "Strong Labels": strong_labels})
