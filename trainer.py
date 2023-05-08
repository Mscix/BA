import wandb
import torch
from torch.optim import AdamW
from transformers import get_scheduler
from torch.utils.data import DataLoader
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
    def train(self, train_dataloader: DataLoader, al_iteration, epochs=1):
        # need criterion?
        wandb.watch(self.model, log='all', log_freq=10)
        training_steps = epochs * len(train_dataloader)  # steps would be 64 but 8 Batches a 4 a 2 epochs

        if not train_dataloader:
            raise Exception('Before training, Trainer requires DataLoader to be set. Please use '
                            '\'set_train_loader(DataLoader)\'')

        # Need this?
        scheduler = get_scheduler(name="linear",
                                       optimizer=self.optimizer,
                                       num_warmup_steps=0,
                                       num_training_steps=training_steps)

        # Set the progress bar
        # progress_bar = tqdm(range(training_steps))  # had a problem with the progress bar before... or not?
        # Tells the model that we are training the model
        self.model.train()
        # While accuracy does not go down don't stop then continue the outer loop (AL-Iterations)
        # Loop through the epochs
        current_accuracy = 0
        temp_model = self.model
        epoch = 0
        #for epoch in range(epochs):
        # Early Stoppage
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
                # Learning rate scheduler
                scheduler.step()
                # Clear the gradients
                self.optimizer.zero_grad()
                # Update the progress bar
                # progress_bar.update(1)
                self.log_training(al_iteration, loss, epoch, self.step)

                self.step += 1
            print(f'Epoch {epoch}')
            epoch += 1
            if epoch > 30:
                raise Exception('Epoch too High eats too much GPU epoch=30.')

            self.evaluator.eval(self.model)
            # Stops if accuracy got worse and returns model from the iteration before
            # TODO how to log this?
            if current_accuracy < self.evaluator.metrics_results['accuracy']:
                current_accuracy = self.evaluator.metrics_results['accuracy']
            else:
                torch.cuda.empty_cache()
                return temp_model

        #torch.cuda.empty_cache()
        #
        #return self.model

    @staticmethod
    def log_training(al_iteration, loss, epoch, step):
        wandb.log({"AL Iteration": al_iteration, "epoch": epoch, "loss": loss})  # Deleted epoch for now
        # print(f"AL Iteration: {al_iteration}, Epoch: {epoch},"
        #      f" Loss {loss:.3f} after total batches {str(step).zfill(7)}")
