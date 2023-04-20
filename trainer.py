import wandb
from torch.optim import AdamW
from transformers import get_scheduler
from torch.utils.data import DataLoader
# Progress bar
# from tqdm.auto import tqdm


class Trainer:
    def __init__(self,  model, device):
        self.model = model
        self.device = device
        # Number of training steps
        self.num_training_steps = 0
        self.optimizer = AdamW(params=self.model.parameters(), lr=5e-6)  # check if the optimizer ok like this
        self.al_iteration = 0

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
        step = 0
        # Tells the model that we are training the model
        self.model.train()
        # Loop through the epochs
        for epoch in range(epochs):
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
                self.log_training(al_iteration, loss, epoch, step)
                step += 1

        return self.model

    @staticmethod
    def log_training(al_iteration, loss, epoch, step):
        wandb.log({"AL Iteration": al_iteration, "loss": loss})  # Deleted epoch for now
        # print(f"AL Iteration: {al_iteration}, Epoch: {epoch},"
        #      f" Loss {loss:.3f} after total batches {str(step).zfill(7)}")
