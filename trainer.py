from torch.optim import AdamW
from transformers import get_scheduler
import logging
# Progress bar
from tqdm.auto import tqdm


class Trainer:
    def __init__(self,  model, device, num_epochs, train_dataloader=None):
        self.model = model
        self.device = device
        self.train_dataloader = train_dataloader
        self.num_epochs = num_epochs
        # Number of training steps
        self.num_training_steps = num_epochs * len(self.train_dataloader)
        self.optimizer = AdamW(params=self.model.parameters(), lr=5e-6)
        self.scheduler = get_scheduler(name="linear",
                                       optimizer=self.optimizer,
                                       num_warmup_steps=0,
                                       num_training_steps=self.num_training_steps)

    def train(self):
        if not self.train_dataloader:
            raise Exception('Before training, Trainer requires DataLoader to be set. Please use '
                            '\'set_train_loader(DataLoader)\'')
        # Set the progress bar
        progress_bar = tqdm(range(self.num_training_steps))  # had a problem with the progress bar before... or not?

        # Tells the model that we are training the model
        self.model.train()
        # Loop through the epochs
        for epoch in range(self.num_epochs):
            # Loop through the batches
            for batch in self.train_dataloader:
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
                self.scheduler.step()
                # Clear the gradients
                self.optimizer.zero_grad()
                # Update the progress bar
                progress_bar.update(1)

    def get_model(self):
        return self.model

    def set_train_loader(self, train_loader):
        self.train_dataloader = train_loader
        self.num_training_steps = self.num_epochs * len(self.train_dataloader)
