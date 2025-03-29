# Created by PeterC - 09/16/2024 - Reference: https://lightning.ai/docs/pytorch/stable/starter/introduction.html, https://lightning.ai/docs/pytorch/stable/starter/converting.html

import os
import torch
from torch import optim, nn, utils, Tensor
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
import lightning as L

# define any number of nn.Modules (or use your current ones)
encoder = nn.Sequential(nn.Linear(28 * 28, 64), nn.ReLU(), nn.Linear(64, 3))
decoder = nn.Sequential(nn.Linear(3, 64), nn.ReLU(), nn.Linear(64, 28 * 28))

# NOTE: the main difference Lighting has is the definition of the training, validation, test and some more steps within the model class itself.
# The trainer object then uses these implementations to provide all the API functionalities by means of its methods.

# define the LightningModule
class LitAutoEncoder(L.LightningModule):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward
        x, _ = batch
        x = x.view(x.size(0), -1)
        z = self.encoder(x)
        x_hat = self.decoder(z)
        loss = nn.functional.mse_loss(x_hat, x)
        # Logging to TensorBoard (if installed) by default
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer


# Init the autoencoder
autoencoder = LitAutoEncoder(encoder, decoder)

# Setup data
dataset = MNIST(os.getcwd(), download=True, transform=ToTensor())
train_loader = utils.data.DataLoader(dataset) # dataloaders and datasets are treated just like the straight torch ones

# train the model (hint: here are some helpful Trainer arguments for rapid idea iteration)
trainer = L.Trainer(limit_train_batches=100, max_epochs=100, accelerator="gpu", devices=1)
trainer.fit(model=autoencoder, train_dataloaders=train_loader)


# load checkpoint
#checkpoint = "./lightning_logs/version_0/checkpoints/epoch=0-step=100.ckpt"
#autoencoder = LitAutoEncoder.load_from_checkpoint(
#    checkpoint, encoder=encoder, decoder=decoder)

# choose your trained nn.Module
encoder = autoencoder.encoder
encoder.eval()

# embed 4 fake images!
fake_image_batch = torch.rand(4, 28 * 28, device=autoencoder.device)
embeddings = encoder(fake_image_batch)
print("⚡" * 20, "\nPredictions (4 image embeddings):\n",
      embeddings, "\n", "⚡" * 20)
