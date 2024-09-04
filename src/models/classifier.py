import pytorch_lightning as pl

import torch
from torch import nn
from torch.nn import functional as F


def get_accuracy(logits: torch.Tensor, y: torch.Tensor):
    return ((logits > 0.0) == y).float().mean()


class MalariaDiseaseClassifier(pl.LightningModule):
    def __init__(self, input_shape, hidden_units, output_shape, lr=1e-4):
        super().__init__()

        self.conv_block = nn.Sequential(

            nn.Conv2d(in_channels = input_shape, 
                      out_channels = hidden_units,
                      kernel_size = 2,
                      stride = 2),
            
            nn.ReLU(),

            nn.MaxPool2d(kernel_size = 2, stride = 1),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=hidden_units*100*256, out_features = output_shape),
        )

        
        self.lr = lr

    def forward(self, x):
        x = self.conv_block(x)
        x = self.classifier(x)
        return x

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

    def __compute_batch_loss(self, batch):
        x, y, _ = batch
        y = y.unsqueeze(1)
        y_pred_logits = self.classifier(x)

        y_pred = torch.sigmoid(y_pred_logits)
        
        y, y_pred = y.float(), y_pred.float()
        loss = nn.BCELoss(reduction="mean")(y_pred, y)
        accuracy = get_accuracy(y_pred, y)
        batch_size = x.size(0)
        return loss, accuracy, batch_size
        
    def training_step(self, train_batch, batch_idx):
        loss, accuracy, batch_size = self.__compute_batch_loss(train_batch)
        self.log("train_loss", loss, batch_size=batch_size)
        self.log("train_accuracy", accuracy, batch_size=batch_size)
        return loss

    def validation_step(self, val_batch, batch_idx):
        loss, accuracy, batch_size = self.__compute_batch_loss(val_batch)
        self.log("val_loss", loss, batch_size=batch_size)
        self.log("val_accuracy", accuracy, batch_size=batch_size)

    def test_step(self, test_batch, batch_idx):
        loss, accuracy, batch_size = self.__compute_batch_loss(test_batch)
        self.log("test_loss", loss, batch_size=batch_size)
        self.log("test_accuracy", accuracy, batch_size=batch_size)