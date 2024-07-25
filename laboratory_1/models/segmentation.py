import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

#NOTE: add log capability

class MNISTSegmentationModel(pl.LightningModule):
    def __init__(self, num_classes, args = None):
        super().__init__()
        self.args = args
        self.num_classes = num_classes  

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),     # [batch_size, 1, 28, 28] -> [batch_size, 64, 28, 28]
            nn.BatchNorm2d(64),                                 
            nn.ReLU(inplace=True),                              
            nn.MaxPool2d(kernel_size=2, stride=2),          # [batch_size, 64, 28, 28] -> [batch_size, 64, 14, 14]    

            nn.Conv2d(64, 128, kernel_size=3, padding=1),   # [batch_size, 64, 14, 14] -> [batch_size, 128, 14, 14]
            nn.BatchNorm2d(128),                            
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),          # [batch_size, 128, 14, 14] -> [batch_size, 128, 7, 7]

            nn.Conv2d(128, 256, kernel_size=3, padding=1),  # [batch_size, 128, 7, 7] -> [batch_size, 256, 7, 7]
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),          # [batch_size, 256, 7, 7] -> [batch_size, 256, 3, 3]
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),          # [batch_size, 256, 3, 3] -> [batch_size, 128, 7, 7]
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),                                  

            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),           # [batch_size, 128, 7, 7] -> [batch_size, 64, 14, 14]
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(64, 1, kernel_size=2, stride=2),   # [batch_size, 64, 14, 14] -> [batch_size, 1, 28, 28]
            nn.BatchNorm2d(num_classes),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return F.softmax(x, dim=1)
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        return loss
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.args.lr)
        return optimizer




