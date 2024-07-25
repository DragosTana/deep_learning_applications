import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

class MLP(pl.LightningModule):
    """Over engineered MLP model."""
    
    def __init__(self,args, input_size=3*32*32, num_class=10):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(in_features = input_size, out_features=512)
        self.fc2 = nn.Linear(in_features = 512, out_features=512)
        self.fc3 = nn.Linear(in_features = 512, out_features=256)
        self.fc4 = nn.Linear(in_features = 256, out_features=256)
        self.fc5 = nn.Linear(in_features = 256, out_features=128)
        self.dropout = nn.Dropout(p=0.5)
        self.fc_last = nn.Linear(in_features = 128, out_features=num_class)
        self.auxil = nn.Linear(in_features = 512, out_features=num_class)
        self.batchnorm = nn.BatchNorm1d(512)
        self.batchnorm2 = nn.BatchNorm1d(256)
        self.batchnorm3 = nn.BatchNorm1d(128)
        self.args = args


    def forward(self, x):
        x = torch.flatten(x, 1)
        first = self.batchnorm(F.gelu(self.fc1(x)))   
        second = self.batchnorm(F.gelu(self.fc2(first))) + first
        third = self.batchnorm2(F.gelu(self.fc3(second)))
        fourth = self.batchnorm2(F.gelu(self.fc4(third))) + third
        x = self.batchnorm3(F.gelu(self.fc5(fourth)))
        x = self.dropout(x)
        out = self.fc_last(x)
        return out


    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("train_acc", (y_hat.argmax(1) == y).float().mean(), on_epoch=True, prog_bar=True, logger=True)
        return loss
    
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        self.log("val_loss", loss, on_epoch=True, logger=True)
        self.log("val_acc", (y_hat.argmax(1) == y).float().mean(),  on_epoch=True, logger=True)
        return loss
    
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        self.log("test_loss", loss, on_epoch=True, logger=True)
        self.log("test_acc", (y_hat.argmax(1) == y).float().mean(),  on_epoch=True, logger=True)
        return loss
    

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.args.lr, betas=(0.9, 0.99), weight_decay=5e-5)
        self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 300, eta_min=1e-6, last_epoch=-1, verbose=False)
        return{"optimizer": optimizer, "lr_scheduler": self.lr_scheduler, "monitor": "train_loss"}
 
    



