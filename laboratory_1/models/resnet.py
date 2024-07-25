import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import pytorch_lightning as pl
from lightning.pytorch.utilities import grad_norm
from typing import Optional, Callable
from torch import Tensor
import numpy as np


class BasicBlock(nn.Module):
    def __init__(self, in_planes, planes, stride=1, use_skip_connection=True):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.use_skip_connection = use_skip_connection

        if use_skip_connection:
            if stride != 1 or in_planes != planes:
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(planes)
                )
            else:
                self.shortcut = nn.Identity() #this is needed to log the gradients of the shortcut connection

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.use_skip_connection:
            out += self.shortcut(x)
        out = F.relu(out)
        return out
    
    
class ResNet18_torch(nn.Module):
    def __init__(self, num_blocks, input_channels = 1, num_classes=10, use_skip_connection=True):
        super(ResNet18_torch, self).__init__()
        self.in_planes = 64
        self.use_skip_connection = use_skip_connection

        self.conv1 = nn.Conv2d(input_channels, self.in_planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_planes)
        self.layer1 = self._make_layer(BasicBlock, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(BasicBlock, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(BasicBlock, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(BasicBlock, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)  # [stride, 1, 1, ...] (num_blocks times) ensures that the first block has stride=stride
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, self.use_skip_connection))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))   # [batch_size, 3, 32, 32] -> [batch_size, 64, 32, 32]
        out = self.layer1(out)                  # [batch_size, 64, 32, 32] -> [batch_size, 64, 32, 32] (stride=1)
        out = self.layer2(out)                  # [batch_size, 64, 32, 32] -> [batch_size, 128, 16, 16] (stride=2)
        out = self.layer3(out)                  # [batch_size, 128, 16, 16] -> [batch_size, 256, 8, 8] (stride=2)
        out = self.layer4(out)                  # [batch_size, 256, 8, 8] -> [batch_size, 512, 4, 4] (stride=2)
        out = F.avg_pool2d(out, 4)              # [batch_size, 512, 4, 4] -> [batch_size, 512, 1, 1] (avg_pool2d)
        out = out.view(out.size(0), -1)         # [batch_size, 512, 1, 1] -> [batch_size, 512]
        out = self.linear(out)                  # [batch_size, 512] -> [batch_size, num_classes]
        return out


class ResNet18(pl.LightningModule):
    def __init__(self, num_blocks: list, input_channels: int = 1,  num_classes: int = 10, args: Optional[Callable] = None, use_skip_connection: bool = True):
        super().__init__()
        self.model = ResNet18_torch(num_blocks, input_channels, num_classes, use_skip_connection)
        self.args = args
        hyperparam = dict(num_blocks=num_blocks, num_classes=num_classes)
        self.save_hyperparameters(hyperparam)
        
        
    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)
    
    def training_step(self, batch: tuple, batch_idx: int):
        x, y = batch
        y_hat = self.model(x)
        loss = F.cross_entropy(y_hat, y)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("train_acc", (y_hat.argmax(1) == y).float().mean(), on_epoch=True, prog_bar=True, logger=True)
        return loss
    
    def validation_step(self, batch: tuple, batch_idx: int):
        x, y = batch
        y_hat = self.model(x)
        loss = F.cross_entropy(y_hat, y)
        self.log("val_loss", loss, on_epoch=True, logger=True)
        self.log("val_acc", (y_hat.argmax(1) == y).float().mean(),  on_epoch=True, logger=True)
        return loss
    
    def test_step(self, batch: tuple, batch_idx: int):
        x, y = batch
        y_hat = self.model(x)
        loss = F.cross_entropy(y_hat, y)
        self.log("test_loss", loss, on_epoch=True, logger=True)
        self.log("test_acc", (y_hat.argmax(1) == y).float().mean(),  on_epoch=True, logger=True)
        return loss
    
    def predict_step(self, batch: tuple, batch_idx: int, dataloader_idx: int):
        x, y = batch
        y_hat = self.model(x)
        return y_hat
    
    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=self.args.lr, momentum=0.9, weight_decay=5e-4)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
        return{"optimizer": optimizer, "lr_scheduler": self.scheduler, "monitor": "train_loss"}
    
    def on_before_optimizer_step(self, optimizer):
        gradients = {"shortcut": 0, "weights": 0}
        grad_n = 0
        grad_shortcut_n = 0
        global_step = self.global_step
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                if "shortcut" in name:
                    gradients["shortcut"] += torch.norm(param.grad, p=2).item()
                    grad_shortcut_n += 1
                else:
                    gradients["weights"] += torch.norm(param.grad, p=2).item()
                    grad_n += 1
        
        # Calculate the gradient of identity shortcuts
        if self.model.use_skip_connection:
            for module in self.model.modules():
                if isinstance(module, BasicBlock):
                    if isinstance(module.shortcut, nn.Identity):
                        if module.use_skip_connection:
                            # The gradient of the identity shortcut is the same as the gradient
                            # flowing back from the addition operation
                            identity_grad = module.conv2.weight.grad
                            gradients["shortcut"] += torch.norm(identity_grad, p=2).item()
                            grad_shortcut_n += 1
    
        # Compute averages
        gradients["shortcut"] = gradients["shortcut"] / grad_shortcut_n if grad_shortcut_n > 0 else 0
        gradients["weights"] = gradients["weights"] / grad_n if grad_n > 0 else 0
    
        for key, value in gradients.items():
            self.log(key, value, on_step=True)
        
        