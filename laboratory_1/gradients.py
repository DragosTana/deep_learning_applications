from models.resnet import ResNet18
import torch

from torchvision import transforms, datasets
from lightning.pytorch.loggers import WandbLogger
from pytorch_lightning import Trainer
import argparse

def main():

    parser = argparse.ArgumentParser(description='PyTorch vision clssifier')
    parser.add_argument('--model', type=str, default='cnn', metavar='M',
                        help='model to use. Options: cnn, mlp (default: cnn)')
    parser.add_argument('--data', type=str, default='cifar10', metavar='D',
                        help='dataset to use. Options: mnist, cifar10 (default: cifar10)')
    parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=14, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.1)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--num-workers', type=int, default=15, metavar='N',
                        help='number of workers for data loader (default: 15)')
    parser.add_argument('--log', action='store_true', default=True,
                        help='Enables logging of the loss and accuracy metrics to Weights & Biases.')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='Enables saving of training checkpoints and final model weights. \
                        Useful for resuming training or model deployment')
    args = parser.parse_args()

    #transform = transforms.Compose([
    #    transforms.ToTensor(),
    #    transforms.Normalize((0.1307,), (0.3081,))
    #])
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
    ])

    #dataset1 = datasets.MNIST('./data', train=True, download=True, transform=transform)
    #dataset2 = datasets.MNIST('./data', train=False, transform=transform)
    dataset1 = datasets.CIFAR10('./data', train=True, download=True, transform=transform)
    dataset2 = datasets.CIFAR10('./data', train=False, transform=transform)


    train_loader = torch.utils.data.DataLoader(dataset1, batch_size = 128, num_workers = 16)
    val_loader = torch.utils.data.DataLoader(dataset2, batch_size = 128, num_workers = 16)

    wandb_logger = WandbLogger(project='Gradients')
    epochs = 50
    path = "./runs/"
    model = ResNet18(num_blocks=[2, 2, 2, 2], input_channels=3, num_classes=10, args=args, use_skip_connection=False)
    print(model)

    trainer = Trainer(logger=wandb_logger,
                      max_epochs=epochs,
                      enable_checkpointing=True,
                      default_root_dir=path)
    
    trainer = Trainer(max_epochs=epochs,
                        enable_checkpointing=True,
                        default_root_dir=path)

    trainer.fit(model=model, 
                train_dataloaders=train_loader, 
                val_dataloaders=val_loader)

if __name__ == "__main__":
    main()