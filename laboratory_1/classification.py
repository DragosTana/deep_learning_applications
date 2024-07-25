from __future__ import print_function
import torch
import argparse
import wandb
from torchvision import datasets, transforms
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from models.resnet import ResNet18
from models.mlp import MLP

def get_data_loaders(data, batch_size, test_batch_size, num_workers):
    """Prepare data loaders for training and validation."""
    if data == 'mnist':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
        test_dataset = datasets.MNIST('./data', train=False, transform=transform)
    elif data == 'cifar10':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
        ])
        train_dataset = datasets.CIFAR10('./data', train=True, download=True, transform=transform)
        test_dataset = datasets.CIFAR10('./data', train=False, transform=transform)
    else:
        raise ValueError(f"Unsupported dataset: {data}")

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=True, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=test_batch_size, num_workers=num_workers, pin_memory=True, shuffle=False)

    return train_loader, test_loader

def get_model(model_name, data_name, args):
    """Instantiate the appropriate model based on the provided arguments."""
    if model_name == 'cnn':
        if data_name == 'mnist':
            return ResNet18(num_blocks=[2, 2, 2, 2], input_channels=1, num_classes=10, args=args)
        elif data_name == 'cifar10':
            return ResNet18(num_blocks=[2, 2, 2, 2], input_channels=3, num_classes=10, args=args)
    elif model_name == 'mlp':
        if data_name == 'mnist':
            return MLP(args=args, input_size=28*28, num_class=10)
        elif data_name == 'cifar10':
            return MLP(args=args, input_size=3*32*32, num_class=10)
    else:
        raise ValueError(f"Unsupported model: {model_name}")

def main():
    parser = argparse.ArgumentParser(description='PyTorch vision classifier')
    parser.add_argument('--train', action='store_true', default=True,
                        help='Train the model (default: True)')
    parser.add_argument('--test', action='store_true', default=False,
                        help='Test the model (default: False)')
    parser.add_argument('--model', type=str, default='cnn', choices=['cnn', 'mlp'],
                        help='Model to use: cnn, mlp (default: cnn)')
    parser.add_argument('--data', type=str, default='mnist', choices=['mnist', 'cifar10'],
                        help='Dataset to use: mnist, cifar10 (default: mnist)')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Input batch size for training (default: 32)')
    parser.add_argument('--test-batch-size', type=int, default=1000,
                        help='Input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=14,
                        help='Number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=0.1,
                        help='Learning rate (default: 0.1)')
    parser.add_argument('--gamma', type=float, default=0.7,
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--seed', type=int, default=1,
                        help='Random seed (default: 1)')
    parser.add_argument('--num-workers', type=int, default=15,
                        help='Number of workers for data loader (default: 15)')
    parser.add_argument('--log', action='store_true', default=True,
                        help='Enables logging of the loss and accuracy metrics to Weights & Biases.')
    parser.add_argument('--save', action='store_true', default=False,
                        help='Enables saving of training checkpoints and final model weights.')
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    train_loader, test_loader = get_data_loaders(args.data, args.batch_size, args.test_batch_size, args.num_workers)

    if args.log:
        wandb_logger = WandbLogger(project='Classification_ResNet')
    else:
        wandb_logger = None

    model = get_model(args.model, args.data, args)

    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath='./runs/',
        filename='{epoch:02d}-{val_loss:.2f}',
        save_top_k=3,
        mode='min',
    ) if args.save else None

    trainer = Trainer(
        logger=wandb_logger,
        max_epochs=args.epochs,
        enable_checkpointing=args.save,
        default_root_dir='./runs/',
        callbacks=[checkpoint_callback] if checkpoint_callback else None
    )

    if args.train:
        trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=test_loader)
        if args.save:
            torch.save(model.state_dict(), f'{args.data}_{args.model}.pth')

    if args.test:
        trainer.test(model, test_dataloaders=test_loader)

if __name__ == '__main__':
    main()