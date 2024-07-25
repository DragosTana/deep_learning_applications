import torch
import torchvision
import torchvision.transforms as transforms
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning import Trainer
from model import CNN
import argparse

def get_cifar10_transforms(train=True):
    if train:
        transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
    else:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
    
    return transform

def train(args):
    train_transform = get_cifar10_transforms(train=True)
    test_transform = get_cifar10_transforms(train=False)

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=train_transform)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=test_transform)

    batch_size = args.batch_size    
    
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=8)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=8)

    if args.log:
        wandb_logger = WandbLogger(project='CNN-CIFAR10')
    else:
        wandb_logger = None

    model = CNN(args=args, num_classes=10)

    trainer = Trainer(max_epochs=args.epochs, logger=wandb_logger)

    if args.train:
        trainer.fit(model, trainloader, testloader) 
        if args.save:
            torch.save(model.state_dict(), 'laboratory_4/cifar10_CNN.pth')

    if args.test:
        model.load_state_dict(torch.load('laboratory_4/cifar10_CNN.pth'))
        trainer.test(model, testloader)

    if args.confusion_matrix:
        from sklearn import metrics
        import numpy as np
        import matplotlib.pyplot as plt

        classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
        model.load_state_dict(torch.load('laboratory_4/cifar10_CNN.pth'))
        model.eval()

        y_true = []
        y_pred = []

        for images, labels in testloader:
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            y_true += labels.tolist()
            y_pred += predicted.tolist()

        cm = metrics.confusion_matrix(y_true, y_pred)
        cmn = cm.astype(np.float32)
        cmn /= cmn.sum(axis=1)
        cmn = (100*cmn).astype(np.int32)
        disp = metrics.ConfusionMatrixDisplay(cmn, display_labels=classes)
        disp.plot()
        plt.show()


if __name__ == '__main__':
    parsers = argparse.ArgumentParser(description='CIFAR-10 CNN')
    parsers.add_argument('--train', action='store_true', default=False,
                         help='Train the model (default: True)')
    parsers.add_argument('--test', action='store_true', default=False,
                            help='Test the model (default: False)')
    parsers.add_argument('--confusion_matrix', action='store_true', default=False,
                            help='Plot confusion matrix')
    parsers.add_argument('--batch_size', type=int, default=128,
                            help='Input batch size for training (default: 128)')
    parsers.add_argument('--epochs', type=int, default=20,
                            help='Number of epochs to train (default: 20)')
    parsers.add_argument('--lr', type=float, default=0.001,
                            help='Learning rate (default: 0.001)')
    parsers.add_argument('--log', action='store_true', default=False,
                            help='Enables logging of the loss and accuracy metrics to Weights & Biases.')
    parsers.add_argument('--save', action='store_true', default=False,
                            help='Enables saving of training checkpoints and final model weights.')
    args = parsers.parse_args()

    train(args)