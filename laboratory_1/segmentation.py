import matplotlib.pyplot as plt
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from src.segmentation import MNISTSegmentationModel
from pytorch_lightning.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from pytorch_lightning import Trainer
import torch.nn.functional as F
import torch

import argparse

class MNISTSegmentationDataset(Dataset):
    def __init__(self, mnist_data, transform=None):
        self.mnist_data = mnist_data
        self.transform = transform

    def __len__(self):
        return len(self.mnist_data)

    def __getitem__(self, idx):
        image, label = self.mnist_data[idx]
        image = np.array(image)
        label_map = np.zeros_like(image, dtype=np.longlong)  # Initialize with background class (0)
        label_map[image > 0] = label + 1  # Assign digit class (1-10) to non-background pixels

        if self.transform:
            image = self.transform(image)
            label_map = torch.tensor(label_map, dtype=torch.long)

        return image, label_map

def get_data_loaders(batch_size, test_batch_size, num_workers):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_dataset = datasets.MNIST('./data', train=True, download=True)
    test_dataset = datasets.MNIST('./data', train=False)

    train_dataset = MNISTSegmentationDataset(train_dataset, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    test_dataset = MNISTSegmentationDataset(test_dataset, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False)

    return train_loader, test_loader

def visualize_results(model, dataloader, num_samples=10):
    model.eval()
    images, labels = next(iter(dataloader))
    images = images[:num_samples]
    labels = labels[:num_samples]

    with torch.no_grad():
        outputs = model(images)

    fig, axs = plt.subplots(num_samples, 3, figsize=(10, num_samples * 3))
    for i in range(num_samples):
        img = images[i].squeeze().cpu().numpy()
        true_mask = labels[i].cpu().numpy()
        pred_mask = outputs[i].argmax(dim=0).cpu().numpy()

        axs[i, 0].imshow(img, cmap='gray')
        axs[i, 0].set_title('Input Image')
        axs[i, 0].axis('off')

        axs[i, 1].imshow(true_mask, cmap='viridis')
        axs[i, 1].set_title('True Mask')
        axs[i, 1].axis('off')

        axs[i, 2].imshow(pred_mask, cmap='viridis')
        axs[i, 2].set_title('Predicted Mask')
        axs[i, 2].axis('off')

    plt.tight_layout()
    plt.show()

def main():
    parser = argparse.ArgumentParser(description='PyTorch vision classifier')
    parser.add_argument('--train', action='store_true', default=True,
                        help='Train the model (default: True)')
    parser.add_argument('--test', action='store_true', default=False,
                        help='Test the model (default: False)')
    parser.add_argument('--visualize', action='store_true', default=False,
                        help='Randomly samples 10 images from the test set and visualize the resul of the segmentation model (default: False)')
    parser.add_argument('--batch-size', type=int, default=128,
                        help='Input batch size for training (default: 128)')
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
    parser.add_argument('--log', action='store_true', default=False,
                        help='Enables logging of the loss and accuracy metrics to Weights & Biases.')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='Enables saving of training checkpoints and final model weights.')
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    train_loader, val_loader = get_data_loaders(args.batch_size, args.test_batch_size, args.num_workers)

    if args.log:
        wandb_logger = WandbLogger(project='pytorch-vision')
    else:
        wandb_logger = None

    model = MNISTSegmentationModel(num_classes=10, args=args)

    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath='./runs/',
        filename='{epoch:02d}-{val_loss:.2f}',
        save_top_k=3,
        mode='min',
    ) if args.save_model else None

    trainer = Trainer(
        logger=wandb_logger,
        max_epochs=args.epochs,
        enable_checkpointing=args.save_model,
        default_root_dir='./runs/',
        callbacks=[checkpoint_callback] if checkpoint_callback else None
    )

    if args.train:
        trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
        if args.save_model:
            torch.save(model.state_dict(), f'{args.data}_{args.model}.pth')

    if args.test:
        model.load_state_dict(torch.load(f'{args.data}_{args.model}.pth'))
        trainer.test(model, test_dataloaders=val_loader)

    if args.visualize:
        model.load_state_dict(torch.load(f'{args.data}_{args.model}.pth'))
        visualize_results(model, val_loader)

if __name__ == '__main__':
    main()