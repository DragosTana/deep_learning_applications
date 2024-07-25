import matplotlib.pyplot as plt
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from laboratory_1.models import LitFCN
from lightning.pytorch.loggers import WandbLogger
from pytorch_lightning import Trainer
import torch.nn.functional as F
import torch
import random

class MNISTSegmentationDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        # Here, the label is used as the segmentation target
        segmentation_label = (image > 125).float()  # binary segmentation mask
        return image, segmentation_label, label

def plot_image_and_segmentation(image, segmentation_label, label):
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))

    axs[0].imshow(image.squeeze(), cmap='gray')
    axs[0].set_title(f'Original Image - Label: {label}')
    axs[0].axis('off')

    # Define a color map for segmentation labels
    colors = plt.cm.get_cmap('tab10', 10)
    colored_segmentation = np.zeros((*segmentation_label.squeeze().shape, 3))
    segmentation_mask = segmentation_label.squeeze().numpy()

    for i in range(10):
        colored_segmentation[segmentation_mask == i] = colors(i / 10.0)[:3]

    axs[1].imshow(colored_segmentation)
    axs[1].set_title(f'Segmentation Label - Label: {label}')
    axs[1].axis('off')

    plt.show()
    
mnist_train = datasets.MNIST('./data', train=True, download=True, transform=transforms.ToTensor())
mnist_val = datasets.MNIST('./data', train=False, transform=transforms.ToTensor())

train_dataset = MNISTSegmentationDataset(mnist_train)
val_dataset = MNISTSegmentationDataset(mnist_val)


model = LitFCN()
state_dict = torch.load('fcn_mnist_segmentation.pth')
model.load_state_dict(state_dict)

indx = random.randint(0, len(val_dataset))
example_image, example_segmentation, label = val_dataset[indx]
example_image = example_image.unsqueeze(0)
output = model(example_image).detach()
fig, axs = plt.subplots(1, 2, figsize=(10, 5))
axs[0].imshow(example_image.squeeze(), cmap='gray')
axs[0].set_title(f'Original Image - Label: {label}')
axs[0].axis('off')
axs[1].imshow(output.detach().squeeze(0).squeeze(0), cmap='gray')
axs[1].set_title(f'Segmentation Output - Label: {label}')
axs[1].axis('off')
plt.show()  