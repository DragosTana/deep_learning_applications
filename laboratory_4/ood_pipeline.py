import torch
import torchvision
from torchvision.datasets import FakeData
from torchvision import transforms
import matplotlib.pyplot as plt
import torch.nn.functional as F
from model import CNN
from sklearn import metrics
from odin import compute_odin_scores, grid_search, ood_detector
import numpy as np
from torch.utils.data import Subset
import json
from torch.utils.data import Dataset, DataLoader
from sklearn.datasets import fetch_lfw_people
import argparse
from PIL import Image

# for reproducibility
np.random.seed(420)
torch.manual_seed(420)

class UniformNoiseDataset(Dataset):
    def __init__(self, size, image_size, transform=None):
        self.size = size
        self.image_size = image_size
        self.transform = transform

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        image = np.random.uniform(0, 1, self.image_size).astype(np.float32)
        image = (image * 255).astype(np.uint8)  
        image = Image.fromarray(image.transpose(1, 2, 0))  
        if self.transform:
            image = self.transform(image)
        return image, 0  

    
def visualize(args):

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)) ])

    batch_size = 2048

    # CIFAR10
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=8)

    # fakeset gaussian
    gaussian_set = FakeData(size=len(testset), image_size=(3, 32, 32), num_classes=10, transform=transform)
    fakeloader_gaussian = DataLoader(gaussian_set, batch_size=batch_size, shuffle=False, num_workers=8)

    # fakeset uniform
    uniform_set = UniformNoiseDataset(size=len(testset), image_size=(3, 32, 32), transform=transform)
    fakeloader_uniform = DataLoader(uniform_set, batch_size=batch_size, shuffle=False, num_workers=8)

    # caltech
    transform_caltech = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)) 
    ])
    #testset = torchvision.datasets.Food101(root='./data', split='test', transform=transform_caltech, download=True)
    #fakeloader_caltech = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=8)

    # CIFAR100 subset
    with open('laboratory_4/classes_not_in_cifar10.json', 'r') as file:
        data = json.load(file)
    cifar100_classes_not_in_cifar10 = data['cifar100_classes_not_in_cifar10']
    classes = np.random.choice(cifar100_classes_not_in_cifar10, 10, replace=False)
    classes = ["aquarium", "bycicle", "bottle", "bed", "rocket", "can", "girl", "chair"]
    cifar100 = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)
    selected_indices = [i for i, (_, label) in enumerate(cifar100) if cifar100.classes[label] in classes]
    custom_cifar100 = Subset(cifar100, selected_indices)
    cifar100_loader = torch.utils.data.DataLoader(custom_cifar100, batch_size=batch_size, shuffle=False, num_workers=8)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'


    if args.model == 'custom':
        model = CNN(args=None, num_classes=10)
        model.to(device)
        try:
            model.load_state_dict(torch.load('laboratory_4/cifar10_CNN.pth'))
        except:
            raise ValueError('Model not found, please train it first or use the pre-trained resnet20 model')
        model.eval()
    elif args.model == 'resnet':
        model = torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar10_resnet20", pretrained=True)
        model.to(device)
        model.eval()

    if args.compare == 'gaussian':
        fakeloader = fakeloader_gaussian
    elif args.compare == 'uniform':
        fakeloader = fakeloader_uniform
    elif args.compare == 'cifar100':
        fakeloader = cifar100_loader
    #elif args.compare == 'caltech':
    #    fakeloader = fakeloader_caltech

    # baseline
    scores_test = ood_detector(testloader, model, device)
    scores_fake = ood_detector(fakeloader, model, device)
    prediction = torch.cat((scores_test, scores_fake))
    target = torch.cat((torch.ones_like(scores_test), torch.zeros_like(scores_fake)))
    fpr, tpr, _ = metrics.roc_curve(target.cpu().numpy(), prediction.cpu().numpy())
    auc = metrics.auc(fpr, tpr)
    print(f'AUC baseline: {auc}')
    tpr_95_index = np.argmin(np.abs(tpr - 0.95))
    fpr_at_95_tpr = fpr[tpr_95_index]
    print(f'FPR at 95% TPR: {fpr_at_95_tpr}')

    if args.compare == 'gaussian':
        temp = 1000
        eps = 0.03
    elif args.compare == 'uniform':
        temp = 1000
        eps = 0.02
    elif args.compare == 'cifar100':
        temp = 1000
        eps = 0.02
    elif args.compare == 'caltech':
        temp = 1000
        eps = 0.02

    if args.grid_search:
        temperatures = [1, 10, 50, 100, 200, 500, 750, 1000]
        epsilons = torch.linspace(0, 0.05, 10)
        temp, eps, _ = grid_search(temperatures, epsilons, testloader, fakeloader, target, model, device)
        print(f'Best temperature: {temp}, Best epsilon: {eps}')

    # odin
    scores_test_odin = compute_odin_scores(testloader, model, temp, eps, device)
    scores_fake_odin = compute_odin_scores(fakeloader, model, temp, eps, device)
    prediction_odin = torch.cat((scores_test_odin, scores_fake_odin))
    fpr_odin, tpr_odin, _ = metrics.roc_curve(target.cpu().numpy(), prediction_odin.cpu().numpy())
    auc_odin = metrics.auc(fpr_odin, tpr_odin)
    print(f'AUC odin: {auc_odin}')
    tpr_95_index = np.argmin(np.abs(tpr_odin - 0.95))
    fpr_at_95_tpr_odin = fpr_odin[tpr_95_index]
    print(f'FPR at 95% TPR: {fpr_at_95_tpr_odin}')

    if args.roc:
        plt.plot(fpr, tpr, label='Max softmax')
        plt.plot(fpr_odin, tpr_odin, label='ODIN')
        plt.xlabel('False positive rate')
        plt.ylabel('True positive rate')
        plt.axhline(y=0.95, color='r', linestyle='--')
        plt.legend()
        plt.show()

    if args.pr:
        precision, recall, _ = metrics.precision_recall_curve(target.cpu().numpy(), prediction.cpu().numpy())
        precision_odin, recall_odin, _ = metrics.precision_recall_curve(target.cpu().numpy(), prediction_odin.cpu().numpy())
        plt.plot(recall, precision, label='Max softmax')
        plt.plot(recall_odin, precision_odin, label='ODIN')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.legend()
        plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='OOD')
    parser.add_argument('--compare', type=str, default='gaussian', choices=['gaussian', 'uniform', 'cifar100', 'caltech'],
                        help='Dataset to compare: gaussian, uniform, cifar100 (default: gaussian)')
    parser.add_argument('--grid_search', action='store_true', default=False,
                        help='Perform grid search for ODIN hyperparameters (default: False)')
    parser.add_argument('--roc', action='store_true', default=False,
                        help='Plot the ROC curve (default: False)')
    parser.add_argument('--pr', action='store_true', default=False,
                        help='Plot the PR curve (default: False)')
    parser.add_argument('--model', type=str, default='resnet', choices=['custom', 'resnet'],
                        help='Model to use: custom, resnet (default: resnet)')
    args = parser.parse_args()

    visualize(args)