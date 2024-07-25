import torch
from torch import nn
import torch.nn.functional as F
from torchvision import transforms
import torchvision
import matplotlib.pyplot as plt
import numpy as np
import argparse 
from model import CNN

class NormalizeInverse(torchvision.transforms.Normalize):
    
    def __init__(self, mean, std):
        mean = torch.as_tensor(mean)
        std = torch.as_tensor(std)
        std_inv = 1 / (std + 1e-7)
        mean_inv = -mean * std_inv
        super().__init__(mean=mean_inv, std=std_inv)

    def __call__(self, tensor):
        return super().__call__(tensor.clone())

def fgsm_attack(model, 
                loss_fn, 
                data, 
                epsilon, 
                ground_truth,
                target=None,
                max_iterations=100):
    """
    Perform a Fast Gradient Sign Method attack on the given model.
    ### Arguments:
        - model: The model to attack.
        - loss_fn: The loss function to use.
        - data: The input data to perturb.
        - epsilon: The perturbation amount.
        - ground_truth: The ground truth label of the input data.
        - target: The target label for the attack (None for untargeted attack).
        - max_iterations: The maximum number of iterations to perform.
    ### Returns:
        - original: The original input data.
        - perturbed: The perturbed input data.
    """

    device = data.device
    
    if target is not None:
        target = torch.tensor([target]).to(device)
        if len(target.shape) == 0:
            target = target.unsqueeze(0)

    if len(data.shape) == 3:
        data = data.unsqueeze(0)
    if len(ground_truth.shape) == 0:
        ground_truth = ground_truth.unsqueeze(0)
    
    original = data.clone().detach()
    perturbed = data.clone().detach().requires_grad_(True)

    for i in range(max_iterations):
        output = model(perturbed)
        model.zero_grad()

        if target is None:
            # Untargeted attack
            loss = loss_fn(output, ground_truth)
            if output.argmax().item() != ground_truth.item():
                #print(f"Attack succeeded after {i+1} iterations.")
                #print(f'Prediction: {output.argmax().item()}')
                return original, perturbed.detach()
        else:
            # Targeted attack
            loss = -loss_fn(output, target)  # Negative loss for targeted attack
            if output.argmax().item() == target.item():
                #print(f"Targeted attack succeeded after {i+1} iterations.")
                return original, perturbed.detach()

        loss.backward()
        
        # Update the image
        with torch.no_grad():
            perturbed += epsilon * torch.sign(perturbed.grad)
        
        perturbed.requires_grad_(True)
        #print(f'Iteration {i+1}, Loss: {loss.item()}, Prediction: {output.argmax().item()}')


    #print("Attack did not succeed within the maximum number of iterations.")
    return original, perturbed.detach()
    

def visualize_fgsm_attack(args):
    """
    Simple function to visualize the FGSM attack.
    """

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = CNN(args=None, num_classes=10)

    try:
        model.load_state_dict(torch.load('laboratory_4/cifar10_CNN.pth'))
    except:
        raise FileNotFoundError('Model not found. Please train the model first.')
    
    model.to(device)

    transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))])

    inv = NormalizeInverse((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))

    batch_size = 16

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=True, num_workers=8)

    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    example = next(iter(testloader))
    idx = np.random.randint(0, batch_size)
    image = example[0][idx]
    label = example[1][idx]

    original, adversarial = fgsm_attack(model,
                                        nn.CrossEntropyLoss(),
                                        image.unsqueeze(0).to(device),
                                        args.epsilon,
                                        label.to(device), 
                                        target=args.target, 
                                        max_iterations=args.max_iterations)

    fig, ax = plt.subplots(1, 3)
    ax[0].imshow(inv(original.squeeze(0)).permute(1, 2, 0).detach().cpu().numpy())
    ax[0].set_title(f'Original: {classes[label]}')
    ax[0].axis('off')

    if adversarial is not None:
        fig.suptitle('Adversarial attack')
    else:
        fig.suptitle('Classifier already wrong')

    if adversarial is not None:
        ax[1].imshow(inv(adversarial.squeeze(0)).permute(1, 2, 0).detach().cpu().numpy())
        ax[1].set_title(f'Adversarial: {classes[model(adversarial).argmax().item()]}')
        ax[1].axis('off')

        ax[2].imshow(inv(adversarial - original).squeeze(0).permute(1, 2, 0).detach().cpu().numpy())
        ax[2].set_title('Difference')
        ax[2].axis('off')

    else:
        ax[1].set_title(f'Predicted: {classes[model(original).argmax().item()]}')
        ax[1].axis('off')
        ax[1].imshow(inv(original.squeeze(0)).permute(1, 2, 0).detach().cpu().numpy())  

        ax[2].axis('off')
        ax[2].imshow(torch.zeros_like(original.squeeze(0)).permute(1, 2, 0).detach().cpu().numpy())
        ax[2].set_title('Difference')

    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epsilon', type=float, default=0.01)
    parser.add_argument('--target', type=int, default=None)
    parser.add_argument('--max_iterations', type=int, default=10)
    
    args = parser.parse_args()
    visualize_fgsm_attack(args)