import torch
import torch.nn.functional as F
from torchvision import models, transforms
from torchvision import datasets
from PIL import Image
import requests
import numpy as np
import cv2
import matplotlib.pyplot as plt
import argparse

class GradCAM:
    """
    GradCAM implementation in PyTorch
    """
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0]

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_backward_hook(backward_hook)

    def generate_heatmap(self, input_tensor, class_idx=None):
        self.model.eval()
        output = self.model(input_tensor)

        if class_idx is None:
            class_idx = torch.argmax(output)

        self.model.zero_grad()
        output[0, class_idx].backward()

        gradients = self.gradients.cpu().data.numpy()[0]
        activations = self.activations.cpu().data.numpy()[0]

        weights = np.mean(gradients, axis=(1, 2))
        cam = np.zeros(activations.shape[1:], dtype=np.float32)

        for i, w in enumerate(weights):
            cam += w * activations[i]

        cam = np.maximum(cam, 0)  # Apply ReLU 
        cam = cv2.resize(cam, (input_tensor.shape[2], input_tensor.shape[3]))
        cam = cam - np.min(cam)
        cam = cam / np.max(cam)
        return cam, class_idx

def preprocess_image(image_path):
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    img = Image.open(requests.get(image_path, stream=True).raw).convert('RGB')
    img = preprocess(img).unsqueeze(0)
    return img

def show_cam_on_image(img, mask):
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cam = heatmap + np.float32(img)
    cam = cam / np.max(cam)
    cam_img = np.uint8(255 * cam)
    return cam_img


def main():
    parser = argparse.ArgumentParser(description='PyTorch vision classifier')
    parser.add_argument('--image_path', type=str, default='https://imgs.classicfm.com/images/33669?crop=16_9&width=660&relax=1&format=webp&signature=2Ibku5850P02pF5YbYc84-U3ocg=',
                        help='URL to the image to visualize')
    parser.add_argument('--class_idx', type=int, default=None, #281
                        help='Class index to visualize (default: None), plese reference https://deeplearning.cms.waikato.ac.nz/user-guide/class-maps/IMAGENET/')
    args = parser.parse_args()
    model = models.vgg16(weights='IMAGENET1K_V1')
    target_layer = model.features[28]
    grad_cam = GradCAM(model, target_layer)

    input_tensor = preprocess_image(args.image_path)
    mask, class_idx = grad_cam.generate_heatmap(input_tensor, args.class_idx)

    # Load ImageNet class labels
    LABELS_URL = 'https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/master/imagenet-simple-labels.json'
    labels = requests.get(LABELS_URL).json()
    predicted_class_label = labels[class_idx]

    img = input_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
    img = img - np.min(img)
    img = img / np.max(img)

    cam_img = show_cam_on_image(img, mask)
    plt.imshow(cam_img)
    plt.axis('off')
    plt.title(f'Predicted Class: {predicted_class_label}')
    plt.show()
 
if __name__ == '__main__':
    main()