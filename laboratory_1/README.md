# On the coolness of Convolutional Neural Networks

This aims to be a study on convolution neural networks, how to stuck them in cool architectures, how to make theese architectures work and get some insights in what they are doing.
The focus is:
- Understand why (and how!!) residual connections work.
- Get a grasp on what CNNs are doing under the hood thanks to [Grad-CAM](https://arxiv.org/abs/1610.02391).
- Creating a classification pipeline.

Despite my initial enthusiasm, I accepted the fact that I won't be able to compete with Lightning AI and opted to use their framework, [Python Lightning](https://lightning.ai/docs/pytorch/stable/), instead of coding my own. 


## Classification

Taking for granted what classification is in the context of CV, I use a simple [MLP](/laboratory_1/src/mlp.py) to get a baseline for my classification pipeline. To improve performance, a simple [ResNet](/laboratory_1/src/resnet.py) was implemented following the [original implementation](https://arxiv.org/abs/1512.03385). This was a good thing since I finally understood how dimensions of feature maps across the various layer were handled when using skip connnections. Both models were converted to LightningModules and provided with the additional methods required to make them work with the PyTorch Lightning Trainer.


Performance measured as accuracy is reported in the following table.

<div align="center">

|       |Cifar10| MNIST |
|:-----:|:-----:|:-----:|
|MLP       | 52.47%  | 97.19% |
|ResNet    | 86.36%  | 99.6%  |


</div>

### Usage
To run the script, use the following command:

```python3 classification.py [arguments]```

- `--train`: Train the model (default: True)
- `--test`: Test the model (default: False)
- `--model`: Model to use: cnn or mlp (default: cnn)
- `--data`: Dataset to use: mnist or cifar10 (default: mnist)
- `--batch-size`: Input batch size for training (default: 128)
- `--test-batch-size`: Input batch size for testing (default: 1000)
- `--epochs`: Number of epochs to train (default: 14)
- `--lr`: Learning rate (default: 0.1)
- `--gamma`: Learning rate step gamma (default: 0.7)
- `--seed`: Random seed (default: 1)
- `--num-workers`: Number of workers for data loader (default: 16)
- `--log`: Enables logging of the loss and accuracy metrics to Weights & Biases (default: False)
- `--save-model`: Enables saving of training checkpoints and final model weights (default: False)

For example to Train an MLP model on CIFAR-10 dataset for 20 epochs:

```python classification.py --model mlp --data cifar10 --epochs 20```

## On skip connections
A skip connection, as described in the paper ["Deep Residual Learning for Image Recognition"](https://arxiv.org/abs/1512.03385) is a type of shortcut connection used in neural networks where the input of a layer is added directly to the output of a subsequent layer, effectively skipping over intermediate layers. This technique allows the network to learn residual functions more effectively by facilitating the flow of gradients during training, thus mitigating issues like vanishing or exploding gradients and enabling the training of much deeper networks.

The reasons why skip connections work are:
- Mitigating the Vanishing Gradient Problem.
- Addressing Degradation Problem.
- Encouraging Feature Reuse.
- Easing Optimization.
- and many more...

To understand the effect of skip connections, the first step was to analyze the performance of the network with and without skip connections under the same settings. In both cases, the models were trained on the CIFAR-10 dataset using a learning rate of 0.01 and a batch size of 128. As we can see, when using skip connections, the network not only converges faster but also achieves higher accuracy on the test set.

<center>

|                           |Skip connection   |No skip connection   |
|:-------------------------:|:----------------:|:-------------------:|
|Accuracy                   |   86.36          |           80.81     |

</center>

The velocity of convergence was analyzed by simply looking at the loss functions of the two models. Furthermore if we take a look at the gradient magnitudes we can clearly see that when using skip connnection the gradients of the convolutional layers tend to be smaller and less noisy. This confirms the idea that [residual connections facilitate the optimization process by smoothing the loss surface](https://papers.nips.cc/paper_files/paper/2018/hash/a41b3bb3e6b050b6c9067c67f663b915-Abstract.html), suggesting that the skip connections could play as regularizations in training a deep neural network

 Loss |   Gradients |
|:----------------------------:|:-------------------------:|
|![](/laboratory_1/doc/LossNetworks.png)  |  ![](/laboratory_1/doc/GradientsNetworks.png)|


Another interesting aspect to observe was the differing magnitude of the gradients between the convolutional layers and the skip connections in the ResNet. As shown in the image below, the gradients of the skip connections tend to be higher in the initial phases of the training process. However, this difference diminishes as training progresses and the computationally intensive layers begin to kick in.

<center>

|![](/laboratory_1/doc/GradientResNet.png)|

</center>


## GradCAM
Since GradCAM is essentially model-agnostic I decided to work with a pretrained VGG16 provided by torchvision. My reasons were: first, I wanted to experiment with the pretrained networks available from torchvision, and second, the feature maps of the VGG16 before the fully connected layers have a dimension of 14x14. This is slightly larger than the 4x4 feature maps in my ResNet18 implementation and provided a more visually appealing result when upscaling the heatmap. To visualize the results, I used various [samples from ImageNet](https://github.com/EliSchwartz/imagenet-sample-images/tree/master). However, feel free to use any image you prefer as long as it can be classified with one of the classes from ImageNet. To check the ImageNet classes please reference this [page](https://deeplearning.cms.waikato.ac.nz/user-guide/class-maps/IMAGENET/).

### Usage

To run the script, use the following command:

```python3 grad_cam.py [--image_path IMAGE_PATH] [--class_idx CLASS_IDX]```

- `--image_path`: URL to the image to visualize.
- `--class_idx`: Class index to visualize. If None the class predicted by the VGG16 will be used.

Or simply run the ```grad_cam.py``` script in your IDE.

### Examples and results

GradCam does more or less what we expect: it shows us which features we are using to make the prediction through a saliency map (I'm trying really hard not to use the word _attention_ here). The most interesting thing to observe is how the heatmap changes when we provide different class indices in images that have multiple classes, like the one shown here. When class_idx is None, we are using the VGG16 prediction. As we see, the network correctly predicts [Kuvasz](https://en.wikipedia.org/wiki/Kuvasz), and the heatmap highlights the dog's face. However, when we provide class_idx 281, which corresponds to the tabby cat class, the network uses the features belonging to the cat's face instead.

class_idx: 281 (tabby cat) |  class_idx: None
:-------------------------:|:-------------------------:
![](/laboratory_1/doc/cat.png)  |  ![](/laboratory_1/doc/dog.png)





## Notes for me
- PyTorch Lightning and Weights and Biases integrates flawlesly.
- Grad-CAM is one of the coolest thing ever.
- Setting the warmup in PyTorch Lightning is a pain in the ass.

