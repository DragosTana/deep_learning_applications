# OOD detection and adversarial attacks

Here we are focusing on two topics that seem very different but, as it turns out, are related: out of distribution detection and adversarial attacks. The main objectives here were:
    - Implement and evaluate with the ODIN detector.
    - Implement and evaluate the FGSM algorithm both in targeted and untargeted forms and evaluate its performance.

## ODIN detector

The implementation of the ODIN detector was based on the [Enhancing The Reliability of Out-of-distribution Image Detection in Neural Networks](https://arxiv.org/abs/1706.02690) paper. The main  observation of this work is that using temperature scaling and adding small perturbations to the input can separate the softmax score distributions between in- and out-of-distribution images, allowing for more effective detection. This is in contrast with the work of [Hendrycks & Gimpel, 2017](https://arxiv.org/abs/1610.02136) which utilizes only the softmax score distributions. The idea here was that correctly classified examples tend to have greater maximum softmax probabilities than erroneously classified and out-of-distribution examples, allowing for their detection. Both methods are implemented and evaluated confirming that ODIN tends to outperform the baseline of Hendrycks & Gimpel.

### Models

Two different models were used to test the OOD detection pipeline.

- A small custom CNN.
- A [pretrained ResNet](https://github.com/chenyaofo/pytorch-cifar-models) from PyTorch Hub. 

To train, evaluate or show the confusion matrix of the custom CNN use the ```train.py``` script as follows:

```python3 train.py [arguments]```

Arguments:

- ```--train```: Train the model (default: False)
- ```--test```: Test the model (default: False)
- ```--confusion_matrix```: Plot confusion matrix (default: False)
- ```--batch_size``` BATCH_SIZE: Input batch size for training (default: 128)
- ```--epochs``` EPOCHS: Number of epochs to train (default: 20)
- ```--log```: Enables logging of the loss and accuracy metrics to Weights & Biases (default: True)
- ```--save```: Enables saving of training checkpoints and final model weights (default: False)

The model was trained on CIFAR10 with the following hyperparameters:

- epochs: 50
- batch size: 128
- learning rate: 1e-3

and it achives an accuracy of 83.29%


### Setup
At test time, the test images from CIFAR-10 dataset can be viewed as the in-distribution (positive) examples. For out-of-distribution (negative) examples the same setting of the paper is followed and test on natural image datasets and synthetic noise datasets. I consider the following out-of-distribution test datasets.

- Uniform Noise. The synthetic uniform noise dataset consists of 10,000 images where each RGB
value of every pixel is independently and identically sampled from a uniform distribution on [0, 1].
- Gaussian Noise. The synthetic Gaussian noise dataset consists of 10,000 random 2D Gaussian
noise images, where each RGB value of every pixel is sampled from an i.i.d Gaussian distribution with mean 0 and unit variance. We further clip each pixel value into the range [0, 1].
- CIFAR100 subset. Natural image dataset containing a subset of CIFAR100. Only the image belonging to a class different from the ones contained in CIFAR10 were included.

I adopt the following two different metrics to measure the effectiveness of a neural network in
distinguishing in- and out-of-distribution images.

- FPR at 95% TPR can be interpreted as the probability that a negative (out-of-distribution) example is misclassified as positive (in-distribution) when the true positive rate (TPR) is as high as 95%.

- AUROC is the Area Under the Receiver Operating Characteristic curve, which is also a threshold-independent metric.

Hyperparameters of the ODIN detector were selected using grid search. For temperature T I selecet among 1, 10, 50, 100, 200, 500, 750, 1000. For perturbation magnitude I choose 10 evenly spaced numbers starting from 0 to 0.05.

The model used here is the pretrained ResNet from PyTorch Hub.

### Results

<center>

| OOD-datasets |  Bow Metrics  |       |
|--------------|:-------------:|-------|
|              | Baseline/ODIN |             |
|              | FPR (95% TPR) | AUROC       |
| Gaussian     |0.40/**0.0**   |0.95/**0.99**|
| Uniform      |0.42/**0.0**   |0.95/**0.99**|
| CIFAR100     |0.69/**0.50**   |0.85/**0.87**|

</center>

We see that while we obtain basically a perfect classifier when dealing with gaussian and uniform noise the improvments are lower when using as ood dataset CIFAR100. This is explainable if we consider the fact that these datasets are really similar even if we consider only different classes.

| ROC Gaussian | ROC Uniform | ROC Cifar100      |
|--------------|:-------------:|-------|
|![](/laboratory_4/doc/ROCgaussian.png)|![](/laboratory_4/doc/ROCuniform.png)|![](/laboratory_4/doc/ROCcifar100.png.png)|

### Observation

In none of the paper is mationed the influence of the model used when evaluating this OOD detectors. Here are the results obtained in the same condition but using my custom CNN insted of the pretrained ResNet. Note that my Custom CNN achived a much worse accuracy with respect to the ResNet.

<center>

| OOD-datasets |  Bow Metrics  |       |
|--------------|:-------------:|-------|
|              | Baseline/ODIN |             |
|              | FPR (95% TPR) | AUROC       |
| Gaussian     |0.64/**0.17**   |0.87/**0.97**|
| Uniform      |0.87/**0.17**   |0.87/**0.97**|
| CIFAR100     |0.81/**0.71**   |0.76/**0.82**|

</center>

We conclude that the ability to separate the softmax score depends also on the performance of the network.



## FGSM targeted attack

The Fast Gradient Sign Method (FGSM) is a technique primarily used in the field of adversarial machine learning to generate adversarial examples. FGSM exploits the vulnerabilities in machine learning models by making small, purposeful modifications to input data to deceive the model into making incorrect predictions. This method calculates the gradient of the loss with respect to the input data, then adjusts the input in the direction that maximizes the loss, thus creating a subtly altered yet adversarially effective example.

Here I implemented both the targeted version of FGSM (where one specifies the class we want our model adversarial example to belong to) as well as the untargeted version. 

One can evaluate the result both qualitatively by simply looking at the adversarial samples generated or quantitatively.

![](/laboratory_4/doc/fsgm.png)

### run

To run and visualize an FGSM use the command

```python3 fgsm.py [arguments]```

- ```--epsilon```: epsilon value of FGSM attack (default: 0.01)
- ```--target```: target class if you want to performa a targeted attack (default: ```None```)
- ```--max_iterations```: max number of iteration to be performed by the attack if classifier still is not fooled (default: 100)

### Evaluation

To quantitatively evaluate the fsgm attack I considered the following metrics:

1. Attack Success Rate: Measure the percentage of adversarial samples that successfully fool the model. This is calculated by dividing the number of successful adversarial attacks by the total number of adversarial samples.

2. Perturbation Measurement: Quantify the magnitude of the changes made to the original samples. Common metrics include:

    - L2 Norm: Measures the Euclidean distance between the original and adversarial samples.
    - L∞ Norm: Measures the maximum change to any individual feature.

Metrics are reported as a function of epsilon in the following image:
As we expect attack success rate increases as a function of epsilon but so do the degradation measures.
![](/laboratory_4/doc/fsgm_eval.png)