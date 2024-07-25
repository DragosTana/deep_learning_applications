import torch
import torch.nn.functional as F
from sklearn import metrics
import gc

def input_preprocessing(inputs, epsilon, gradient):
    return inputs - epsilon * gradient.sign()

def compute_gradient(inputs, model, temperature):
    inputs.requires_grad = True
    logits = model(inputs)
    logits = logits / temperature
    max_logit, _ = torch.max(logits, dim=1)
    model.zero_grad()
    max_logit.backward(torch.ones(max_logit.shape, device=max_logit.device))
    gradient = -inputs.grad
    return gradient

def odin_detector(inputs, model, temperature, epsilon):
    gradient = compute_gradient(inputs, model, temperature)
    gradient = gradient.detach()
    preprocessed_inputs = input_preprocessing(inputs, epsilon, gradient)
    
    with torch.no_grad():
        logits = model(preprocessed_inputs)
        logits = logits / temperature
        softmax_scores = F.softmax(logits, dim=1)
        max_softmax_scores, _ = torch.max(softmax_scores, dim=1)
    
    return max_softmax_scores

def compute_odin_scores(data_loader, model, temperature, epsilon, device):
    """
    Compute the ODIN scores for a given data loader.
    Based on the paper "Enhancing The Reliability of Out-of-distribution Image Detection in Neural Networks" (https://arxiv.org/abs/1706.02690).
    """
    scores = []
    model.eval()
    for data in data_loader:
        x, _ = data
        x = x.to(device)
        s = odin_detector(x, model, temperature, epsilon)
        scores.append(s.cpu())
        del x, s
        torch.cuda.empty_cache()
        gc.collect()
    scores_t = torch.cat(scores)
    return scores_t

def ood_detector(data_loader, model, device):
    """
    Simple baseline for OOD detection.
    Based on the paper "A Baseline for Detecting Misclassified and Out-of-Distribution Examples in Neural Networks" (https://arxiv.org/abs/1610.02136).
    """
    scores = []
    with torch.no_grad():
        for data in data_loader:
            x, _ = data
            x = x.to(device)
            output = model(x)
            softmax_scores = F.softmax(output, dim=1)
            max_softmax_scores, _ = torch.max(softmax_scores, dim=1)
            scores.append(max_softmax_scores.cpu())

    scores_t = torch.cat(scores)
    return scores_t


def grid_search(temperatures, epsilons, testloader, fakeloader, target, model, device):
    """
    Perform a grid search to find the best temperature and epsilon for the ODIN detector.
    """
    best_auc = 0
    best_temp = None
    best_eps = None
    for temp in temperatures:
        for eps in epsilons:
            scores_test = compute_odin_scores(testloader, model, temp, eps, device)
            scores_fake = compute_odin_scores(fakeloader, model, temp, eps, device)
            prediction = torch.cat((scores_test, scores_fake))
            fpr, tpr, _ = metrics.roc_curve(target.cpu().numpy(), prediction.cpu().numpy())
            auc = metrics.auc(fpr, tpr)
            if auc > best_auc:
                print(f'Temperature: {temp}, Epsilon: {eps}, AUC: {auc}')
                best_auc = auc
                best_temp = temp
                best_eps = eps
    return best_temp, best_eps, best_auc