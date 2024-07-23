import torch
import torch.nn as nn

import torch.nn.functional as F

class LinearWeightedMSELoss(nn.Module):
    def __init__(self, alpha=0.01, max_value=150):
        super(LinearWeightedMSELoss, self).__init__()
        self.alpha = alpha
        self.max_value = max_value

    def forward(self, pred, label):
        scaled_target = label / self.max_value
        weights = self.alpha * scaled_target
        return torch.mean(weights * (pred - label) ** 2)

class ScaledExponentialWeightedMSELoss(nn.Module):
    def __init__(self, alpha=0.01, max_value=150):
        super(ScaledExponentialWeightedMSELoss, self).__init__()
        self.alpha = alpha
        self.max_value = max_value

    def forward(self, pred, label):
        scaled_target = label / self.max_value
        weights = torch.exp(self.alpha * scaled_target)
        return torch.mean(weights * (pred - label) ** 2)

class LogarithmicWeightedMSELoss(nn.Module):
    def __init__(self, alpha=0.01, epsilon=1e-6, max_value=150):
        super(LogarithmicWeightedMSELoss, self).__init__()
        self.alpha = alpha
        self.epsilon = epsilon  # Small constant to avoid log(0)
        self.max_value = max_value

    def forward(self, pred, label):
        # Ensure target scaling is within a reasonable range
        scaled_target = label / self.max_value
        # Apply logarithmic weighting
        weights = torch.log(1 + self.alpha * scaled_target + self.epsilon)
        # Calculate the weighted MSE
        return torch.mean(weights * (pred - label) ** 2)

class LinearWeightedL1Loss(nn.Module):
    def __init__(self, alpha=0.01, max_value=150):
        super(LinearWeightedL1Loss, self).__init__()
        self.alpha = alpha
        self.max_value = max_value

    def forward(self, pred, label):
        scaled_target = label / self.max_value
        weights = self.alpha * scaled_target
        return torch.mean(weights * torch.abs(pred - label))

class LinearWeightedSmoothL1Loss(nn.Module):
    def __init__(self, beta=1.0, alpha=0.01, max_value=150):
        super(LinearWeightedSmoothL1Loss, self).__init__()
        self.beta = beta
        self.alpha = alpha
        self.max_value = max_value

    def forward(self, pred, label):
        scaled_target = label / self.max_value
        weights = self.alpha * scaled_target
        l1_loss = torch.abs(pred - label)
        condition = l1_loss < self.beta
        loss = torch.where(condition, 0.5 * l1_loss ** 2 / self.beta, l1_loss - 0.5 * self.beta)
        return torch.mean(weights * loss)

class LinearWeightedHuberLoss(nn.Module):
    def __init__(self, delta=1.0, alpha=0.01, max_value=150):
        super(LinearWeightedHuberLoss, self).__init__()
        self.delta = delta
        self.alpha = alpha
        self.max_value = max_value

    def forward(self, pred, label):
        scaled_target = label / self.max_value
        weights = self.alpha * scaled_target
        loss = F.huber_loss(pred, label, delta=self.delta, reduction='none')
        return torch.mean(weights * loss)

class NLLLoss(nn.Module):
    def __init__(self, class_frequencies=None):
        super(NLLLoss, self).__init__()
        if class_frequencies:
            self.loss_function = nn.NLLLoss(weight=1 / torch.tensor(class_frequencies))
        else:
            self.loss_function = nn.NLLLoss()

    def forward(self, pred, label):
        return self.loss_function(pred, label)