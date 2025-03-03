import torch
import torch.nn as nn

import torch.nn.functional as F

class LinearWeightedMSELoss(nn.Module):
    def __init__(self, alpha=1.0, max_value=150):
        super(LinearWeightedMSELoss, self).__init__()
        self.alpha = alpha
        self.max_value = max_value

    def forward(self, pred, label):
        scaled_target = label / self.max_value
        weights = self.alpha * scaled_target
        return torch.mean(weights * (pred - label) ** 2)

class ScaledExponentialWeightedMSELoss(nn.Module):
    def __init__(self, alpha=1.0, max_value=150):
        super(ScaledExponentialWeightedMSELoss, self).__init__()
        self.alpha = alpha
        self.max_value = max_value

    def forward(self, pred, label):
        scaled_target = label / self.max_value
        weights = torch.exp(self.alpha * scaled_target)
        return torch.mean(weights * (pred - label) ** 2)

class LogarithmicWeightedMSELoss(nn.Module):
    def __init__(self, alpha=1.0, epsilon=1e-6, max_value=150):
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
    
class CombinedLoss(nn.Module):
    def __init__(self, weight=None, alpha=0.5, weighted_loss=None, scaling_factor=500):
        super(CombinedLoss, self).__init__()
        self.ce_loss = nn.CrossEntropyLoss(weight=weight)
        self.alpha = alpha
        self.weighted_loss = weighted_loss
        self.scaling_factor = scaling_factor
        
        if self.weighted_loss == "lin":
            self.mse_loss = LinearWeightedMSELoss(alpha=1.0)
        elif self.weighted_loss == "exp":
            self.mse_loss = ScaledExponentialWeightedMSELoss(alpha=1.0)
            self.scaling_factor = scaling_factor * 2 # Adjust scaling factor (empirically)
        elif self.weighted_loss == "log":
            self.mse_loss = LogarithmicWeightedMSELoss(alpha=1.0)
        else:
            self.mse_loss = nn.MSELoss()

    def forward(self, pred_class, pred_cont, label_class, label_cont):
        """
        Combined classification and regression loss.
        For negative class (label=0): Combines classification and time prediction losses
        For positive class (label=1): Only uses classification loss since time is undefined
        
        Args:
            pred_class: Classification predictions
            pred_cont: Time predictions 
            label_class: True class labels (0 for possible tasks, 1 for impossible tasks)
            label_cont: True continuous values (time predictions)
        """
        # Mask continuous predictions for positive class samples
        masked_pred = pred_cont * (1 - label_class)
        masked_label = label_cont * (1 - label_class)

        class_loss = self.alpha * self.ce_loss(pred_class, label_class)
        reg_loss = (1 / self.scaling_factor) * (1 - self.alpha) * self.mse_loss(masked_pred, masked_label)

        # print(f"Total loss: {class_loss + reg_loss:.4f} | Class loss: {class_loss:.4f}, Reg loss: {reg_loss:.4f} | Ratio: {reg_loss / class_loss:.2f}")
        
        return class_loss + reg_loss