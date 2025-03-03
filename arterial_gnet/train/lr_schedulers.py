import math
from torch.optim.lr_scheduler import _LRScheduler

class PolyLRScheduler(_LRScheduler):
    def __init__(self, optimizer, initial_lr: float, max_steps: int, exponent: float = 0.9, current_step: int = None):
        self.optimizer = optimizer
        self.initial_lr = initial_lr
        self.max_steps = max_steps
        self.exponent = exponent
        self.ctr = 0
        
        # Set initial_lr in optimizer's param_groups
        for param_group in optimizer.param_groups:
            param_group['initial_lr'] = initial_lr
            
        super().__init__(optimizer, current_step if current_step is not None else -1, False)

    def step(self, current_step=None):
        if current_step is None or current_step == -1:
            current_step = self.ctr
            self.ctr += 1

        new_lr = self.initial_lr * (1 - current_step / self.max_steps) ** self.exponent
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = new_lr

class CosineDecayWithWarmupLRScheduler(_LRScheduler):
    def __init__(self, optimizer, max_lr: float, max_steps: int, 
                 warmup_steps: int = 0, min_lr: float = 0.0, current_step: int = 0):
        self.optimizer = optimizer
        self.max_lr = max_lr
        self.max_steps = max_steps
        self.warmup_steps = warmup_steps
        self.min_lr = min_lr
        self.ctr = current_step
        
        # Set initial_lr in optimizer's param_groups
        for param_group in optimizer.param_groups:
            param_group['initial_lr'] = max_lr if warmup_steps == 0 else max_lr / warmup_steps
            param_group['lr'] = max_lr if warmup_steps == 0 else max_lr / warmup_steps
            
        super().__init__(optimizer, current_step if current_step is not None else -1, False)

    def step(self, current_step=None):
        if current_step is None or current_step == -1:
            current_step = self.ctr
            self.ctr += 1

        if current_step < self.warmup_steps:
            new_lr = self.max_lr * (current_step + 1) / self.warmup_steps
        elif current_step > self.max_steps:
            new_lr = self.min_lr
        else:
            decay_ratio = (current_step - self.warmup_steps) / (self.max_steps - self.warmup_steps)
            assert 0 < decay_ratio < 1
            coeff = 0.5 * (1.0 - math.cos(math.pi * decay_ratio))
            new_lr = self.min_lr + coeff * (self.max_lr - self.min_lr)

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = new_lr
        