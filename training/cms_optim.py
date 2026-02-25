from turtle import update

import torch
from torch.optim import Optimizer

class CMSOptimizerWrapper:
    """
    Wrapper for PyTorch optimizers to implement Multi-Time Scale Updates for Nested Learning.
    It masks gradients for lower-frequency parameters based on the global step.
    """
    def __init__(self, optimizer: Optimizer, model: torch.nn.Module, k_factor: int = 5):
        """
        Args:
            optimizer: The base optimizer
            model: The model containing parameters with level information.
            k_factor: The frequency multiplier base
        """
        self.optimizer = optimizer
        self.model = model
        self.k_factor = k_factor
        self.global_step = 0
        self.param_levels = self._map_params_to_levels()

    def _map_params_to_levels(self):
        """
        Maps each parameter to its corresponding Nested Learning level based on naming convention.
        Expected naming: 'level_X' in the parameter name implies level X.
        Default level is 0 (updates every step).
        """
        level_map = {}
        for name, param in self.model.named_parameters():
            level = 0
            
            # Parsing level from name
            parts = name.split('.')
            for part in parts:
                if part.startswith('level_'):
                    try:
                        level = int(part.split('_')[1])
                    except ValueError:
                        pass
            
            level_map[param] = level
        return level_map

    def step(self):
        """
        Performs a single optimization step with frequency masking.
        """
        self.global_step += 1
        hidden_grads = {}
        
        # 1. Mask gradients before the optimizer step
        for group in self.optimizer.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                
                level = self.param_levels.get(p, 0)
                update_freq = self.k_factor ** level
                
                if self.global_step % update_freq != 0:
                    hidden_grads[p] = p.grad
                    p.grad = None
                else:
                    if update_freq > 1:
                        p.grad.data.div_(update_freq)

        # 2. Standard optimizer step
        self.optimizer.step()
        
        # 3. restore gradient for slow class
        for p, grad in hidden_grads.items():
            p.grad = grad
        

    def zero_grad(self, set_to_none: bool = False):
        for group in self.optimizer.param_groups:
            for p in group['params']:
                if p.grad is not None:
                    level = self.param_levels.get(p, 0)
                    update_freq = self.k_factor ** level
                    
                    if self.global_step == 0 or self.global_step % update_freq == 0:
                        if set_to_none:
                            p.grad = None
                        else:
                            p.grad.detach_()
                            p.grad.zero_()

    def state_dict(self):
        return self.optimizer.state_dict()

    def load_state_dict(self, state_dict):
        self.optimizer.load_state_dict(state_dict)