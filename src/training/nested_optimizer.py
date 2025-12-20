"""
Nested Optimizer for Multi-Frequency Parameter Updates

This implements the Deep Optimizer concept from Nested Learning paper:
- Different parameter groups update at different frequencies
- Gradients accumulate until update frequency is met
- Enables both fast adaptation and slow integration

Key concepts from paper:
1. Nested optimization: Optimization itself is optimized [cite: 111, 213, 295]
2. Multi-frequency updates: theta_{i+1} = theta_i - eta * sum(grad) if i % freq == 0
3. Associative memory through gradient accumulation [cite: 59, 217]

[cite: 213, 217, 295, 304]
"""

import torch
from torch.optim.optimizer import Optimizer
from typing import List, Dict, Any, Optional, Callable
import math


class NestedOptimizer(Optimizer):
    """
    Nested Learning Optimizer with multi-frequency parameter updates.
    
    Unlike standard optimizers that update all parameters every step,
    this optimizer updates different parameter groups at different frequencies:
    - Level 0 (Fast): Update every 1 step
    - Level 1 (Medium): Update every 10 steps
    - Level 2 (Slow): Update every 100 steps
    
    This creates a continuum of memory operating at multiple time scales,
    crucial for continual learning without catastrophic forgetting.
    
    [cite: 38, 213, 217, 223, 291, 295]
    """
    
    def __init__(self, param_groups: List[Dict[str, Any]], lr: float = 1e-3,
                 betas: tuple = (0.9, 0.999), eps: float = 1e-8,
                 weight_decay: float = 0.0, amsgrad: bool = False):
        """
        Initialize Nested Optimizer.
        
        Args:
            param_groups: List of parameter groups with 'frequency' and 'level_id' keys
            lr: Learning rate (can be overridden per group)
            betas: Coefficients for computing running averages (Adam-style)
            eps: Term for numerical stability
            weight_decay: Weight decay (L2 penalty)
            amsgrad: Whether to use AMSGrad variant
        """
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if eps < 0.0:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        
        # Add default values to parameter groups
        for group in param_groups:
            if 'lr' not in group:
                group['lr'] = lr
            if 'frequency' not in group:
                raise ValueError("Each param_group must have 'frequency' key")
            if 'level_id' not in group:
                raise ValueError("Each param_group must have 'level_id' key")
            group.setdefault('betas', betas)
            group.setdefault('eps', eps)
            group.setdefault('weight_decay', weight_decay)
            group.setdefault('amsgrad', amsgrad)
            group.setdefault('layer_name', f"Level_{group['level_id']}")
        
        defaults = dict(lr=lr, betas=betas, eps=eps, 
                       weight_decay=weight_decay, amsgrad=amsgrad)
        super(NestedOptimizer, self).__init__(param_groups, defaults)
        
        # Global step counter for frequency-based updates
        self.global_step = 0
        
        # Statistics tracking
        self.update_stats = {
            'total_steps': 0,
            'updates_per_level': {},
            'skipped_per_level': {}
        }
    
    def step(self, closure: Optional[Callable] = None):  # type: ignore[override]
        """
        Perform a single optimization step with frequency-based updates.
        
        Key difference from standard optimizers:
        - Not all parameters update every step
        - Updates depend on global_step % frequency == 0
        - Gradients accumulate between updates
        
        [cite: 213, 217, 295]
        
        Args:
            closure: A closure that reevaluates the model and returns the loss
        
        Returns:
            Loss value if closure is provided, else None
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        
        # Increment global step
        self.global_step += 1
        self.update_stats['total_steps'] += 1
        
        # Process each parameter group
        for group_idx, group in enumerate(self.param_groups):
            frequency = group['frequency']
            level_id = group['level_id']
            layer_name = group.get('layer_name', f'Level_{level_id}')
            
            # Initialize stats tracking
            if level_id not in self.update_stats['updates_per_level']:
                self.update_stats['updates_per_level'][level_id] = 0
                self.update_stats['skipped_per_level'][level_id] = 0
            
            # Check if this group should update
            should_update = (self.global_step % frequency) == 0
            
            if not should_update:
                # Skip update, but accumulate gradients
                self.update_stats['skipped_per_level'][level_id] += 1
                self._accumulate_gradients(group)
                continue
            
            # Perform update
            self.update_stats['updates_per_level'][level_id] += 1
            self._update_parameters(group)
        
        return loss
    
    def _accumulate_gradients(self, group: Dict[str, Any]):
        """
        Accumulate gradients for later update.
        
        This implements the associative memory aspect of NL:
        gradients from multiple steps are integrated before parameter update.
        [cite: 59, 217]
        """
        for p in group['params']:
            if p.grad is None:
                continue
            
            state = self.state[p]
            
            # Initialize gradient accumulator
            if 'grad_acc' not in state:
                state['grad_acc'] = torch.zeros_like(p.grad)
            
            # Accumulate gradient
            state['grad_acc'].add_(p.grad)
    
    def _update_parameters(self, group: Dict[str, Any]):
        """
        Update parameters using accumulated gradients (Adam-style).
        
        Implements: theta_{i+1} = theta_i - eta * accumulated_gradients
        with Adam momentum and adaptive learning rates.
        
        [cite: 213, 295]
        """
        beta1, beta2 = group['betas']
        
        for p in group['params']:
            if p.grad is None:
                continue
            
            # Get gradient (use accumulated if available)
            state = self.state[p]
            if 'grad_acc' in state and state['grad_acc'] is not None:
                grad = state['grad_acc']
            else:
                grad = p.grad
            
            # Apply weight decay
            if group['weight_decay'] != 0:
                grad = grad.add(p, alpha=group['weight_decay'])
            
            # State initialization
            if len(state) == 0 or 'step' not in state:
                state['step'] = 0
                # Exponential moving average of gradient values
                state['exp_avg'] = torch.zeros_like(p)
                # Exponential moving average of squared gradient values
                state['exp_avg_sq'] = torch.zeros_like(p)
                if group['amsgrad']:
                    # Maintains max of exp_avg_sq for better convergence
                    state['max_exp_avg_sq'] = torch.zeros_like(p)
            
            exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
            state['step'] += 1
            
            # Decay the first and second moment running average coefficient
            exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
            exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
            
            if group['amsgrad']:
                max_exp_avg_sq = state['max_exp_avg_sq']
                # Maintains the maximum of all 2nd moment running avg. till now
                torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                # Use the max for normalizing running avg. of gradient
                denom = max_exp_avg_sq.sqrt().add_(group['eps'])
            else:
                denom = exp_avg_sq.sqrt().add_(group['eps'])
            
            # Bias correction
            bias_correction1 = 1 - beta1 ** state['step']
            bias_correction2 = 1 - beta2 ** state['step']
            step_size = group['lr'] * math.sqrt(bias_correction2) / bias_correction1
            
            # Update parameters
            p.addcdiv_(exp_avg, denom, value=-step_size)
            
            # Reset gradient accumulator
            if 'grad_acc' in state:
                state['grad_acc'].zero_()
    
    def zero_grad(self, set_to_none: bool = False):
        """
        Zero out gradients.
        
        Note: We don't zero accumulated gradients here - they persist
        until the next update for their frequency level.
        """
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is not None:
                    if set_to_none:
                        p.grad = None
                    else:
                        p.grad.zero_()
    
    def get_update_stats(self) -> Dict[str, Any]:
        """
        Get statistics about update frequencies.
        
        Useful for debugging and verifying that multi-frequency
        updates are working correctly.
        
        Returns:
            Dictionary with update statistics per level
        """
        stats = {
            'global_step': self.global_step,
            'total_steps': self.update_stats['total_steps'],
            'levels': {}
        }
        
        for level_id in self.update_stats['updates_per_level'].keys():
            updates = self.update_stats['updates_per_level'][level_id]
            skipped = self.update_stats['skipped_per_level'][level_id]
            total = updates + skipped
            
            # Find frequency for this level
            frequency = None
            layer_name = None
            for group in self.param_groups:
                if group['level_id'] == level_id:
                    frequency = group['frequency']
                    layer_name = group.get('layer_name', f'Level_{level_id}')
                    break
            
            stats['levels'][level_id] = {
                'layer_name': layer_name,
                'frequency': frequency,
                'updates': updates,
                'skipped': skipped,
                'total_opportunities': total,
                'update_rate': updates / total if total > 0 else 0,
                'expected_update_rate': 1.0 / frequency if frequency else 1.0
            }
        
        return stats
    
    def print_update_stats(self):
        """Print human-readable update statistics"""
        stats = self.get_update_stats()
        
        print(f"\n{'='*70}")
        print(f"Nested Optimizer Update Statistics")
        print(f"{'='*70}")
        print(f"Global Step: {stats['global_step']}")
        print(f"Total Steps: {stats['total_steps']}")
        print(f"{'-'*70}")
        
        for level_id in sorted(stats['levels'].keys()):
            level_stats = stats['levels'][level_id]
            print(f"\nLevel {level_id}: {level_stats['layer_name']}")
            print(f"  Frequency: {level_stats['frequency']}")
            print(f"  Updates: {level_stats['updates']}")
            print(f"  Skipped: {level_stats['skipped']}")
            print(f"  Update Rate: {level_stats['update_rate']:.4f} "
                  f"(Expected: {level_stats['expected_update_rate']:.4f})")
        
        print(f"{'='*70}\n")
    
    def should_log_stats(self, log_frequency: int = 100) -> bool:
        """Check if it's time to log statistics"""
        return (self.global_step % log_frequency) == 0
    
    def reset_stats(self):
        """Reset update statistics"""
        self.update_stats = {
            'total_steps': 0,
            'updates_per_level': {},
            'skipped_per_level': {}
        }


def create_nested_optimizer(model, lr: float = 1e-3, **kwargs) -> NestedOptimizer:
    """
    Convenience function to create NestedOptimizer from a model.
    
    The model must implement get_nested_param_groups() method that returns
    parameter groups with 'frequency' and 'level_id' metadata.
    
    Args:
        model: Model with get_nested_param_groups() method
        lr: Base learning rate
        **kwargs: Additional arguments for NestedOptimizer
    
    Returns:
        Configured NestedOptimizer instance
    
    Example:
        >>> model = NestedLearningNetwork()
        >>> optimizer = create_nested_optimizer(model, lr=1e-3)
    """
    if not hasattr(model, 'get_nested_param_groups'):
        raise ValueError(
            "Model must implement get_nested_param_groups() method. "
            "See NestedLearningNetwork for reference implementation."
        )
    
    param_groups = model.get_nested_param_groups()
    
    # Filter out empty parameter groups
    param_groups = [g for g in param_groups if g.get('params') and 
                    any(True for _ in g['params'])]
    
    if not param_groups:
        raise ValueError("No parameters found in model.get_nested_param_groups()")
    
    return NestedOptimizer(param_groups, lr=lr, **kwargs)
