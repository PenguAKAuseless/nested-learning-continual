"""
Rivalry Strategies for Continual Learning

Implements various continual learning algorithms including:
- Elastic Weight Consolidation (EWC)
- Learning without Forgetting (LwF)
- Gradient Episodic Memory (GEM)
- PackNet
- Synaptic Intelligence (SI)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
from abc import ABC, abstractmethod
import copy


class RivalryStrategy(ABC):
    """Base class for continual learning strategies"""
    
    def __init__(self, model: nn.Module, device: str = 'cuda'):
        self.model = model
        self.device = device
        self.task_id = 0
        
    @abstractmethod
    def train_step(self, x: torch.Tensor, y: torch.Tensor, optimizer: torch.optim.Optimizer) -> float:
        """Perform one training step"""
        pass
    
    def train_step_amp(self, x: torch.Tensor, y: torch.Tensor, optimizer: torch.optim.Optimizer, 
                       scaler: torch.cuda.amp.GradScaler) -> float:
        """Perform one training step with automatic mixed precision"""
        self.model.train()
        optimizer.zero_grad()
        
        # Forward pass with autocast
        with torch.cuda.amp.autocast():
            output = self.model(x)
            loss = F.cross_entropy(output, y)
            
            # Add strategy-specific loss components
            loss = self._compute_loss_components(x, y, output, loss)
        
        # Backward pass with gradient scaling
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        return loss.item()
    
    def _compute_loss_components(self, x: torch.Tensor, y: torch.Tensor, 
                                 output: torch.Tensor, base_loss: torch.Tensor) -> torch.Tensor:
        """Override in subclasses to add strategy-specific loss terms (called within autocast)"""
        return base_loss
    
    @abstractmethod
    def after_task(self, dataloader: torch.utils.data.DataLoader):
        """Called after completing a task"""
        pass
    
    def before_task(self, task_id: int):
        """Called before starting a new task"""
        self.task_id = task_id


class EWCStrategy(RivalryStrategy):
    """Elastic Weight Consolidation (Kirkpatrick et al., 2017)"""
    
    def __init__(self, model: nn.Module, device: str = 'cuda', lambda_ewc: float = 1000.0, 
                 fisher_sample_size: int = 200):
        super().__init__(model, device)
        self.lambda_ewc = lambda_ewc
        self.fisher_sample_size = fisher_sample_size
        self.fisher_dict = {}
        self.optpar_dict = {}
        
    def train_step(self, x: torch.Tensor, y: torch.Tensor, optimizer: torch.optim.Optimizer) -> float:
        """Training step with EWC penalty"""
        self.model.train()
        optimizer.zero_grad()
        
        # Forward pass
        output = self.model(x)
        loss = F.cross_entropy(output, y)
        
        # Add EWC penalty
        if self.task_id > 0:
            ewc_loss = 0.0
            for name, param in self.model.named_parameters():
                if name in self.fisher_dict:
                    fisher = self.fisher_dict[name]
                    optpar = self.optpar_dict[name]
                    ewc_loss += (fisher * (param - optpar) ** 2).sum()
            
            loss += self.lambda_ewc * ewc_loss
        
        loss.backward()
        optimizer.step()
        
        return loss.item()
    
    def _compute_loss_components(self, x: torch.Tensor, y: torch.Tensor, 
                                 output: torch.Tensor, base_loss: torch.Tensor) -> torch.Tensor:
        """Add EWC penalty to base loss (called within autocast)"""
        loss = base_loss
        
        if self.task_id > 0:
            ewc_loss = 0.0
            for name, param in self.model.named_parameters():
                if name in self.fisher_dict:
                    fisher = self.fisher_dict[name]
                    optpar = self.optpar_dict[name]
                    ewc_loss += (fisher * (param - optpar) ** 2).sum()
            
            loss = loss + self.lambda_ewc * ewc_loss
        
        return loss
    
    def after_task(self, dataloader: torch.utils.data.DataLoader):
        """Compute Fisher information matrix"""
        self.model.eval()
        fisher_dict = {}
        
        # Initialize Fisher information accumulators
        for name, param in self.model.named_parameters():
            fisher_dict[name] = torch.zeros_like(param.data)
        
        # Sample from dataloader
        try:
            dataset_size = len(dataloader.dataset)  # type: ignore
        except (TypeError, AttributeError):
            dataset_size = self.fisher_sample_size
        num_samples = min(self.fisher_sample_size, dataset_size)
        sample_count = 0
        
        for x, y in dataloader:
            if sample_count >= num_samples:
                break
            
            x, y = x.to(self.device), y.to(self.device)
            
            self.model.zero_grad()
            output = self.model(x)
            loss = F.cross_entropy(output, y)
            loss.backward()
            
            # Accumulate squared gradients
            for name, param in self.model.named_parameters():
                if param.grad is not None:
                    fisher_dict[name] += param.grad.data ** 2
            
            sample_count += x.size(0)
        
        # Normalize Fisher information
        for name in fisher_dict:
            fisher_dict[name] /= num_samples
        
        # Update stored Fisher and optimal parameters
        for name, param in self.model.named_parameters():
            self.fisher_dict[name] = fisher_dict[name]
            self.optpar_dict[name] = param.data.clone()


class LwFStrategy(RivalryStrategy):
    """Learning without Forgetting (Li & Hoiem, 2017)"""
    
    def __init__(self, model: nn.Module, device: str = 'cuda', lambda_lwf: float = 1.0, 
                 temperature: float = 2.0):
        super().__init__(model, device)
        self.lambda_lwf = lambda_lwf
        self.temperature = temperature
        self.prev_model = None
        
    def train_step(self, x: torch.Tensor, y: torch.Tensor, optimizer: torch.optim.Optimizer) -> float:
        """Training step with knowledge distillation"""
        self.model.train()
        optimizer.zero_grad()
        
        # Forward pass
        output = self.model(x)
        loss = F.cross_entropy(output, y)
        
        # Add distillation loss if previous model exists
        if self.prev_model is not None:
            self.prev_model.eval()
            with torch.no_grad():
                prev_output = self.prev_model(x)
            
            # Knowledge distillation
            distill_loss = F.kl_div(
                F.log_softmax(output / self.temperature, dim=1),
                F.softmax(prev_output / self.temperature, dim=1),
                reduction='batchmean'
            ) * (self.temperature ** 2)
            
            loss += self.lambda_lwf * distill_loss
        
        loss.backward()
        optimizer.step()
        
        return loss.item()
    
    def after_task(self, dataloader: torch.utils.data.DataLoader):
        """Store copy of model for distillation"""
        self.prev_model = copy.deepcopy(self.model)
        self.prev_model.eval()


class GEMStrategy(RivalryStrategy):
    """Gradient Episodic Memory (Lopez-Paz & Ranzato, 2017)"""
    
    def __init__(self, model: nn.Module, device: str = 'cuda', memory_size: int = 256):
        super().__init__(model, device)
        self.memory_size = memory_size
        self.memory_data = []
        self.memory_labels = []
        
    def train_step(self, x: torch.Tensor, y: torch.Tensor, optimizer: torch.optim.Optimizer) -> float:
        """Training step with gradient projection"""
        self.model.train()
        
        # Compute gradient on current batch
        optimizer.zero_grad()
        output = self.model(x)
        loss = F.cross_entropy(output, y)
        loss.backward()
        
        # Store current gradients
        current_grads = []
        for param in self.model.parameters():
            if param.grad is not None:
                current_grads.append(param.grad.data.clone().flatten())
        
        if len(self.memory_data) > 0:
            # Compute gradients on memory
            memory_grads_list = []
            
            for mem_x, mem_y in zip(self.memory_data, self.memory_labels):
                self.model.zero_grad()
                mem_output = self.model(mem_x.unsqueeze(0).to(self.device))
                mem_loss = F.cross_entropy(mem_output, mem_y.unsqueeze(0).to(self.device))
                mem_loss.backward()
                
                mem_grads = []
                for param in self.model.parameters():
                    if param.grad is not None:
                        mem_grads.append(param.grad.data.clone().flatten())
                memory_grads_list.append(torch.cat(mem_grads))
            
            # Project gradient if it violates constraints
            current_grad = torch.cat(current_grads)
            
            for mem_grad in memory_grads_list:
                dot_product = torch.dot(current_grad, mem_grad)
                if dot_product < 0:
                    # Project current gradient
                    current_grad = current_grad - (dot_product / (mem_grad.norm() ** 2 + 1e-8)) * mem_grad
            
            # Replace gradients with projected gradients
            offset = 0
            for param in self.model.parameters():
                if param.grad is not None:
                    param_size = param.grad.numel()
                    param.grad.data = current_grad[offset:offset + param_size].view_as(param.grad.data)
                    offset += param_size
        
        optimizer.step()
        return loss.item()
    
    def after_task(self, dataloader: torch.utils.data.DataLoader):
        """Store exemplars in memory"""
        # Simple random sampling
        dataset = dataloader.dataset
        try:
            dataset_size = len(dataset)  # type: ignore
        except (TypeError, AttributeError):
            dataset_size = self.memory_size
        indices = torch.randperm(dataset_size)[:self.memory_size]
        
        for idx in indices:
            x, y = dataset[idx]
            self.memory_data.append(x)
            self.memory_labels.append(y)


class PackNetStrategy(RivalryStrategy):
    """PackNet (Mallya & Lazebnik, 2018)"""
    
    def __init__(self, model: nn.Module, device: str = 'cuda', prune_ratio: float = 0.5):
        super().__init__(model, device)
        self.prune_ratio = prune_ratio
        self.masks = {}
        
    def train_step(self, x: torch.Tensor, y: torch.Tensor, optimizer: torch.optim.Optimizer) -> float:
        """Training step with parameter masking"""
        self.model.train()
        optimizer.zero_grad()
        
        output = self.model(x)
        loss = F.cross_entropy(output, y)
        loss.backward()
        
        # Apply masks to gradients
        if self.task_id > 0:
            for name, param in self.model.named_parameters():
                if name in self.masks and param.grad is not None:
                    param.grad.data *= self.masks[name]
        
        optimizer.step()
        return loss.item()
    
    def after_task(self, dataloader: torch.utils.data.DataLoader):
        """Prune and freeze parameters"""
        for name, param in self.model.named_parameters():
            # Compute importance (magnitude-based)
            importance = param.data.abs()
            
            # Determine threshold for pruning
            threshold = torch.quantile(importance.flatten(), self.prune_ratio)
            
            # Create mask: 0 for pruned (frozen) weights
            mask = (importance >= threshold).float()
            
            if name in self.masks:
                # Combine with previous masks
                self.masks[name] *= mask
            else:
                self.masks[name] = mask


class SynapticIntelligence(RivalryStrategy):
    """Synaptic Intelligence (Zenke et al., 2017)"""
    
    def __init__(self, model: nn.Module, device: str = 'cuda', si_lambda: float = 1.0, 
                 xi: float = 0.1):
        super().__init__(model, device)
        self.si_lambda = si_lambda
        self.xi = xi
        self.omega = {}
        self.W = {}
        self.small_omega = {}
        
        # Initialize
        for name, param in self.model.named_parameters():
            self.omega[name] = torch.zeros_like(param.data)
            self.W[name] = param.data.clone()
            self.small_omega[name] = torch.zeros_like(param.data)
    
    def train_step(self, x: torch.Tensor, y: torch.Tensor, optimizer: torch.optim.Optimizer) -> float:
        """Training step with SI penalty"""
        self.model.train()
        optimizer.zero_grad()
        
        # Forward pass
        output = self.model(x)
        loss = F.cross_entropy(output, y)
        
        # Add SI penalty
        if self.task_id > 0:
            si_loss = 0.0
            for name, param in self.model.named_parameters():
                if name in self.omega:
                    si_loss += (self.omega[name] * (param - self.W[name]) ** 2).sum()
            
            loss += self.si_lambda * si_loss
        
        loss.backward()
        
        # Update small omega (importance accumulation)
        for name, param in self.model.named_parameters():
            if param.grad is not None and name in self.small_omega:
                self.small_omega[name] += param.grad.data * (param.data - self.W[name])
        
        optimizer.step()
        return loss.item()
    
    def after_task(self, dataloader: torch.utils.data.DataLoader):
        """Update importance weights"""
        for name, param in self.model.named_parameters():
            # Compute parameter importance
            delta = (param.data - self.W[name]) ** 2
            importance = self.small_omega[name] / (delta + self.xi)
            
            # Update omega (cumulative importance)
            self.omega[name] += importance.abs()
            
            # Reset for next task
            self.W[name] = param.data.clone()
            self.small_omega[name].zero_()


class NaiveStrategy(RivalryStrategy):
    """Naive fine-tuning baseline (no continual learning)"""
    
    def train_step(self, x: torch.Tensor, y: torch.Tensor, optimizer: torch.optim.Optimizer) -> float:
        """Standard training step with gradient clipping"""
        self.model.train()
        optimizer.zero_grad()
        
        output = self.model(x)
        loss = F.cross_entropy(output, y)
        loss.backward()
        
        # Clip gradients for subsequent tasks to reduce catastrophic forgetting
        if self.task_id > 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        return loss.item()
    
    def after_task(self, dataloader: torch.utils.data.DataLoader):
        """No additional processing"""
        pass
