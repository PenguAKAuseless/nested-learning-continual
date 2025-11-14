import torch
import torch.nn as nn
from tqdm import tqdm
import copy
import gc
from data.stream_loader import NoisyBoundaryStreamLoader
from evaluation.metrics import ContinualLearningMetrics


class ContinualLearner:
    """
    Continual Learning with realistic data streaming and Nested Learning optimization:
    
    Realistic Streaming Features:
    - Single-pass through data (no epoch repeating)
    - Blurry boundaries (future task data appears early)
    - Other task interference (past task data mixed in)
    - OOD noise injection (filtered during training to avoid label -1 errors)
    - Old tasks only evaluated on test set (not mixed in training)
    
    Nested Learning Features:
    - Multi-frequency parameter updates (fast/medium/slow levels)
    - Different components update at different rates
    - Prevents catastrophic forgetting through frequency-based isolation
    - Fast layers adapt to new tasks, slow layers preserve old knowledge
    
    Note: OOD samples have label -1 and are filtered out during loss calculation
    to avoid CrossEntropyLoss errors. They add realism to the data stream but
    don't contribute to gradient updates.
    
    [cite: Google Nested Learning paper - 38, 213, 217, 291, 295]
    """
    def __init__(self, model, optimizer, criterion, device='cpu', memory_size=1000, 
                 log_optimizer_stats_freq=100):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        
        # GPU optimization: ensure device is torch.device
        self.device = torch.device(device) if isinstance(device, str) else device
        
        # Move model to device immediately
        self.model = self.model.to(self.device)
        
        self.current_task = 0
        self.memory_buffer = {}  # Changed to dict for easier indexing
        self.memory_size = memory_size
        self.task_history = []
        self.memory_count = 0
        self.log_optimizer_stats_freq = log_optimizer_stats_freq
        
        # Track datasets for each task (for test evaluation only)
        self.train_task_datasets = []  # Store train datasets for future task blurring
        self.test_task_datasets = []  # Store test datasets for evaluation
        
        # Initialize metrics tracker
        self.metrics = ContinualLearningMetrics()
        
        self.data_loader = None
        
        # Check if using Nested Optimizer
        self.is_nested_optimizer = hasattr(optimizer, 'get_update_stats')
        if self.is_nested_optimizer:
            print("✓ Using Nested Optimizer with multi-frequency updates")
        
        # GPU optimization flags
        self.use_cuda = self.device.type == 'cuda'
        if self.use_cuda:
            print(f"✓ GPU acceleration enabled: {torch.cuda.get_device_name(0)}")
            print(f"  Available GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
            # Enable cudnn autotuner for faster convolutions
            torch.backends.cudnn.benchmark = True

    def train_on_task(self, task_dataset, test_dataset=None, all_task_datasets=None,
                     ood_dataset=None, batch_size=32, task_name="Task", 
                     evaluate_old_tasks=True, blur_ratio=0.1, other_task_ratio=0.05, ood_ratio=0.05):
        """
        Train on a specific task with realistic data streaming:
        - Single pass through data (no epoch repeating)
        - Blurry boundaries (future task data appears before task starts)
        - Other task interference (past/unrelated task data mixed in)
        - OOD noise injection
        - Old tasks only evaluated on test set
        
        Parameters:
        - task_dataset: Dataset for current task
        - test_dataset: Test dataset for current task (optional)
        - all_task_datasets: List of ALL task datasets (for future task blurring)
        - ood_dataset: Out-of-distribution dataset for noise
        - batch_size: Batch size
        - task_name: Name for logging
        - evaluate_old_tasks: Whether to evaluate on old tasks after training
        - blur_ratio: Ratio of future task data to inject (blurry boundary)
        - other_task_ratio: Ratio of other task data to inject (realistic interference)
        - ood_ratio: Ratio of OOD data to inject
        """
        print(f"\n=== Training on {task_name} {self.current_task} ===")
        
        # Store datasets for future reference
        self.train_task_datasets.append(task_dataset)
        if test_dataset:
            self.test_task_datasets.append(test_dataset)
        
        # Update metrics tracker
        self.metrics.set_current_task(self.current_task)
        
        # Prepare future task datasets for blurry boundaries
        future_task_datasets = []
        if all_task_datasets and self.current_task < len(all_task_datasets) - 1:
            future_task_datasets = all_task_datasets[self.current_task + 1:]
            print(f"Blurry boundary: {blur_ratio*100:.0f}% future task data injection")
        
        # Prepare other task datasets (past tasks) for realistic interference
        other_task_datasets = []
        if self.current_task > 0 and self.train_task_datasets:
            # Include past tasks (already learned) as interference
            other_task_datasets = self.train_task_datasets[:-1]  # All except current
            print(f"Other task interference: {other_task_ratio*100:.0f}% past task data injection")
        
        if ood_dataset:
            print(f"OOD noise: {ood_ratio*100:.0f}% synthetic noise injection")
        
        # Create realistic data stream (SINGLE PASS - no epochs!)
        print(f"📊 Data stream: {len(task_dataset)} samples (single pass)")
        stream_loader = NoisyBoundaryStreamLoader(
            current_task_dataset=task_dataset,
            future_task_datasets=future_task_datasets,
            other_task_datasets=other_task_datasets,
            ood_dataset=ood_dataset,
            batch_size=batch_size,
            blur_ratio=blur_ratio,
            other_task_ratio=other_task_ratio,
            ood_ratio=ood_ratio,
            shuffle=True,
            task_id=self.current_task
        )
        
        # Train on the stream (SINGLE PASS)
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        blur_count = 0
        other_task_count = 0
        ood_count = 0
        batch_count = 0
        
        # GPU optimization: use automatic mixed precision for faster training
        use_amp = self.use_cuda and torch.cuda.is_available()
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
        
        pbar = tqdm(stream_loader, desc=f'{task_name} {self.current_task}')
        for batch_data in pbar:
            # Unpack batch (includes metadata about data source)
            images, labels, metadata = batch_data
            
            # GPU optimization: non_blocking transfer for overlap with computation
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)
            
            # Track data composition
            blur_count += sum(1 for m in metadata if m == 'blur')
            other_task_count += sum(1 for m in metadata if m == 'other_task')
            ood_count += sum(1 for m in metadata if m == 'ood')
            
            # Filter out OOD samples for training (they have label -1)
            # OOD samples are for realism but shouldn't contribute to loss
            valid_mask = labels >= 0
            valid_images = images[valid_mask]
            valid_labels = labels[valid_mask]
            
            # Add only valid samples to memory buffer
            if valid_mask.sum() > 0:
                # Move to CPU for memory buffer to save GPU memory
                self._update_memory(valid_images.cpu(), valid_labels.cpu())
            
            # Skip batch if all samples are OOD
            if valid_mask.sum() == 0:
                continue
            
            self.optimizer.zero_grad(set_to_none=True)  # More efficient than zero_grad()
            
            # Mixed precision training for speed
            with torch.cuda.amp.autocast(enabled=use_amp):
                outputs = self.model(valid_images)
                loss = self.criterion(outputs, valid_labels)
            
            # Scaled backward pass
            scaler.scale(loss).backward()
            scaler.step(self.optimizer)
            scaler.update()
            
            # Update model's global step counter if using Nested Learning
            if hasattr(self.model, 'increment_step'):
                self.model.increment_step()
            
            total_loss += loss.item()
            
            # Compute accuracy without creating new tensors
            with torch.no_grad():
                _, predicted = torch.max(outputs.data, 1)
                total += valid_labels.size(0)
                correct += (predicted == valid_labels).sum().item()
            
            batch_count += 1
            
            # Prepare progress bar info
            pbar_info = {
                'loss': f'{loss.item():.4f}',
                'acc': f'{100 * correct / total:.2f}%',
                'blur': blur_count,
                'other': other_task_count,
                'ood': ood_count
            }
            
            # Add optimizer stats if using Nested Optimizer
            if self.is_nested_optimizer and hasattr(self.optimizer, 'global_step'):
                pbar_info['step'] = self.optimizer.global_step
            
            pbar.set_postfix(pbar_info)
            
            # Log optimizer statistics periodically
            if (self.is_nested_optimizer and 
                hasattr(self.optimizer, 'should_log_stats') and
                self.optimizer.should_log_stats(self.log_optimizer_stats_freq)):
                self.optimizer.print_update_stats()
            
            # Periodically clear GPU cache to prevent memory buildup
            if self.use_cuda and batch_count % 50 == 0:
                torch.cuda.empty_cache()
        
        avg_loss = total_loss / len(stream_loader)
        final_accuracy = 100 * correct / total
        
        print(f'\n{task_name} {self.current_task} Results:')
        print(f'  Loss: {avg_loss:.4f}, Accuracy: {final_accuracy:.2f}%')
        print(f'  Data composition: {total + ood_count} total samples processed')
        print(f'    - Current task: {len(task_dataset)} samples (base)')
        print(f'    - Future task (blur): {blur_count} samples')
        print(f'    - Other tasks (interference): {other_task_count} samples')
        print(f'    - OOD noise (synthetic): {ood_count} samples')
        print(f'    - Valid for training: {total} samples')
        
        # Print final optimizer stats for this task
        if self.is_nested_optimizer and hasattr(self.optimizer, 'print_update_stats'):
            print(f"\nNested Optimizer Final Stats for {task_name} {self.current_task}:")
            self.optimizer.print_update_stats()
        
        # GPU memory cleanup after task
        if self.use_cuda:
            torch.cuda.empty_cache()
            gc.collect()
        
        # Update metrics
        self.metrics.update_task_accuracy(self.current_task, final_accuracy)
        
        # Evaluate on current task test set if available
        if test_dataset:
            test_acc = self.evaluate_task(test_dataset, self.current_task, batch_size)
            self.metrics.update_task_accuracy(self.current_task, test_acc)
        
        # Evaluate on old tasks (TEST SET ONLY)
        if evaluate_old_tasks and self.current_task > 0:
            self.evaluate_old_tasks(batch_size)
        
        # Print metrics summary
        self.metrics.print_summary()
        
        self.task_history.append({
            'task_id': self.current_task,
            'task_name': task_name,
            'accuracy': final_accuracy,
            'blur_samples': blur_count,
            'other_task_samples': other_task_count,
            'ood_samples': ood_count
        })

    def evaluate_task(self, test_dataset, task_id, batch_size=32):
        """Evaluate model on a specific task's test set"""
        print(f"\n=== Evaluating on Task {task_id} Test Set ===")
        self.model.eval()
        
        from torch.utils.data import DataLoader
        test_loader = DataLoader(
            test_dataset, 
            batch_size=batch_size, 
            shuffle=False,
            num_workers=2,  # Parallel data loading
            pin_memory=self.use_cuda,  # Faster GPU transfer
            persistent_workers=True if torch.utils.data.get_worker_info() is None else False
        )
        
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in tqdm(test_loader, desc=f'Evaluating Task {task_id}'):
                images = images.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)
                
                outputs = self.model(images)
                
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        accuracy = 100 * correct / total
        print(f'Task {task_id} Test Accuracy: {accuracy:.2f}%')
        return accuracy
    
    def evaluate_old_tasks(self, batch_size=32):
        """Evaluate model on all old tasks"""
        print(f"\n=== Evaluating Old Tasks ===")
        
        for old_task_id in range(self.current_task):
            if old_task_id < len(self.test_task_datasets) and self.test_task_datasets[old_task_id]:
                old_acc = self.evaluate_task(
                    self.test_task_datasets[old_task_id], 
                    old_task_id, 
                    batch_size
                )
                self.metrics.update_task_accuracy(old_task_id, old_acc)

    def _update_memory(self, images, labels):
        """Update memory buffer using reservoir sampling"""
        for img, label in zip(images, labels):
            if self.memory_count < self.memory_size:
                self.memory_buffer[self.memory_count] = (img.cpu(), label.cpu().item())
                self.memory_count += 1
            else:
                # Reservoir sampling
                idx = torch.randint(0, self.memory_count + 1, (1,)).item()
                if idx < self.memory_size:
                    self.memory_buffer[idx] = (img.cpu(), label.cpu().item())
                self.memory_count += 1
    
    def _sample_memory(self, batch_size):
        """Sample from memory buffer"""
        max_idx = min(self.memory_size, len(self.memory_buffer))
        indices = torch.randperm(max_idx)[:batch_size]
        return [self.memory_buffer[i.item()] for i in indices]

    def update_task(self, task_name="Task"):
        """Move to next task"""
        self.current_task += 1
        self.metrics.set_current_task(self.current_task)
        print(f"\n>>> Switching to {task_name} {self.current_task}")

    def evaluate(self, evaluation_loader, task_name="Test"):
        """Evaluate model on test set (legacy method)"""
        print(f"\n=== Evaluating on {task_name} ===")
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in tqdm(evaluation_loader, desc='Evaluating'):
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                total_loss += loss.item()
                
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        average_loss = total_loss / len(evaluation_loader)
        accuracy = 100 * correct / total
        print(f'{task_name} Loss: {average_loss:.4f}, Accuracy: {accuracy:.2f}%')
        return average_loss, accuracy
    
    def get_task_history(self):
        """Return training history across all tasks"""
        return self.task_history
    
    def get_metrics(self):
        """Return the metrics tracker"""
        return self.metrics