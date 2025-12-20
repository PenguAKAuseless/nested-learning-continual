import random
import torch
from torch.utils.data import DataLoader, Dataset, ConcatDataset, Subset
import numpy as np


class StreamDataset(Dataset):
    """Wrapper to stream data continuously with optimized memory access"""
    def __init__(self, dataset, task_id=0):
        self.dataset = dataset
        self.task_id = task_id
        # Pre-compute length to avoid repeated calls
        self._len = len(dataset)
        
    def __len__(self):
        return self._len
    
    def __getitem__(self, idx):
        return self.dataset[idx]


class NoisyBoundaryStreamLoader:
    """
    Realistic continual learning data stream with:
    - Single-pass through data (no epoch repeating)
    - Blurry boundaries: future task data appears before task officially starts
    - Other task interference: past and unrelated task data mixed in (realistic scenario)
    - OOD (out-of-distribution) noise data
    - No mixing of old task training data (old tasks only evaluated on test set)
    """
    def __init__(self, current_task_dataset, future_task_datasets=None, 
                 other_task_datasets=None, ood_dataset=None, batch_size=32, 
                 blur_ratio=0.1, other_task_ratio=0.05, ood_ratio=0.05, 
                 shuffle=True, num_workers=0, task_id=0):
        """
        Parameters:
        - current_task_dataset: Dataset for the current task
        - future_task_datasets: List of datasets from future tasks (for blurry boundaries)
        - other_task_datasets: List of datasets from past/unrelated tasks (realistic interference)
        - ood_dataset: Out-of-distribution dataset (synthetic noise)
        - batch_size: int, batch size
        - blur_ratio: float, ratio of future task samples to inject (default 0.1 = 10%)
        - other_task_ratio: float, ratio of other task samples to inject (default 0.05 = 5%)
        - ood_ratio: float, ratio of OOD samples to inject (default 0.05 = 5%)
        - shuffle: bool, whether to shuffle
        - num_workers: int, number of workers
        - task_id: int, current task ID
        """
        self.current_task_dataset = current_task_dataset
        self.future_task_datasets = future_task_datasets if future_task_datasets else []
        self.other_task_datasets = other_task_datasets if other_task_datasets else []
        self.ood_dataset = ood_dataset
        self.batch_size = batch_size
        self.blur_ratio = blur_ratio
        self.other_task_ratio = other_task_ratio
        self.ood_ratio = ood_ratio
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.task_id = task_id
        
        # Build the complete stream (single pass)
        self.stream_data = self._build_stream()
        self.current_idx = 0
        
    def _build_stream(self):
        """Build the complete data stream with blurry boundaries, other task interference, and OOD noise"""
        # Get all current task data
        current_indices = list(range(len(self.current_task_dataset)))
        if self.shuffle:
            random.shuffle(current_indices)
        
        stream = []
        total_samples = len(current_indices)
        
        # Calculate how many samples to inject
        num_blur_samples = int(total_samples * self.blur_ratio)
        num_other_task_samples = int(total_samples * self.other_task_ratio)
        num_ood_samples = int(total_samples * self.ood_ratio)
        
        # Prepare future task samples for blurry boundaries
        blur_samples = []
        if self.future_task_datasets and num_blur_samples > 0:
            # Combine all future task datasets
            combined_future = ConcatDataset(self.future_task_datasets)
            future_len = len(combined_future)
            if future_len > 0:
                # Bound check: ensure we don't sample more than available
                num_blur_samples = min(num_blur_samples, future_len)
                future_indices = random.sample(range(future_len), num_blur_samples)
                blur_samples = [(combined_future[i], 'blur') for i in future_indices]
        
        # Prepare other task samples (past/unrelated tasks)
        other_task_samples = []
        if self.other_task_datasets and num_other_task_samples > 0:
            # Combine all other task datasets (past and unrelated)
            combined_other = ConcatDataset(self.other_task_datasets)
            other_len = len(combined_other)
            if other_len > 0:
                # Bound check: ensure we don't sample more than available
                num_other_task_samples = min(num_other_task_samples, other_len)
                other_indices = random.sample(range(other_len), num_other_task_samples)
                other_task_samples = [(combined_other[i], 'other_task') for i in other_indices]
        
        # Prepare OOD samples (synthetic noise)
        ood_samples = []
        if self.ood_dataset and num_ood_samples > 0:
            ood_len = len(self.ood_dataset)
            if ood_len > 0:
                # Bound check: ensure we don't sample more than available
                num_ood_samples = min(num_ood_samples, ood_len)
                ood_indices = random.sample(range(ood_len), num_ood_samples)
                ood_samples = [(self.ood_dataset[i], 'ood') for i in ood_indices]
        
        # Build stream with current task data
        for idx in current_indices:
            stream.append((self.current_task_dataset[idx], 'current'))
        
        # Inject blur samples at random positions (more at the end for realistic boundary blur)
        # Blurry boundary: more samples from next task appear toward the end
        if blur_samples:
            # Weight positions toward the end (last 30% of stream)
            blur_positions = []
            stream_len = len(stream)
            for _ in range(len(blur_samples)):
                if random.random() < 0.7:  # 70% chance to appear in last 30%
                    pos = random.randint(int(stream_len * 0.7), stream_len)
                else:  # 30% chance to appear anywhere
                    pos = random.randint(0, stream_len)
                blur_positions.append(pos)
            
            blur_positions.sort(reverse=True)
            for pos, (sample, label) in zip(blur_positions, blur_samples):
                # Bound check for insertion
                pos = min(pos, len(stream))
                stream.insert(pos, (sample, label))
        
        # Inject other task samples uniformly throughout the stream (realistic interference)
        if other_task_samples:
            stream_len = len(stream)
            other_positions = [random.randint(0, stream_len) for _ in range(len(other_task_samples))]
            other_positions.sort(reverse=True)
            for pos, (sample, label) in zip(other_positions, other_task_samples):
                # Bound check for insertion
                pos = min(pos, len(stream))
                stream.insert(pos, (sample, label))
        
        # Inject OOD samples at random positions throughout the stream
        if ood_samples:
            stream_len = len(stream)
            ood_positions = [random.randint(0, stream_len) for _ in range(len(ood_samples))]
            ood_positions.sort(reverse=True)
            for pos, (sample, label) in zip(ood_positions, ood_samples):
                # Bound check for insertion
                pos = min(pos, len(stream))
                stream.insert(pos, (sample, label))
        
        return stream
    
    def __iter__(self):
        self.current_idx = 0
        return self
    
    def __next__(self):
        """Get next batch from stream (single pass)"""
        if self.current_idx >= len(self.stream_data):
            raise StopIteration
        
        # Get batch
        batch_end = min(self.current_idx + self.batch_size, len(self.stream_data))
        batch_items = self.stream_data[self.current_idx:batch_end]
        
        self.current_idx = batch_end
        
        if len(batch_items) > 0:
            # Unpack data and metadata
            samples = [(item[0][0], item[0][1]) for item in batch_items]  # (image, label)
            metadata = [item[1] for item in batch_items]  # 'current', 'blur', 'ood'
            
            images = torch.stack([s[0] for s in samples])
            labels = torch.tensor([s[1] for s in samples])
            
            return images, labels, metadata
        else:
            raise StopIteration
    
    def __len__(self):
        """Return number of batches"""
        return (len(self.stream_data) + self.batch_size - 1) // self.batch_size


class ContinualStreamLoader:
    """
    Legacy data stream loader (kept for backward compatibility).
    For new implementations, use NoisyBoundaryStreamLoader.
    """
    def __init__(self, current_task_dataset, old_task_datasets=None, 
                 batch_size=32, current_mix_ratio=0.9, shuffle=True, num_workers=0):
        """
        Parameters:
        - current_task_dataset: Dataset for the current task
        - old_task_datasets: List of datasets from previous tasks (empty for task 0)
        - batch_size: int, batch size
        - current_mix_ratio: float, ratio of current task samples (default 0.9 for 90%)
        - shuffle: bool, whether to shuffle
        - num_workers: int, number of workers
        """
        self.current_task_dataset = current_task_dataset
        self.old_task_datasets = old_task_datasets if old_task_datasets else []
        self.batch_size = batch_size
        self.current_mix_ratio = current_mix_ratio
        self.shuffle = shuffle
        self.num_workers = num_workers
        
        # Calculate batch composition
        self.current_batch_size = int(batch_size * current_mix_ratio)
        self.old_batch_size = batch_size - self.current_batch_size
        
        # Create indices
        self.current_indices = list(range(len(current_task_dataset)))
        
        # Combine old task datasets if available
        if self.old_task_datasets:
            self.combined_old_dataset = ConcatDataset(self.old_task_datasets)
            self.old_indices = list(range(len(self.combined_old_dataset)))
        else:
            self.combined_old_dataset = None
            self.old_indices = []
        
        self.current_idx = 0
        self.old_idx = 0
        
    def get_loader(self):
        """Return a PyTorch DataLoader that yields mixed batches"""
        return MixedBatchLoader(
            self.current_task_dataset,
            self.combined_old_dataset,
            self.current_batch_size,
            self.old_batch_size,
            self.shuffle,
            self.num_workers
        )
    
    def __iter__(self):
        if self.shuffle:
            random.shuffle(self.current_indices)
            if self.old_indices:
                random.shuffle(self.old_indices)
        self.current_idx = 0
        self.old_idx = 0
        return self
    
    def __next__(self):
        """Get next mixed batch"""
        if self.current_idx >= len(self.current_task_dataset):
            raise StopIteration
        
        # Get current task samples
        current_end = min(self.current_idx + self.current_batch_size, 
                         len(self.current_task_dataset))
        current_batch_indices = self.current_indices[self.current_idx:current_end]
        current_batch = [self.current_task_dataset[i] for i in current_batch_indices]
        
        self.current_idx = current_end
        
        # Get old task samples if available
        if self.combined_old_dataset and self.old_batch_size > 0:
            # Wrap around old task indices if needed
            old_batch_indices = []
            for _ in range(self.old_batch_size):
                if self.old_idx >= len(self.old_indices):
                    if self.shuffle:
                        random.shuffle(self.old_indices)
                    self.old_idx = 0
                old_batch_indices.append(self.old_indices[self.old_idx])
                self.old_idx += 1
            
            old_batch = [self.combined_old_dataset[i] for i in old_batch_indices]
            combined_batch = current_batch + old_batch
            
            # Shuffle the combined batch
            if self.shuffle:
                random.shuffle(combined_batch)
        else:
            combined_batch = current_batch
        
        if len(combined_batch) > 0:
            images = torch.stack([item[0] for item in combined_batch])
            labels = torch.tensor([item[1] for item in combined_batch])
            return images, labels
        else:
            raise StopIteration
    
    def __len__(self):
        """Return number of batches"""
        return (len(self.current_task_dataset) + self.batch_size - 1) // self.batch_size
    
    def reset(self):
        """Reset the iterator"""
        self.current_idx = 0
        self.old_idx = 0
        if self.shuffle:
            random.shuffle(self.current_indices)
            if self.old_indices:
                random.shuffle(self.old_indices)


class MixedBatchLoader:
    """DataLoader-compatible class for mixed batches"""
    def __init__(self, current_dataset, old_dataset, current_batch_size, 
                 old_batch_size, shuffle, num_workers):
        self.current_dataset = current_dataset
        self.old_dataset = old_dataset
        self.current_batch_size = current_batch_size
        self.old_batch_size = old_batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        
    def __iter__(self):
        return ContinualStreamLoader(
            self.current_dataset,
            [self.old_dataset] if self.old_dataset else [],
            self.current_batch_size + self.old_batch_size,
            self.current_batch_size / (self.current_batch_size + self.old_batch_size),
            self.shuffle,
            self.num_workers
        ).__iter__()
    
    def __len__(self):
        return (len(self.current_dataset) + self.current_batch_size - 1) // self.current_batch_size


class StreamLoader:
    """Data stream loader for continual learning scenarios (legacy)"""
    def __init__(self, dataset, batch_size, shuffle=True, num_workers=0):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.current_index = 0
        self.indices = list(range(len(self.dataset)))

    def get_loader(self):
        """Return a PyTorch DataLoader"""
        return DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            pin_memory=True
        )
    
    def __iter__(self):
        if self.shuffle:
            random.shuffle(self.indices)
        self.current_index = 0
        return self

    def __next__(self):
        if self.current_index >= len(self.dataset):
            raise StopIteration
        
        batch_indices = self.indices[self.current_index:self.current_index + self.batch_size]
        batch_data = [self.dataset[i] for i in batch_indices]
        self.current_index += self.batch_size
        
        # Stack batch data
        if len(batch_data) > 0:
            images = torch.stack([item[0] for item in batch_data])
            labels = torch.tensor([item[1] for item in batch_data])
            return images, labels
        else:
            raise StopIteration

    def reset(self):
        self.current_index = 0
        if self.shuffle:
            random.shuffle(self.indices)