import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import numpy as np
import os

class ContinualImageNet:
    def __init__(self, root_dir, num_tasks=5, total_classes=256, batch_size=64, num_workers=4):
        self.root_dir = root_dir
        self.num_tasks = num_tasks
        self.total_classes = total_classes
        self.batch_size = batch_size
        self.num_workers = num_workers
        
        # Transform tiêu chuẩn cho ImageNet
        self.train_transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        self.val_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        self.full_train_dataset = datasets.ImageFolder(os.path.join(root_dir, 'train'), transform=self.train_transform)
        self.full_val_dataset = datasets.ImageFolder(os.path.join(root_dir, 'val'), transform=self.val_transform)
        
        # Tạo mapping giai đoạn (tasks)
        self.classes_per_task = self.total_classes // self.num_tasks
        self.class_indices = list(range(self.total_classes)) # Giả sử thư mục đã sort đúng
        
        self.train_targets = np.array(self.full_train_dataset.targets)
        self.val_targets = np.array(self.full_val_dataset.targets)

    def get_data_loader(self, task_id, cumulative=False):
        """
        Lấy DataLoader cho task_id.
        Args:
            task_id (int): Index của task hiện tại (0 đến num_tasks-1).
            cumulative (bool): Nếu True (cho Offline Learning), trả về dữ liệu của 
                               tất cả các task từ 0 đến task_id.
        """
        if cumulative:
            target_classes = self.class_indices[:(task_id + 1) * self.classes_per_task]
        else:
            start_class = task_id * self.classes_per_task
            end_class = (task_id + 1) * self.classes_per_task
            target_classes = self.class_indices[start_class:end_class]

        # Lọc indices cho Train
        train_indices = np.where(np.isin(self.train_targets, target_classes))[0]
        train_subset = Subset(self.full_train_dataset, train_indices)
        
        # Lọc indices cho Val (Luôn trả về cumulative val để test forgetting)
        # Tuy nhiên, trong training loop ta thường test trên task hiện tại hoặc all seen tasks.
        # Ở đây ta trả về tập val tương ứng với tập train (cumulative)
        val_indices = np.where(np.isin(self.val_targets, target_classes))[0]
        val_subset = Subset(self.full_val_dataset, val_indices)

        train_loader = DataLoader(train_subset, batch_size=self.batch_size, 
                                  shuffle=True, num_workers=self.num_workers, pin_memory=True)
        val_loader = DataLoader(val_subset, batch_size=self.batch_size, 
                                shuffle=False, num_workers=self.num_workers, pin_memory=True)
        
        print(f"Task {task_id} (Cumulative={cumulative}): Classes {target_classes[0]}-{target_classes[-1]}")
        print(f"Train samples: {len(train_subset)} | Val samples: {len(val_subset)}")
        
        return train_loader, val_loader