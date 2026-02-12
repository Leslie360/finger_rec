import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from config import BATCH_SIZE, CROP_HEIGHT, NORM_MEAN, NORM_STD, RESIZE_SIZE

class CropBottom:
    """Crops the bottom N rows of the image."""
    def __init__(self, crop_height=CROP_HEIGHT):
        self.crop_height = crop_height

    def __call__(self, img):
        w, h = img.size
        return img.crop((0, 0, w, h - self.crop_height))
    
    def __repr__(self):
        return f"{self.__class__.__name__}(crop_height={self.crop_height})"

def get_dataloaders(root_dir='Dataset5'):
    # 修复：加入了 Resize，且 Normalize 适配 3 通道
    transform = transforms.Compose([
        CropBottom(CROP_HEIGHT),
        transforms.Resize(RESIZE_SIZE),  # [关键修复] 必须缩放
        transforms.ToTensor(),
        transforms.Normalize(mean=NORM_MEAN, std=NORM_STD)
    ])
    
    train_dir = os.path.join(root_dir, 'train')
    test_dir = os.path.join(root_dir, 'test')
    
    if not os.path.exists(train_dir):
        print(f"Warning: Train directory {train_dir} not found.")
    
    # ImageFolder 加载为 RGB (3通道)
    train_dataset = datasets.ImageFolder(root=train_dir, transform=transform)
    test_dataset = datasets.ImageFolder(root=test_dir, transform=transform)
    
    # 建议开启 pin_memory=True 和 num_workers
    train_loader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        num_workers=4, 
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False, 
        num_workers=4, 
        pin_memory=True
    )
    
    return train_loader, test_loader