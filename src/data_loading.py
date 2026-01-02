import torch
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import CIFAR100
from src.preprocessing import get_preprocess_transforms

def get_dataloaders(config: dict):
    transform = get_preprocess_transforms(config)

    train_dataset = CIFAR100(root=config["dataset"]["root"], train=True,
                             transform=transform, download=True)
    test_dataset = CIFAR100(root=config["dataset"]["root"], train=False,
                            transform=transform, download=True)

    # Split train/val
    val_size = 5000
    train_size = len(train_dataset) - val_size
    train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])

    batch_size = config["train"]["batch_size"]
    num_workers = config["dataset"].get("num_workers", 4)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    meta = {
        "num_classes": 100,
        "input_shape": (3,32,32)
    }

    return train_loader, val_loader, test_loader, meta