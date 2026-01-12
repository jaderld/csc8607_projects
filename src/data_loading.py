import torch
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import CIFAR100
from torchvision import transforms
from src.preprocessing import get_preprocess_transforms
from src.augmentation import get_augmentation_transforms

def get_dataloaders(config: dict):
    # Prétraitement de base
    base_transform = get_preprocess_transforms(config)
    # Augmentation (ou None)
    aug_transform = get_augmentation_transforms(config)

    # Compose transformation train (preprocess + augmentation si définie)
    if aug_transform is not None:
        train_transform = transforms.Compose([base_transform, aug_transform])
    else:
        train_transform = base_transform

    # Datasets
    full_train_dataset = CIFAR100(root=config["dataset"]["root"], train=True,
                                  transform=train_transform, download=True)
    test_dataset = CIFAR100(root=config["dataset"]["root"], train=False,
                            transform=base_transform, download=True)

    # Split train/val (val n'a pas d'augmentation)
    val_size = 5000
    train_size = len(full_train_dataset) - val_size
    train_dataset, val_dataset = random_split(full_train_dataset, [train_size, val_size])
    # Remplacer la transform de val pour être sûr qu'elle n'a pas d'augmentation
    val_dataset.dataset.transform = base_transform

    # DataLoaders
    batch_size = config["train"]["batch_size"]
    num_workers = config["dataset"].get("num_workers", 4)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    # Metadata
    meta = {
        "num_classes": 100,
        "input_shape": (3,32,32)
    }

    return train_loader, val_loader, test_loader, meta