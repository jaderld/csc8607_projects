import argparse
import os
import yaml
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from src.model import build_model
from src.data_loading import get_dataloaders

# ----------------------------
# Utils
# ----------------------------
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def save_checkpoint(model, path):
    torch.save(model.state_dict(), path)

# ----------------------------
# Train / Validate
# ----------------------------
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        outputs = model(x)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * x.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == y).sum().item()
        total += y.size(0)
    return running_loss / total, correct / total

def validate(model, loader, criterion, device):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            outputs = model(x)
            loss = criterion(outputs, y)
            running_loss += loss.item() * x.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == y).sum().item()
            total += y.size(0)
    return running_loss / total, correct / total

# ----------------------------
# Main
# ----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--overfit_small", action="store_true")
    parser.add_argument("--max_epochs", type=int, default=None)
    args = parser.parse_args()

    # Lire config
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    if args.seed: set_seed(args.seed)
    elif config["train"].get("seed"): set_seed(config["train"]["seed"])

    device = torch.device("cuda" if torch.cuda.is_available() and config["train"]["device"] != "cpu" else "cpu")
    print(f"Using device: {device}")

    train_loader, val_loader, test_loader, meta = get_dataloaders(config)

    if args.overfit_small or config["train"].get("overfit_small", False):
        train_loader = torch.utils.data.DataLoader(
            torch.utils.data.Subset(train_loader.dataset, range(50)),
            batch_size=config["train"]["batch_size"], shuffle=True)
        val_loader = torch.utils.data.DataLoader(
            torch.utils.data.Subset(val_loader.dataset, range(50)),
            batch_size=config["train"]["batch_size"], shuffle=False)

    model = build_model(config["model"]).to(device)
    criterion = nn.CrossEntropyLoss()

    optimizer_cfg = config["train"]["optimizer"]
    if optimizer_cfg["name"].lower() == "adam":
        optimizer = optim.Adam(model.parameters(), lr=optimizer_cfg["lr"], weight_decay=optimizer_cfg["weight_decay"])
    elif optimizer_cfg["name"].lower() == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=optimizer_cfg["lr"],
                              momentum=optimizer_cfg.get("momentum",0.9),
                              weight_decay=optimizer_cfg["weight_decay"])
    else:
        raise ValueError(f"Optimizer {optimizer_cfg['name']} not supported")

    runs_dir = config["paths"]["runs_dir"]
    os.makedirs(runs_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=runs_dir)

    num_epochs = args.max_epochs or config["train"]["epochs"]
    best_val_acc = 0.0
    artifacts_dir = config["paths"]["artifacts_dir"]
    os.makedirs(artifacts_dir, exist_ok=True)
    best_ckpt_path = os.path.join(artifacts_dir, "best.ckpt")

    for epoch in range(1, num_epochs+1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        print(f"Epoch [{epoch}/{num_epochs}] Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")

        writer.add_scalar("train/loss", train_loss, epoch)
        writer.add_scalar("train/accuracy", train_acc, epoch)
        writer.add_scalar("val/loss", val_loss, epoch)
        writer.add_scalar("val/accuracy", val_acc, epoch)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_checkpoint(model, best_ckpt_path)

    writer.close()
    print(f"Best model saved at: {best_ckpt_path}")

if __name__ == "__main__":
    main()