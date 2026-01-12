"""
Entraînement principal.

Exécutable via :
    python -m src.train --config configs/config.yaml [--seed 42]

Respecte :
- runs/ et artifacts/
- TensorBoard scalars
- --overfit_small
"""

import argparse
import os
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from src.model import build_model
from src.data_loading import get_dataloaders
from src.utils import set_seed


# ----------------------------
# Train / Validate
# ----------------------------
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * x.size(0)
        correct += (out.argmax(1) == y).sum().item()
        total += y.size(0)

    return total_loss / total, correct / total


def validate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss = criterion(out, y)

            total_loss += loss.item() * x.size(0)
            correct += (out.argmax(1) == y).sum().item()
            total += y.size(0)

    return total_loss / total, correct / total


# ----------------------------
# ONE RUN (appelable par grid search)
# ----------------------------
def train_one_run(config, run_name, max_epochs=None, disable_checkpoint=False):
    # Seed
    if config["train"].get("seed") is not None:
        set_seed(config["train"]["seed"])

    # Device
    device = torch.device(
        "cuda" if config["train"]["device"] == "auto" and torch.cuda.is_available() else "cpu"
    )
    print(f"Using device: {device}")

    # Data
    train_loader, val_loader, _, _ = get_dataloaders(config)

    if config["train"].get("overfit_small", False):
        train_loader = torch.utils.data.DataLoader(
            torch.utils.data.Subset(train_loader.dataset, range(50)),
            batch_size=config["train"]["batch_size"],
            shuffle=True,
        )
        val_loader = torch.utils.data.DataLoader(
            torch.utils.data.Subset(val_loader.dataset, range(50)),
            batch_size=config["train"]["batch_size"],
            shuffle=False,
        )

    # Model
    model = build_model(config["model"]).to(device)
    criterion = nn.CrossEntropyLoss()

    # Optimizer
    opt = config["train"]["optimizer"]
    optimizer = optim.Adam(
        model.parameters(),
        lr=opt["lr"],
        weight_decay=opt["weight_decay"],
    )

    # TensorBoard
    runs_dir = config["paths"]["runs_dir"]
    os.makedirs(runs_dir, exist_ok=True)

    run_dir = os.path.join(runs_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)
    writer = SummaryWriter(run_dir)

    # Checkpoint (désactivable pour grid search)
    if not disable_checkpoint:
        artifacts_dir = config["paths"]["artifacts_dir"]
        os.makedirs(artifacts_dir, exist_ok=True)
        ckpt_path = os.path.join(artifacts_dir, "best.ckpt")

    # Training loop
    epochs = max_epochs or config["train"]["epochs"]
    best_val_acc = 0.0
    best_val_loss = float("inf")

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_acc = validate(
            model, val_loader, criterion, device
        )

        print(
            f"Epoch [{epoch}/{epochs}] "
            f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}"
        )

        writer.add_scalar("train/loss", train_loss, epoch)
        writer.add_scalar("train/accuracy", train_acc, epoch)
        writer.add_scalar("val/loss", val_loss, epoch)
        writer.add_scalar("val/accuracy", val_acc, epoch)

        if not disable_checkpoint and val_acc > best_val_acc:
            best_val_acc = val_acc
            best_val_loss = val_loss
            torch.save(model.state_dict(), ckpt_path)

    writer.close()

    return {
        "val_accuracy": best_val_acc,
        "val_loss": best_val_loss,
    }


# ----------------------------
# CLI
# ----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--overfit_small", action="store_true")
    parser.add_argument("--max_epochs", type=int, default=None)
    parser.add_argument("--max_steps", type=int, default=None)
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    if args.seed is not None:
        config["train"]["seed"] = args.seed
    if args.overfit_small:
        config["train"]["overfit_small"] = True
    if args.max_epochs is not None:
        config["train"]["epochs"] = args.max_epochs

    train_one_run(
        config=config,
        run_name="train",
        max_epochs=args.max_epochs,
    )


if __name__ == "__main__":
    main()
