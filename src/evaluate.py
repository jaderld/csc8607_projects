import yaml
import torch
import torch.nn as nn
from src.model import build_model
from src.data_loading import get_dataloaders

def evaluate(model, loader, device):
    model.eval()
    criterion = nn.CrossEntropyLoss()
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

    return running_loss/total, correct/total

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--ckpt", type=str, default=None)
    args = parser.parse_args()

    # Config
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() and config["train"]["device"] != "cpu" else "cpu")

    # DataLoaders
    train_loader, val_loader, test_loader, meta = get_dataloaders(config)

    # Model
    model = build_model(config["model"]).to(device)
    ckpt_path = args.ckpt or f"{config['paths']['artifacts_dir']}/best.ckpt"
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    print(f"Loaded checkpoint from {ckpt_path}")

    # Evaluate
    test_loss, test_acc = evaluate(model, test_loader, device)
    print(f"Test Loss: {test_loss:.4f} | Test Accuracy: {test_acc:.4f}")

if __name__ == "__main__":
    main()
