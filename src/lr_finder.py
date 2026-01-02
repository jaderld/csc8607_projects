# src/lr_finder.py
import argparse
import os
import yaml
import torch
from torch.utils.tensorboard import SummaryWriter
from src.model import build_model
from src.data_loading import get_dataloaders
from src.utils import set_seed, get_device

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # lecture config
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    set_seed(args.seed)
    device = get_device(config["train"].get("device", "auto"))

    # loaders
    train_loader, val_loader, test_loader, meta = get_dataloaders(config)

    # modèle
    model = build_model(config["model"]).to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-7)  # lr très bas pour départ

    # tensorboard
    writer = SummaryWriter(log_dir=os.path.join(config["paths"]["runs_dir"], "lr_finder"))

    lrs = []
    losses = []

    # LR finder : augmenter lr exponentiellement
    lr = 1e-7
    max_lr = 1
    multiplier = (max_lr / lr) ** (1/len(train_loader))  # augmente à chaque batch

    model.train()
    for batch_idx, (x, y) in enumerate(train_loader):
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        outputs = model(x)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()

        # log
        lrs.append(lr)
        losses.append(loss.item())
        writer.add_scalar("lr_finder/loss", loss.item(), batch_idx)
        writer.add_scalar("lr_finder/lr", lr, batch_idx)

        # augmenter lr
        lr *= multiplier
        for g in optimizer.param_groups:
            g['lr'] = lr

        # stop si lr trop grande
        if lr > max_lr:
            break

    writer.close()
    print("LR Finder terminé. Visualisez TensorBoard pour choisir le LR optimal.")

if __name__ == "__main__":
    main()