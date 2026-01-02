# src/utils.py
import os
import random
import torch
import numpy as np
import yaml

def set_seed(seed: int) -> None:
    """Initialise les seeds pour numpy, random, torch pour reproductibilité."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_device(prefer: str | None = "auto") -> str:
    """Retourne 'cuda' si disponible et demandé, sinon 'cpu'."""
    if prefer == "cpu":
        return "cpu"
    if prefer == "cuda":
        return "cuda" if torch.cuda.is_available() else "cpu"
    # auto
    return "cuda" if torch.cuda.is_available() else "cpu"

def count_parameters(model) -> int:
    """Retourne le nombre de paramètres entraînables dans le modèle."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def save_config_snapshot(config: dict, out_dir: str) -> None:
    """Sauvegarde une copie de la config YAML dans out_dir."""
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "config_snapshot.yaml")
    with open(out_path, "w") as f:
        yaml.dump(config, f)
