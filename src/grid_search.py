"""
Mini grid search.

Exécutable via :
    python -m src.grid_search --config configs/config.yaml
"""

import argparse
import itertools
import copy
import os
import yaml
from torch.utils.tensorboard import SummaryWriter

from src.train import train_one_run


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    with open(args.config, "r") as f:
        base_config = yaml.safe_load(f)

    # Récupération des listes d'hyperparamètres
    h = base_config["hparams"]
    lr_list = h["lr"]
    wd_list = h["weight_decay"]
    blocks_list = h["blocks"]           # Hyperparamètre Modèle A
    mid_list = h["bottleneck_mid"]      # Hyperparamètre Modèle B

    runs_root = os.path.join(base_config["paths"]["runs_dir"], "grid_search")
    os.makedirs(runs_root, exist_ok=True)

    run_id = 0

    # On itère sur LR, Weight Decay, Blocks et Bottleneck Mid
    # Le batch_size reste fixe (celui défini dans base_config['train'])
    for lr, wd, blocks, mid in itertools.product(lr_list, wd_list, blocks_list, mid_list):
        run_id += 1
        
        # Astuce : on convertit la liste blocks en string sans espaces pour le nom du fichier
        blocks_str = str(blocks).replace(" ", "")
        
        run_name = f"run{run_id}_lr={lr}_wd={wd}_blocks={blocks_str}_mid={mid}"
        print(f"\n▶ {run_name}")

        config = copy.deepcopy(base_config)
        
        # Injection des hyperparamètres d'entraînement
        config["train"]["optimizer"]["lr"] = lr
        config["train"]["optimizer"]["weight_decay"] = wd
        
        # Injection des hyperparamètres du modèle (A et B)
        config["model"]["blocks"] = blocks
        config["model"]["bottleneck_mid"] = mid

        # Lancement de l'entraînement
        # Note : on utilise config['grid']['epochs'] si dispo, sinon 5 par défaut pour aller vite
        grid_epochs = base_config.get("grid", {}).get("epochs", 5)

        metrics = train_one_run(
            config=config,
            run_name=f"grid_search/{run_name}",
            max_epochs=grid_epochs,
            disable_checkpoint=True,
        )

        # Log des HParams dans TensorBoard
        writer = SummaryWriter(os.path.join(runs_root, run_name))
        writer.add_hparams(
            {
                "lr": lr, 
                "weight_decay": wd, 
                "blocks": str(blocks),  # TensorBoard préfère les strings aux listes
                "bottleneck_mid": mid
            },
            metrics,
        )
        writer.close()

    print("\nGrid search terminée")


if __name__ == "__main__":
    main()