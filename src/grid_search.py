# src/grid_search.py
import argparse
import itertools
import os
import yaml
from torch.utils.tensorboard import SummaryWriter
from src.train import train_single_run  # fonction à créer dans train.py pour un run isolé
from src.utils import save_config_snapshot, set_seed

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # lecture config
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    set_seed(args.seed)
    hparams = config.get("hparams", {})

    # générer toutes les combinaisons possibles
    keys, values = zip(*hparams.items())
    combos = [dict(zip(keys, v)) for v in itertools.product(*values)]

    # créer dossier runs/grid_search
    gs_runs_dir = os.path.join(config["paths"]["runs_dir"], "grid_search")
    os.makedirs(gs_runs_dir, exist_ok=True)

    for i, combo in enumerate(combos):
        print(f"\n=== Grid search run {i+1}/{len(combos)} : {combo} ===")
        run_name = f"run_{i+1}"
        writer = SummaryWriter(log_dir=os.path.join(gs_runs_dir, run_name))

        # mettre à jour config avec cette combinaison
        run_config = config.copy()
        for k, v in combo.items():
            # mettre à jour la section train si l'hparam s'y trouve
            if k in run_config["train"]:
                run_config["train"][k] = v

        # sauvegarder config snapshot
        save_config_snapshot(run_config, os.path.join(gs_runs_dir, run_name))

        # lancer entraînement pour ce set d'hyperparams
        metrics = train_single_run(run_config, writer)  # doit retourner dict métriques finales

        # log hparams + metrics sur TensorBoard
        writer.add_hparams(combo, metrics)
        writer.close()

    print("\n=== Grid search terminé ===")

if __name__ == "__main__":
    main()
