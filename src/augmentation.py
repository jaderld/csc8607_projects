# src/augmentation.py
import torchvision.transforms as T

def get_augmentation_transforms(config: dict):
    """
    Retourne les transformations d'augmentation pour l'entraînement.
    Si aucune augmentation n'est configurée, retourne None.
    """
    augment_cfg = config.get("augment", {})
    transforms_list = []

    # Random horizontal flip
    if augment_cfg.get("random_flip", False):
        transforms_list.append(T.RandomHorizontalFlip())

    # Random crop
    crop_params = augment_cfg.get("random_crop", None)
    if crop_params:
        # crop_params attendu comme dict: {"size": 32, "padding": 4}
        transforms_list.append(T.RandomCrop(
            crop_params.get("size", 32),
            padding=crop_params.get("padding", 4)
        ))

    # Color jitter
    color_jitter_params = augment_cfg.get("color_jitter", None)
    if color_jitter_params:
        # color_jitter_params attendu comme dict: {"brightness": 0.2, "contrast":0.2, ...}
        transforms_list.append(T.ColorJitter(**color_jitter_params))

    if not transforms_list:
        return None

    # Compose all transforms
    return T.Compose(transforms_list)