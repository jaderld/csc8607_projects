import torchvision.transforms as T

def get_preprocess_transforms(config: dict):
    """
    Retourne un objet torchvision.transforms.Compose
    combinant pré-traitement et augmentation selon la config.
    """
    transform_list = []

    # ----------------------------
    # Data augmentation
    # ----------------------------
    augment_cfg = config.get("augment", {})

    if augment_cfg.get("random_crop"):
        crop_cfg = augment_cfg["random_crop"]
        size = crop_cfg.get("size", 32)
        padding = crop_cfg.get("padding", 4)
        transform_list.append(T.RandomCrop(size=size, padding=padding))

    if augment_cfg.get("random_flip", False):
        transform_list.append(T.RandomHorizontalFlip())

    if augment_cfg.get("color_jitter"):
        cj = augment_cfg["color_jitter"]
        transform_list.append(T.ColorJitter(
            brightness=cj.get("brightness",0.0),
            contrast=cj.get("contrast",0.0),
            saturation=cj.get("saturation",0.0),
            hue=cj.get("hue",0.0)
        ))

    # ----------------------------
    # Conversion en Tensor
    # ----------------------------
    transform_list.append(T.ToTensor())

    # ----------------------------
    # Normalisation
    # ----------------------------
    preprocess_cfg = config.get("preprocess", {})
    if preprocess_cfg.get("normalize"):
        mean = preprocess_cfg["normalize"]["mean"]
        std = preprocess_cfg["normalize"]["std"]
        transform_list.append(T.Normalize(mean=mean, std=std))

    return T.Compose(transform_list)