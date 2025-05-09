def train_vgg(model_name, fold_idx, fold_json_path, image_dir, device, config):
    from Model_Training.Data.datasets.isic2020_dataset import ISIC2020Dataset
    from Model_Training.Data.transforms.isic2020_transforms import TorchvisionAugmentation
    from Model_Training.Models.model import get_model
    from torch.utils.data import DataLoader, DistributedSampler
    import torch.nn as nn
    import torch
    import os

    # Custom safe config for VGG
    batch_size = 2
    num_classes = config['num_classes']
    num_epochs = config['num_epochs']

    # Transforms
    train_transforms = TorchvisionAugmentation(level=5, image_size=160).get_transforms(is_train=True)
    val_transforms = TorchvisionAugmentation(level=5, image_size=160).get_transforms(is_train=False)

    # Dataset
    train_dataset = ISIC2020Dataset(
        json_path=fold_json_path,
        split='training',
        images_dir=image_dir,
        transform=train_transforms,
        label_map={'benign': 0, 'malignant': 1}
    )

    val_dataset = ISIC2020Dataset(
        json_path=fold_json_path,
        split='validation',
        images_dir=image_dir,
        transform=val_transforms,
        label_map={'benign': 0, 'malignant': 1}
    )

    train_sampler = DistributedSampler(train_dataset)
    val_sampler = DistributedSampler(val_dataset, shuffle=False)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, sampler=train_sampler, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, sampler=val_sampler, num_workers=4)

    # Model
    model = get_model(model_name=model_name, num_classes=num_classes)
    model = model.to(device)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[device.index])

    from Model_Training.Training.train import train as base_train
    print(f"🔥 Training {model_name.upper()} for Fold {fold_idx}")
    base_train(model, train_loader, val_loader, config, model_name, fold_idx)
