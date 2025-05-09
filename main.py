import os
import json
import time
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from Model_Training.Config.config_training import config
from Model_Training.Data.datasets.isic2020_dataset import ISIC2020Dataset
from Model_Training.Models.model import get_model
from Model_Training.Data.transforms.isic2020_transforms import get_augmentation
from Model_Training.Training.train import train

# Environment settings
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"  # Optional: set to "1" for debugging sync errors

# CUDA configuration (adjusted for stability)
torch.cuda.empty_cache()
os.environ['TORCH_HOME'] = '/work/c-2iia/hn977782/torch_cache'
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.enabled = True

print("cuDNN settings adjusted: benchmark=True, deterministic=False, allow_tf32=True, cudnn.enabled=True")

# Define device (use only the first available GPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Paths and model list
folds_dir = '/work/c-2iia/hn977782/Thesis/Code/Datasets/ISIC2020/Data_splitting/Random_patient_equal_split'
image_dir = '/work/c-2iia/hn977782/Thesis/Code/Datasets/ISIC2020/Images'
model_list = ["resnet50","densenet121"]

# Training loop
for model_name in model_list:
    print(f"\n\n########## Starting Training for {model_name.upper()} ##########\n")
    num_folds = 5

    for fold_idx in range(1, num_folds + 1):
        print(f"Training Fold {fold_idx} for model {model_name}...")

        fold_json_path = os.path.join(folds_dir, f"fold_{fold_idx}", f"fold_{fold_idx}_split.json")
        with open(fold_json_path, 'r') as f:
            fold_split = json.load(f)

        # Transforms
        train_transforms = get_augmentation(source='torchvision', level=5, is_train=True)
        val_transforms = get_augmentation(source='torchvision', level=5, is_train=False)

        # Datasets
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

        # Dataloaders (using standard DataLoader without DistributedSampler)
        batch_size = 16
        num_workers = 16

        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=True
        )

        val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=True
        )

        # Model
        model = get_model(model_name=model_name, num_classes=config['num_classes'])
        model = model.to(device)  # Move model to GPU

        # Train
        print(f"Starting training on Fold {fold_idx} for model {model_name}...")
        train(model, train_loader, val_loader, config, model_name, fold_idx)

        # Memory cleanup and pause after each fold
        del model
        del train_loader
        del val_loader
        del train_dataset
        del val_dataset
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        time.sleep(10)  # Optional: give GPU time to recover
