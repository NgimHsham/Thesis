from torch.utils.data import DataLoader
from Data.datasets.isic2020_dataset import ISIC2020Dataset
from Data.transforms.isic2020_transforms import get_augmentation

def create_dataloaders(
    json_path,
    images_dir,
    batch_size=32,
    augmentation_source='torchvision',
    augmentation_level=5,
    num_workers=4,
    image_extension='.jpg',
    label_map={'benign': 0, 'malignant': 1}
):
    train_transform = get_augmentation(source=augmentation_source, level=augmentation_level, is_train=True)
    val_transform = get_augmentation(source=augmentation_source, level=augmentation_level, is_train=False)

    train_dataset = ISIC2020Dataset(
        json_path=json_path,
        split='training',
        images_dir=images_dir,
        transform=train_transform,
        label_map=label_map,
        image_extension=image_extension
    )

    val_dataset = ISIC2020Dataset(
        json_path=json_path,
        split='validation',
        images_dir=images_dir,
        transform=val_transform,
        label_map=label_map,
        image_extension=image_extension
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader
