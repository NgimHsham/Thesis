import torchvision.transforms as T

def get_augmentation(source='torchvision', level=1, is_train=True):
    if source == 'torchvision':
        return TorchvisionAugmentation(level).get_transforms(is_train)
    else:
        raise ValueError('Only "torchvision" is supported now.')

class TorchvisionAugmentation:
    def __init__(self, level=1, image_size=224):
        self.level = level
        self.image_size = image_size

    def get_transforms(self, is_train=True):
        if not is_train:
            return T.Compose([
                T.Resize((self.image_size, self.image_size)),
                T.ToTensor(),
                T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])

        transforms = [T.Resize((self.image_size, self.image_size))]

        if self.level >= 1:
            transforms += [
                T.RandomHorizontalFlip(),
                T.RandomVerticalFlip(),
            ]

        if self.level >= 2:
            transforms += [
                T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                T.RandomRotation(degrees=20),
            ]

        if self.level >= 3:
            transforms += [
                T.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            ]

        if self.level >= 4:
            transforms += [
                T.RandomPerspective(distortion_scale=0.5, p=0.5),
            ]

        # FIRST: convert to Tensor
        transforms.append(T.ToTensor())

        # THEN: apply RandomErasing (only works on Tensor)
        if self.level >= 5:
            transforms.append(T.RandomErasing(p=0.5, scale=(0.02, 0.15)))

        # Finally normalize
        transforms.append(T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))

        return T.Compose(transforms)
