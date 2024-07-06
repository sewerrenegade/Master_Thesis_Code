from torchvision import transforms
import torch
from enum import Enum
from datasets.dataset_factory import BASE_MODULES
#from datasets.base_dataset_abstraction import BaseDataset
from typing import Union,Type

class Augmentability(Enum):
    COMPLETELY_ROTATIONALY_INVARIANT = 1
    WEAKLY_ROTATIONALY_INVARIANT = 2
    UNAUGMENTABLE = 3
    
DATASET_AUGMENTABIBILITY = {
    "FashionMNIST": Augmentability.WEAKLY_ROTATIONALY_INVARIANT,
    "MNIST": Augmentability.WEAKLY_ROTATIONALY_INVARIANT,
    "CIFAR10": Augmentability.WEAKLY_ROTATIONALY_INVARIANT,
    "SCEMILA/fnl34_feature_data": Augmentability.UNAUGMENTABLE,
    "SCEMILA/image_data": Augmentability.COMPLETELY_ROTATIONALY_INVARIANT,
    "SCEMILA/dinobloom_feature_data": Augmentability.COMPLETELY_ROTATIONALY_INVARIANT
}
DATASET_RGB = {
    "FashionMNIST": False,
    "MNIST": False,
    "CIFAR10": True,
    "SCEMILA/fnl34_feature_data": False,
    "SCEMILA/image_data": True,
    "SCEMILA/dinobloom_feature_data": True
}
#make sure I added all included the augmentation settings for all datasets
assert DATASET_AUGMENTABIBILITY.keys() == BASE_MODULES.keys()

class AugmentationSettings:
    def __init__(
        self, 
        dataset_name: str, 
        color_jitter: bool = True, 
        sharpness_aug: bool = True, 
        horizontal_flip_aug: bool = True, 
        vertical_flip_aug: bool = True, 
        rotation_aug: bool = True, 
        translation_aug: bool = True, 
        gaussian_blur_aug: bool = True, 
        gaussian_noise_aug: bool = True
    ) -> None:
        """
        Initialize the augmentation settings for the dataset.

        Parameters:
        - dataset_name (str): The name of the dataset.
        - color_jitter (bool): Whether to apply color jitter augmentation. Default is True.
        - sharpness_aug (bool): Whether to apply sharpness augmentation. Default is True.
        - horizontal_flip_aug (bool): Whether to apply horizontal flip augmentation. Default is True.
        - vertical_flip_aug (bool): Whether to apply vertical flip augmentation. Default is True.
        - rotation_aug (bool): Whether to apply rotation augmentation. Default is True.
        - translation_aug (bool): Whether to apply translation augmentation. Default is True.
        - gaussian_blur_aug (bool): Whether to apply Gaussian blur augmentation. Default is True.
        - gaussian_noise_aug (bool): Whether to apply Gaussian noise augmentation. Default is True.
        """
        self.dataset_name = dataset_name
        self.color_jitter = color_jitter
        self.sharpness_aug = sharpness_aug
        self.horizontal_flip_aug = horizontal_flip_aug
        self.vertical_flip_aug = vertical_flip_aug
        self.rotation_aug = rotation_aug
        self.translation_aug = translation_aug
        self.gaussian_blur_aug = gaussian_blur_aug
        self.gaussian_noise_aug = gaussian_noise_aug

        self._validate_settings()

    def _validate_settings(self) -> None:
        """Validate the initialization parameters."""
        if not isinstance(self.dataset_name, str):
            raise ValueError("dataset_name must be a string.")
        for attr, value in vars(self).items():
            if attr != 'dataset_name' and not isinstance(value, bool):
                raise ValueError(f"{attr} must be a boolean.")

    def __repr__(self) -> str:
        """Provide a string representation of the configuration settings."""
        attrs = ', '.join(f"{key}={value}" for key, value in vars(self).items())
        return f"AugmentationSettings({attrs})"

    def to_dict(self) -> dict:
        """Convert the configuration settings to a dictionary."""
        return vars(self)
    
    @classmethod
    def all_false_except_one(cls, dataset_name: str, only_enabled: str) -> 'AugmentationSettings':
        """
        Initialize with all augmentations set to False except one specified.

        Parameters:
        - dataset_name (str): The name of the dataset.
        - only_enabled (str): The only augmentation to be enabled.

        Returns:
        - AugmentationSettings instance with only one augmentation enabled.
        """
        valid_augmentations = [
            'color_jitter', 'sharpness_aug', 'horizontal_flip_aug', 
            'vertical_flip_aug', 'rotation_aug', 'translation_aug', 
            'gaussian_blur_aug', 'gaussian_noise_aug'
        ]
        if only_enabled not in valid_augmentations:
            raise ValueError(f"{only_enabled} is not a valid augmentation type.")
        
        settings = {aug: False for aug in valid_augmentations}
        settings[only_enabled] = True
        return cls(dataset_name, **settings)


whiteness = 0.98


def _add_gaussian_noise(image, mean=0, std=0.02):
    noise = torch.randn(image.size()) * std + mean
    noisy_image = image + noise
    noisy_image = torch.clamp(noisy_image, 0, 1)
    return noisy_image

def gaussian_noise_transform(mean=0, std=0.02):
    return transforms.Lambda(lambda x: _add_gaussian_noise(x,mean=mean,std=std))

def identity_transform():
    return transforms.Lambda(lambda x: x)


translation_aug = transforms.RandomAffine(
    0, translate=(0.15, 0.15), fill=255 * whiteness
)
complete_rotation_aug = transforms.RandomRotation(
    180, fill=255 * whiteness, interpolation=transforms.InterpolationMode.BILINEAR
)
limited_rotation_aug = transforms.RandomRotation(
    15, fill=255 * whiteness, interpolation=transforms.InterpolationMode.BILINEAR
)
vertical_flip_aug = transforms.RandomVerticalFlip(p=0.5)
horizontal_flip_aug = transforms.RandomHorizontalFlip(p=0.5)
color_jitter_aug = transforms.ColorJitter(
    brightness=0.15, hue=0.05, contrast=0.15, saturation=0.15
)
sharpness_aug = transforms.RandomAdjustSharpness(sharpness_factor=10, p=0.1)
gaussian_blur_aug = transforms.GaussianBlur(kernel_size=(1, 5), sigma=(0.01, 3))


# IMAGE_AUGMENTATION_TRANSFORM_LIST = [
#     color_jitter_aug,
#     sharpness_aug,
#     horizontal_flip_aug,
#     vertical_flip_aug,
#     rotation_aug,
#     translation_aug,
#     gaussian_blur_aug,
#     transforms.ToTensor(),
#     transforms.Lambda(lambda x: add_gaussian_noise(x)),  # Add Gaussian noise
#     transforms.ToPILImage(),
# ]


def get_augmentation_function(aug_settings: AugmentationSettings):
    dataset_name = aug_settings.dataset_name
    augmentations = []
    if DATASET_AUGMENTABIBILITY[dataset_name] is not Augmentability.UNAUGMENTABLE:
        if aug_settings.color_jitter and DATASET_RGB[dataset_name]:
            augmentations.append(color_jitter_aug)
        if aug_settings.sharpness_aug:
            augmentations.append(sharpness_aug)
        if aug_settings.horizontal_flip_aug and DATASET_AUGMENTABIBILITY[dataset_name] == Augmentability.COMPLETELY_ROTATIONALY_INVARIANT:
            augmentations.append(horizontal_flip_aug)
        if aug_settings.vertical_flip_aug and DATASET_AUGMENTABIBILITY[dataset_name] == Augmentability.COMPLETELY_ROTATIONALY_INVARIANT:
            augmentations.append(vertical_flip_aug)
        if aug_settings.rotation_aug:
            if DATASET_AUGMENTABIBILITY[dataset_name] is Augmentability.COMPLETELY_ROTATIONALY_INVARIANT:
                augmentations.append(complete_rotation_aug)
            if DATASET_AUGMENTABIBILITY[dataset_name] is Augmentability.WEAKLY_ROTATIONALY_INVARIANT:
                augmentations.append(limited_rotation_aug)
        if aug_settings.translation_aug:
            augmentations.append(translation_aug)
        if aug_settings.gaussian_blur_aug:
            augmentations.append(gaussian_blur_aug)
        if aug_settings.gaussian_noise_aug:
            augmentations.append(gaussian_noise_transform)
    if len(augmentations) == 0 or DATASET_AUGMENTABIBILITY[dataset_name] == Augmentability.UNAUGMENTABLE:
        augmentations.append(identity_transform)
    return transforms.Compose(augmentations)
            
            
            