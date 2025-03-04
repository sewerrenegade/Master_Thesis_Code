import random
from torchvision import transforms
import numpy as np
import torch
from PIL import Image
from enum import Enum
import torchvision.transforms as transforms


class Augmentability(Enum):
    COMPLETELY_ROTATIONALY_INVARIANT = 1
    WEAKLY_ROTATIONALY_INVARIANT = 2
    UNAUGMENTABLE = 3
    
    def to_string(self) -> str:
        """Convert the enum to its name as a string."""
        return self.name

    @staticmethod
    def from_string(name: str) -> 'Augmentability':
        """Convert a string name back to the enum."""
        return Augmentability[name]
    
DATASET_AUGMENTABIBILITY = {
    "FashionMNIST": Augmentability.WEAKLY_ROTATIONALY_INVARIANT,
    "MNIST": Augmentability.WEAKLY_ROTATIONALY_INVARIANT,
    "CIFAR10": Augmentability.WEAKLY_ROTATIONALY_INVARIANT,
    "SCEMILA/fnl34_feature_data": Augmentability.UNAUGMENTABLE,
    "SCEMILA/image_data": Augmentability.COMPLETELY_ROTATIONALY_INVARIANT,
    
    "Acevedo": Augmentability.COMPLETELY_ROTATIONALY_INVARIANT,
    "MIL_FashionMNIST": Augmentability.WEAKLY_ROTATIONALY_INVARIANT,
    "MIL_MNIST": Augmentability.WEAKLY_ROTATIONALY_INVARIANT,
    "MIL_CIFAR10": Augmentability.WEAKLY_ROTATIONALY_INVARIANT,
    "MIL_SCEMILA": Augmentability.COMPLETELY_ROTATIONALY_INVARIANT,
    "MIL_Acevedo": Augmentability.COMPLETELY_ROTATIONALY_INVARIANT
}
DATASET_RGB = {
    "FashionMNIST": False,
    "MNIST": False,
    "CIFAR10": True,
    "SCEMILA/fnl34_feature_data": False,
    "SCEMILA/image_data": True,
    "Acevedo": True,
    "MIL_SCEMILA": True,
}
#make sure I added all included the augmentation settings for all datasets
#assert DATASET_AUGMENTABIBILITY.keys() == BASE_MODULES.keys()

class AugmentationSettings:
    ValidAugmentationNames = [
            'color_jitter', 'sharpness_aug', 'horizontal_flip_aug', 
            'vertical_flip_aug', 'rotation_aug', 'translation_aug', 
            'gaussian_blur_aug', 'gaussian_noise_aug', 'all','none'
        ]
    def __init__(
        self, 
        dataset_name: str ="", 
        color_jitter: bool = True, 
        sharpness_aug: bool = True, 
        horizontal_flip_aug: bool = True, 
        vertical_flip_aug: bool = True, 
        rotation_aug: bool = True, 
        translation_aug: bool = True, 
        gaussian_blur_aug: bool = True, 
        gaussian_noise_aug: bool = True,
        auto_generated_notes: str = "", # this input is used only for deserialization process
        probability_of_augmenting: float = 1.0
        
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
        self.auto_generated_notes = auto_generated_notes
        self.probability_of_augmenting = probability_of_augmenting
    def __eq__(self, other):
            if not isinstance(other, AugmentationSettings):
                return False
            return (
                self.dataset_name == other.dataset_name and
                self.color_jitter == other.color_jitter and
                self.sharpness_aug == other.sharpness_aug and
                self.horizontal_flip_aug == other.horizontal_flip_aug and
                self.vertical_flip_aug == other.vertical_flip_aug and
                self.rotation_aug == other.rotation_aug and
                self.translation_aug == other.translation_aug and
                self.gaussian_blur_aug == other.gaussian_blur_aug and
                self.gaussian_noise_aug == other.gaussian_noise_aug and
                self.auto_generated_notes == other.auto_generated_notes and
                self.probability_of_augmenting == other.probability_of_augmenting
            )


    def __repr__(self) -> str:
        """Provide a string representation of the configuration settings."""
        attrs = ', '.join(f"{key}={value}" for key, value in vars(self).items())
        return f"AugmentationSettings({attrs})"

    def to_dict(self) -> dict:
        """Convert the configuration settings to a dictionary."""
        return vars(self)
    
    @classmethod
    def from_dict(cls, data: dict):
        """Create an object from a dictionary."""
        return cls(**data)
    
    @classmethod
    def get_instance_from_unknown_struct(cls,data):
        if isinstance(data,str):
            return AugmentationSettings.create_settings_with_name(augmentation_name=data)
        if isinstance(data,list):
            return AugmentationSettings.create_settings_with_name(augmentation_name=data[0],p=data[1])
        elif isinstance(data,AugmentationSettings):
            return data
        elif isinstance(data,dict):
            return AugmentationSettings.from_dict(data)
        elif data is None:
            return AugmentationSettings.create_settings_with_name("none")
        else:
            raise TypeError("unsupoorted tpye to create augmentation settings settings")
    def is_no_augmentations_setting(self):  
    #If the input is all, then all possible augmentations will be activated``
        no_augmentations = True
        for aug in AugmentationSettings.ValidAugmentationNames[0:-2]:
            no_augmentations = no_augmentations and not getattr(self,aug)
        return no_augmentations


    @classmethod
    def create_settings_with_name(cls, augmentation_name: str, dataset_name: str="",p = 1.0) -> 'AugmentationSettings':
        """
        Initialize with all augmentations set to False except one specified.

        Parameters:
        - dataset_name (str): The name of the dataset. set later
        - only_enabled (str): The only augmentation to be enabled.

        Returns:
        - AugmentationSettings instance with only one augmentation enabled.
        """
        if augmentation_name is None:
            augmentation_name = "none"
        valid_augmentations = AugmentationSettings.ValidAugmentationNames
        if augmentation_name not in valid_augmentations:
            raise ValueError(f"{augmentation_name} is not a valid augmentation type.")
        if augmentation_name == valid_augmentations[-2]:
            x =  AugmentationSettings() #ALL AUGMENTATIONS ACTIVE
        elif augmentation_name == valid_augmentations[-1]:
            settings = {aug: False for aug in valid_augmentations[0:-2]}
            x= cls(dataset_name, **settings)
        else:   
            settings = {aug: False for aug in valid_augmentations[0:-2]}
            settings[augmentation_name] = True
            x = cls(dataset_name, **settings)
        x.probability_of_augmenting = p
        return x

whiteness = 0.98
class PerAugmentationBinomialAugmentor:
    def __init__(self, augmentation_list,per_augmentation_p = 0.1):
        self.per_augmentation_p = per_augmentation_p
        self.augmentation_list = augmentation_list
        self.augmentation_function = []
        for single_augmentation in self.augmentation_list:
            self.augmentation_function.append(BinomialAugmentor(single_augmentation,p=self.per_augmentation_p))
        self.augmentation_function = transforms.Compose(self.augmentation_function)
    def __call__(self,image):
        return self.augmentation_function(image)
    
    def __repr__(self):
        return f"{self.__class__.__name__}(p={self.per_augmentation_p})"
    
class BinomialAugmentor:
    def __init__(self, augmentation_function,p=0.5):
        self.p = p
        self.augmentation_function = augmentation_function
        assert callable(self.augmentation_function)
    def __call__(self, image):
        if random.random() < self.p:
            return self.augmentation_function(image)
        else:
            return image

    def __repr__(self):
        return f"{self.__class__.__name__}(p={self.p})"

class AddGaussianNoise:
    def __init__(self, mean=0.0, std=0.2):
        self.mean = mean
        self.std = std

    def __call__(self, image):
        if isinstance(image, Image.Image):
            # Convert PIL image to NumPy array
            image_np = np.array(image)
        elif isinstance(image, np.ndarray):
            image_np = image
        elif torch.is_tensor(image):
            # Convert Torch tensor to NumPy array
            image_np = image.numpy()
        else:
            raise TypeError("Input should be a PIL Image, NumPy array, or Torch tensor")
        
        # Add Gaussian noise
        noise = np.random.normal(self.mean, self.std, image_np.shape).astype(np.float32)
        image_noisy = image_np + noise
        
        # Clip the values to be in the valid range for image data
        image_noisy = np.clip(image_noisy, 0, 255)
        
        # Convert back to the original type
        if isinstance(image, Image.Image):
            return Image.fromarray(image_noisy.astype(np.uint8))
        elif isinstance(image, np.ndarray):
            return image_noisy
        elif torch.is_tensor(image):
            return torch.from_numpy(image_noisy)
        else:
            raise TypeError("Unexpected type encountered after processing")

    def __repr__(self):
        return f"{self.__class__.__name__}(mean={self.mean}, std={self.std})"


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


def get_dataset_compatible_augmentation_function(aug_settings: AugmentationSettings):
    from datasets.dataset_transforms import IdentityTransform
    dataset_name = aug_settings.dataset_name
    augmentations = []
    if DATASET_AUGMENTABIBILITY[dataset_name] is not Augmentability.UNAUGMENTABLE:
        if aug_settings.color_jitter and DATASET_RGB[dataset_name]:
            augmentations.append(color_jitter_aug)
        else:
            aug_settings.color_jitter = False
        if aug_settings.sharpness_aug:
            augmentations.append(sharpness_aug)
        if aug_settings.horizontal_flip_aug and DATASET_AUGMENTABIBILITY[dataset_name] == Augmentability.COMPLETELY_ROTATIONALY_INVARIANT:
            augmentations.append(horizontal_flip_aug)
        else:
            aug_settings.horizontal_flip_aug = False
        if aug_settings.vertical_flip_aug and DATASET_AUGMENTABIBILITY[dataset_name] == Augmentability.COMPLETELY_ROTATIONALY_INVARIANT:
            augmentations.append(vertical_flip_aug)
        else:
            aug_settings.vertical_flip_aug = False
        if aug_settings.rotation_aug:
            if DATASET_AUGMENTABIBILITY[dataset_name] is Augmentability.COMPLETELY_ROTATIONALY_INVARIANT:
                augmentations.append(complete_rotation_aug)
                aug_settings.auto_generated_notes += "+/-180 degree rotation augmentation applied\n"

            if DATASET_AUGMENTABIBILITY[dataset_name] is Augmentability.WEAKLY_ROTATIONALY_INVARIANT:
                augmentations.append(limited_rotation_aug)
                aug_settings.auto_generated_notes += "+/-15 degree rotation augmentation applied\n"
        if aug_settings.translation_aug:
            augmentations.append(translation_aug)
        if aug_settings.gaussian_blur_aug:
            augmentations.append(gaussian_blur_aug)
        if aug_settings.gaussian_noise_aug:
            augmentations.append(AddGaussianNoise())
    if len(augmentations) == 0 or DATASET_AUGMENTABIBILITY[dataset_name] == Augmentability.UNAUGMENTABLE:
        augmentations.append(IdentityTransform())

    if aug_settings.probability_of_augmenting is not None and 0 < aug_settings.probability_of_augmenting <= 1.0:
        augmentations = [PerAugmentationBinomialAugmentor(augmentation_list=augmentations,per_augmentation_p = aug_settings.probability_of_augmenting)]
    else:
        print(f"WARNING: No augmentations are applied")
        augmentations = [IdentityTransform()]
    return augmentations,aug_settings
            
            
            