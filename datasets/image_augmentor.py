
from torchvision import transforms
import torch

whiteness = 0.98
def add_gaussian_noise(image, mean=0, std=0.02):
    noise = torch.randn(image.size()) * std + mean
    noisy_image = image + noise
    noisy_image = torch.clamp(noisy_image, 0, 1)
    return noisy_image

translation_aug = transforms.RandomAffine(0, translate=(0.15, 0.15),fill=255*whiteness)
rotation_aug = transforms.RandomRotation(360,fill=255*whiteness,interpolation=transforms.InterpolationMode.BILINEAR)
vertical_flip_aug = transforms.RandomVerticalFlip(p=0.5)
horizontal_flip_aug = transforms.RandomHorizontalFlip(p=0.5)
color_jitter_aug = transforms.ColorJitter(
                    brightness=0.15, hue=0.05, contrast=0.15, saturation=0.15
                )
sharpness_aug = transforms.RandomAdjustSharpness(
                    sharpness_factor=10, p=0.1
                )
gaussian_blur_aug = transforms.GaussianBlur(kernel_size=(1, 5), sigma=(0.01, 3))





IMAGE_AUGMENTATION_TRANSFORM_LIST = [
    color_jitter_aug,
    sharpness_aug,
    horizontal_flip_aug,
    vertical_flip_aug,
    rotation_aug,
    translation_aug,
    gaussian_blur_aug,
    transforms.ToTensor(),
    transforms.Lambda(lambda x: add_gaussian_noise(x)), # Add Gaussian noise
    transforms.ToPILImage()
]

def get_translation_aug():
    return 