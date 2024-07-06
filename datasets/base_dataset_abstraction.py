from torch.utils.data import Dataset
from configs.global_config import GlobalConfig
from abc import ABC, abstractmethod
import torch
import torchvision.transforms as transforms
from PIL import Image
import tifffile as tiff
import numpy as np
from datasets.image_augmentor import get_augmentation_function

class BaseDataset(Dataset,ABC):
    def __init__(self, database_name):
        self.name = database_name
        self.classes = None
    @abstractmethod
    def get_n_random_instances_from_class(self, class_name, number_of_instances):
        pass

    def get_n_random_instances_from_every_class(self, number_of_instances):
        instances_dict = {}
        for class_name in self.classes:
            instances_dict[class_name]  = self.get_n_random_instances_from_class(class_name,number_of_instances)[0]
        return instances_dict
    
    def get_transform_function(self,load_tiff = False,augmentation_settings = None,numpy=False,to_gpu = False,flatten = False,grayscale =False,to_tensor = True,extra_transforms = []):
        preload_transforms_list = []
        postload_transforms_list = []
        assert not (numpy and to_gpu)
        assert not (not numpy and flatten)
        assert not (numpy and to_gpu)
        #assert (not augment_image) or (augment_image and load_tiff)
        if load_tiff:
            preload_transforms_list.append(TifToPILimage())
        if augmentation_settings is not None:
            preload_transforms_list.extend(get_augmentation_function(augmentation_settings))
            self.augmentaion_settings = augmentation_settings
        if to_tensor:
            preload_transforms_list.append(transforms.ToTensor())
        if to_gpu:
            preload_transforms_list.append(ToGPUTransform())
        if grayscale:
            preload_transforms_list.append(transforms.Grayscale())
        if type(extra_transforms) is list:
            preload_transforms_list.extend(extra_transforms)
        if numpy:
            postload_transforms_list.append(ToNPTransform())
        if flatten:
            postload_transforms_list.append(FlattenTransform())
        
        return transforms.Compose(preload_transforms_list),transforms.Compose(postload_transforms_list)
    
class ToGPUTransform:
    def __init__(self) -> None:
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def __call__(self, pic):
        """
        Args:
            pic (PIL Image or numpy.ndarray): Image to be converted to tensor.

        Returns:
            Tensor: Converted image.
        """
        return pic.to(self.device)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"

class ToNPTransform:

    def __call__(self, pic):
        if isinstance(pic,torch.Tensor):
            return pic.numpy()
        elif  type(pic) == list or  isinstance(pic,Image.Image):
            return np.array(pic)
        elif isinstance(pic,np.ndarray):
            return pic
        else:
            print("unsupported to np format")
            raise Exception

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"

class FlattenTransform:
    def __call__(self, img):
        return img.reshape(-1)
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"
    
class TifToPILimage:
    def __init__(self,down_sample=False) -> None:
        self.down_sample = down_sample
    def __call__(self, image_path):
        image = tiff.imread(image_path)
        image = Image.fromarray(image)
        if self.down_sample:
            new_size = (image.width // 2, image.height // 2)
            image = image.resize(new_size,Image.LANCZOS)
        return image
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"