import numpy as np
import torch
import torchvision.transforms as transforms
from datasets.image_augmentor import AugmentationSettings
from PIL import Image
import tifffile as tiff
from models.DinoBloom.dinobloom_hematology_feature_extractor import get_dino_bloom,DINOBLOOM_TRANSFORMS,DINOBLOOM_NETWORKS_INFOS,DEFAULT_PATCH_NUM,DINOBLOOM_DEFAULT_IMAGE_DIM

class DatasetTransforms:
    def __init__(self, dataset_name,resize = False,load_tiff=False,load_jpg = False, augmentation_settings :AugmentationSettings=None, numpy=False, to_gpu=False, flatten=False, grayscale=False, to_tensor=False, extra_transforms=None) -> None:
        self.dataset_name = dataset_name
        self.load_tiff = load_tiff
        self.load_jpg = load_jpg
        self.augmentation_settings = augmentation_settings
        self.numpy = numpy
        self.to_gpu = to_gpu
        self.flatten = flatten
        self.grayscale = grayscale
        self.resize = resize
        self.to_tensor = to_tensor
        self.extra_transforms = extra_transforms if extra_transforms is not None else []
        self.preload_transforms,self.postload_transforms = None,None

    def create_preload_and_postload_transforms(self):
        if self.preload_transforms is None:
            from datasets.image_augmentor import get_dataset_compatible_augmentation_function
            preload_transforms_list = []
            postload_transforms_list = []
            assert not (self.numpy and self.to_gpu)
            assert not (not self.numpy and self.flatten)
            assert not (self.numpy and self.to_gpu)
            assert not (self.load_jpg and self.load_tiff)
            #assert (not augment_image) or (augment_image and load_tiff)
            if self.load_tiff:
                preload_transforms_list.append(TifToPILimage())
            if self.load_jpg:
                preload_transforms_list.append(JPGToPILimage())

            if self.grayscale:
                preload_transforms_list.append(transforms.Grayscale())
            if self.resize:
                preload_transforms_list.append(transforms.Resize((100,100),transforms.InterpolationMode.BILINEAR))

            if self.augmentation_settings is not None:
                aug_list, db_compatible_settings = get_dataset_compatible_augmentation_function(self.augmentation_settings)
                preload_transforms_list.extend(aug_list)
                self.augmentation_settings = db_compatible_settings                
            if self.to_tensor:
                preload_transforms_list.append(transforms.ToTensor())
            if type(self.extra_transforms) is list:
                preload_transforms_list.extend(self.extra_transforms)
            if self.to_gpu:
                preload_transforms_list.append(ToGPUTransform())
            if self.numpy:
                postload_transforms_list.append(ToNPTransform())
            if self.flatten:
                postload_transforms_list.append(FlattenTransform())
                
            if len(preload_transforms_list) == 0:
                preload_transforms_list.append(IdentityTransform())
            if len(postload_transforms_list) == 0:
                postload_transforms_list.append(IdentityTransform())
            return transforms.Compose(preload_transforms_list),transforms.Compose(postload_transforms_list)
        else:
            return self.preload_transforms,self.postload_transforms

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
    
#TODO add settings foe dinobloom encoder transforms for configurabilty (model sizes, input dim, normalisation scheme/parameters)
class DinoBloomEncodeTransform:
    def __init__(self,output_input = False) -> None:
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.default_mean = (0.485, 0.456, 0.406)
        self.default_std = (0.229, 0.224, 0.225)
        self.default_image_dim_input=224
        self.output_input = output_input
        self.dinobloom_pretransforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=self.default_mean, std=self.default_std),
        ])
        self.to_pil = transforms.ToPILImage()
        self.dino_bloom_encoder = get_dino_bloom("small")
    def __call__(self, pic):
        image = self.preprocess_image_for_dinobloom(pic)
        feautres_dict = self.dino_bloom_encoder(image)
        image_feaures = feautres_dict["x_norm_clstoken"]
        if self.output_input:
            return image_feaures,pic
        else:
            return image_feaures
    
    def preprocess_image_for_dinobloom(self,image):
        if not isinstance(image, Image.Image):
            # Convert NumPy array to PIL Image
            if isinstance(image, np.ndarray):
                image = Image.fromarray(image)
            # Convert PyTorch tensor to PIL Image
            elif isinstance(image, torch.Tensor):
                # Check if tensor is on GPU, if so move it to CPU
                if image.is_cuda:
                    image = image.cpu()
                image = self.to_pil(image)
            else:
                raise TypeError("Input must be a PIL Image, a NumPy array, or a PyTorch tensor.")

        img_tensor = self.dinobloom_pretransforms(image.convert('RGB').resize((self.default_image_dim_input,self.default_image_dim_input)))
        if len(img_tensor.shape) == 3:
            img_tensor = img_tensor.unsqueeze(0)
        
        return img_tensor.cuda()

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"
    
class IdentityTransform:

    def __call__(self, pic):
            return pic

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
    
class JPGToPILimage:
    def __call__(self, image_path):
        image = Image.open(image_path)
        return image
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"