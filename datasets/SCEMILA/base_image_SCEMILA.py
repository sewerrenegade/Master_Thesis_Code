import sys
sys.path.append('/home/milad/Desktop/Master_Thesis/code/Master_Thesis_Code')
from datasets.base_dataset_abstraction import BaseDataset
import random
from collections.abc import Iterable
import torch

class SCEMILAimage_base(BaseDataset):

    '''MLL mil dataset class. Can be used by pytorch DataLoader '''

    def __init__(
            self,
            training_mode=True,#dummy variable
            encode_with_dino_bloom = False,
            transforms_list = None,
            numpy = False,
            gpu =True,
            flatten = False,
            to_tensor = False,
            balance_dataset_classes = None,
            augmentation_settings = None
            ):
        super().__init__("SCEMILA/image_data",augmentation_settings,balance_dataset_classes=balance_dataset_classes)
        self.encode_with_dino_bloom = encode_with_dino_bloom
        self.augmentation_settings = augmentation_settings
        self.preload_transforms,self.transforms = self.get_transform_function(load_tiff=True,extra_transforms=transforms_list,numpy=numpy,to_gpu=gpu,flatten=flatten,to_tensor=to_tensor,augmentation_settings= augmentation_settings)
        if self.encode_with_dino_bloom:
            self.dino_enc = self.get_dino_bloom_transform()
            self.get_item_function = self.get_and_dino_encoded_tif_image
        else:
            self.get_item_function = self.get_single_tiff_image
        self.loaded_data = {}
                
    def __len__(self):
        '''returns amount of images contained in object'''
        return len(self.indicies_list)

    def __getitem__(self, idx):
        '''returns specific item from this dataset'''
        if type(idx) is int:
            return self.get_item_function(idx)
        elif isinstance(idx,Iterable):
            images = []
            labels = []
            for index in idx:
                image , label = self.get_item_function(index)
                images.append(self.transforms(image))
                labels.append(label)
            return images,labels
        
    def get_single_tiff_image(self, idx):
        if idx not in self.loaded_data:
            image_path = self.indicies_list[idx][0]
            image = self.transforms(self.preload_transforms(image_path))
            self.loaded_data[idx] =  image,self.indexer.convert_from_int_to_label_instance_level(self.indicies_list[idx][1])
        return self.loaded_data[idx]

    
    def get_and_dino_encoded_tif_image(self,idx):
        if idx not in self.loaded_data:
            image_path = self.indicies_list[idx][0]
            preload_trans = self.preload_transforms
            image = preload_trans(image_path)
            dino_features = self.dino_enc(image)
            image_feaures = self.transforms(dino_features[0].cpu().detach())
            self.loaded_data[idx] = image_feaures, self.indexer.convert_from_int_to_label_instance_level(self.indicies_list[idx][1])
        return self.loaded_data[idx]
    
if __name__ == '__main__':
    db = SCEMILAimage_base(balance_dataset_classes = 100,gpu= False)
    x = db[889]
    pass

