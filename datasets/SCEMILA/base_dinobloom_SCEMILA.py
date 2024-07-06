from datasets.base_dataset_abstraction import BaseDataset
from datasets.SCEMILA.SEMILA_indexer import SCEMILA_Indexer
import random
from collections.abc import Iterable
import numpy as np
import os
from models.DinoBloom.dinobloom_hematology_feature_extractor import get_dino_bloom,DINOBLOOM_TRANSFORMS,DINOBLOOM_NETWORKS_INFOS,DEFAULT_PATCH_NUM,DEFAULT_IMAGE_DIM
import torch

class SCEMILA_DinoBloom_feature_base(BaseDataset):

    '''MLL mil dataset class. Can be used by pytorch DataLoader '''

    def __init__(
            self,
            dino_bloom_size = "small",
            training=True,#dummy variable
            transforms_list = None,
            numpy = False,
            gpu =True,
            flatten = False,
            to_tensor = True,
            number_of_per_class_instances = 100,
            augmentation_settings = None
            ):
        super().__init__("SCEMILA/dinobloom_feature_data")
        if number_of_per_class_instances is None:
            self.augment_data = False
        else:
            self.augment_data = True
            self.augment_all = True
            self.number_of_per_class_instances = number_of_per_class_instances
        self.preload_transforms,self.transforms = self.get_transform_function(load_tiff=True,extra_transforms=transforms_list,numpy=numpy,to_gpu=False,flatten=False,to_tensor=False)
        self.augmented_preload_transform,_ = self.get_transform_function(load_tiff=True,augmentation_settings= augmentation_settings,extra_transforms=transforms_list,numpy=numpy,to_gpu=False,flatten=False,to_tensor=False)
        self.data_indicies,self.class_indicies = self.indexer.get_image_class_structure_from_indexer_instance_level()
        self.classes = list(self.class_indicies.keys())
        if self.augment_data:
            self.data_indicies,self.class_indicies, self.augmentation_list  = self.upsample_downsample_to_balance_classes()
        self.dino_bloom_encoder = get_dino_bloom(dino_bloom_size)
        self.features_loaded = {}
        
      
    def upsample_downsample_to_balance_classes(self):
        if self.augment_data:
            
            new_class_indicies = {}
            paths = []
            labels = []
            augmentation_list = []
            for key,value in self.class_indicies.items():
                if len(value)>self.number_of_per_class_instances:
                    paths.extend([self.data_indicies[i][0] for i in value[:self.number_of_per_class_instances]])
                    augmentation_list.extend([self.augment_all]*self.number_of_per_class_instances)
                elif len(value)<self.number_of_per_class_instances:
                    class_list_of_paths = value * (self.number_of_per_class_instances // len(value) + 1)
                    class_list_of_paths = class_list_of_paths[:self.number_of_per_class_instances]
                    paths.extend([self.data_indicies[i][0] for i in class_list_of_paths])
                    augmentation_list.extend([self.augment_all]*len(value))
                    augmentation_list.extend([True]*(self.number_of_per_class_instances - len(value)))
                else:
                    assert len(value) == self.number_of_per_class_instances
                    paths.extend([self.data_indicies[i][0] for i in value])
                    augmentation_list.extend([self.augment_all]*self.number_of_per_class_instances)
                new_class_indicies[key] = list(range(len(labels),len(labels)+self.number_of_per_class_instances))
                labels.extend([key]*self.number_of_per_class_instances)
            return list(zip(paths,labels)),new_class_indicies , augmentation_list
    
    def preprocess_image_for_dinobloom(self,image):
        return DINOBLOOM_TRANSFORMS(image.convert('RGB').resize((DEFAULT_IMAGE_DIM,DEFAULT_IMAGE_DIM)))

    
    def __len__(self):
        '''returns amount of images contained in object'''
        return len(self.data_indicies)

    def __getitem__(self, idx):
        '''returns specific item from this dataset'''
        if type(idx) is int:
            imgage_feaures = self.get_encoded_image(idx)
            return imgage_feaures,self.indexer.convert_from_int_to_label_instance_level(self.data_indicies[idx][1])
        elif isinstance(idx,Iterable):
            images = []
            labels = []
            for index in idx:
                images.append(self.get_encoded_image(index))
                labels.append(self.indexer.convert_from_int_to_label_instance_level(self.data_indicies[index][1]))
            return images,labels

    def get_encoded_image(self,idx):
        if idx not in self.features_loaded:
            image_path = self.data_indicies[idx][0]
            
            if self.augmentation_list is not None and self.augmentation_list[idx]:
                preload_trans = self.augmented_preload_transform
                image = preload_trans(image_path)
                # plt.imshow(image)
                # plt.show()
            else:
                preload_trans = self.preload_transforms
                image = preload_trans(image_path)
            image = torch.stack([self.preprocess_image_for_dinobloom(image)]).cuda()
            feautres_dict = self.dino_bloom_encoder(image)
            imgage_feaures = feautres_dict["x_norm_clstoken"]

            imgage_feaures = self.transforms(imgage_feaures[0].cpu().detach())
            self.features_loaded[idx] = imgage_feaures
            return imgage_feaures
        else:
            return self.features_loaded[idx]


    def get_random_samples_from_class(self, class_name, number_of_instances):
        indicies = self.class_indicies[class_name]
        random_indicies = random.sample(indicies,number_of_instances)
        return self[random_indicies]