import glob
import os
import numpy as np


from datasets.mil_dataset_abstraction import BaseMILDataset

from distance_functions.functions.basic_distance_functions import EuclideanDistance
from distance_functions.functions.embedding_functions import EmbeddingFunction
from datasets.base_dataset_abstraction import BaseDataset
from collections.abc import Iterable
import torch

class SCEMILA_base(BaseDataset):

    '''MLL mil dataset class. Can be used by pytorch DataLoader '''

    def __init__(
            self,
            training_mode=True,#dummy variable
            encode_with_dino_bloom = False,
            transforms_list = None,
            numpy = False,
            gpu =True,
            flatten = False,
            to_tensor = True,
            balance_dataset_classes = None,
            augmentation_settings = None,
            grayscale = False,
            resize=False
            ):
        super().__init__("SCEMILA/image_data",augmentation_settings,balance_dataset_classes=balance_dataset_classes,training=training_mode)
        self.encode_with_dino_bloom = encode_with_dino_bloom
        self.augmentation_settings = augmentation_settings
        self.preload_transforms,self.transforms = self.get_transform_function(load_tiff=True,extra_transforms=transforms_list,numpy=numpy,to_gpu = gpu and not encode_with_dino_bloom,flatten=flatten,to_tensor=to_tensor,augmentation_settings= augmentation_settings,resize=resize,grayscale=grayscale)
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
                image, label = self.get_item_function(index)
                images.append(image)
                labels.append(label)
            return images,labels
        
    def get_only_pretransform_item(self,idx):
        '''returns specific item from this dataset'''
        if type(idx) is int:
            return self.get_item_function(idx)
        elif isinstance(idx,Iterable):
            images = []
            labels = []
            for index in idx:
                image , label = self.get_single_tiff_imagewithout_post_trans(index)
                images.append(image)
                labels.append(label)
            return images,labels
           
    def get_single_tiff_image(self, idx):
        if idx not in self.loaded_data:
            image_path = self.indicies_list[idx][0]
            image = self.transforms(self.preload_transforms(image_path))
            self.loaded_data[idx] =  image,self.indexer.convert_from_int_to_label_instance_level(self.indicies_list[idx][1])
        return self.loaded_data[idx]
    
    def get_single_tiff_imagewithout_post_trans(self, idx):
        image_path = self.indicies_list[idx][0]
        image = self.preload_transforms(image_path)
        return image,self.indexer.convert_from_int_to_label_instance_level(self.indicies_list[idx][1])

    
    def get_and_dino_encoded_tif_image(self,idx):
        if idx not in self.loaded_data:
            image_path = self.indicies_list[idx][0]
            preload_trans = self.preload_transforms
            image = preload_trans(image_path)
            dino_features = self.dino_enc(image)
            image_feaures = self.transforms(dino_features[0].cpu().detach())
            self.loaded_data[idx] = image_feaures, self.indexer.convert_from_int_to_label_instance_level(self.indicies_list[idx][1])
        return self.loaded_data[idx]
    
    
class SCEMILA_MIL_base(BaseMILDataset):

    '''MLL mil dataset class. Can be used by pytorch DataLoader '''

    def __init__(
            self,
            training_mode=True,#dummy variable
            input_type = "images",
            encode_with_dino_bloom=False,
            root_dir = "data/",
            balance_dataset_classes = None,
            gpu = True, grayscale= False,
            numpy = False,flatten = False,
            to_tensor = True,
            augmentation_settings = None,
            topo_settings = None
            ):
        super().__init__("MIL_SCEMILA",training = training_mode,augmentation_settings=augmentation_settings,balance_dataset_classes=balance_dataset_classes)      
        self.input_type = input_type
        self.gpu = gpu
        self.topo_settings = topo_settings
        self.loaded_data = {}
        self.encode_with_dino_bloom = encode_with_dino_bloom
        self.preload_transforms,self.transforms,self.to_gpu_transform = self.get_transform_function(load_tiff=True,extra_transforms=[],numpy=numpy,to_gpu = self.gpu and not encode_with_dino_bloom,flatten=flatten,to_tensor=to_tensor,augmentation_settings= augmentation_settings,grayscale=grayscale)
        if self.input_type == "images":
            if self.encode_with_dino_bloom:
                self.loaded_dinobloom_embedded_ds = self.create_or_get_serialized_dinobloom_dataset()
                self.get_item_function = self.get_saved_dino_bag
            else:
                self.get_item_function = self.get_single_tiff_bag
            self.topo_labels = None
        elif self.input_type == "fnl34":
            self.prefix = "fnl34_"
            self.get_item_function = self.get_fnl34_bag
        if self.topo_settings is not None:
            self.add_topological_label(topo_settings)
            
    def __getitem__(self, idx):
        '''returns specific item from this dataset'''
        #print("get")
        if type(idx) is int:
            x = self.get_item_function(idx)
            return self.to_gpu_transform(x)
        elif isinstance(idx,Iterable):# wont work for training
            images = []
            labels = []
            for index in idx:
                image , label = self.get_item_function(index)
                images.append(image)
                labels.append(label)
            return self.to_gpu_transform(images),labels
        elif isinstance(idx,np.int64):
            return self[int(idx)]

    
    
    def create_or_get_serialized_dinobloom_dataset(self):
        print("!!!!!!!!reate_or_get_serialized_dinobloom_dataset!!!!!!!!!")
        recalculate = False
        assert self.encode_with_dino_bloom
        self.get_item_function = self.get_and_dino_encode_tif_bag
        from datasets.embedded_datasets.generators.embedding_descriptor import EmbeddingDescriptor
        dataset_descriptor = EmbeddingDescriptor("Serialized DinoBloom",self,None,None,None)
        if not dataset_descriptor.already_exists() or recalculate:
            self.dino_enc = self.get_dino_bloom_transform()
        embedded_dataset = dataset_descriptor.generate_embedding_from_descriptor(recalculate)
        return embedded_dataset
        
    def get_fnl34_bag(self,idx):
        patient_path,patient_label = self.indicies_list[idx]
        added_folder = "processed"
        fnl_34_patient_path = patient_path.replace("image_data", "fnl34_feature_data")
        bag = np.load(
                os.path.join(
                    fnl_34_patient_path,added_folder,
                    self.prefix +
                    'bn_features_layer_7.npy'))
        
        label_regular = self.indexer.convert_from_int_to_label_bag_level(patient_label)
        self.loaded_data[idx] = bag, label_regular
        return self.loaded_data[idx]
    
    def get_saved_dino_bag(self,idx):
        bag_features,patient_label = self.loaded_dinobloom_embedded_ds[idx]
        bag_features,patient_label = torch.tensor(bag_features[0]),torch.tensor(patient_label[0])
        
        if self.gpu:
            bag_features,patient_label = self.to_gpu_transform(bag_features),self.to_gpu_transform(patient_label)
        self.loaded_data[idx] = bag_features,patient_label
        return self.loaded_data[idx]
        

    def add_topological_label(self,topo_dataset_settings):
        print("!!!!!!!!!!!!!!!!!!!!!!!!!add_topological_label!!!!!!!!!!!!!!!!!!")
        normalize_distance_matricies = True
        if "normalize_distance_matricies" in  topo_dataset_settings:
            normalize_distance_matricies = topo_dataset_settings.get("normalize_distance_matricies",True)
            del topo_dataset_settings["normalize_distance_matricies"]
            
        dataset = SCEMILA_MIL_base(training_mode= True,
            gpu = False, 
            numpy = True, **topo_dataset_settings["dataset_settings"])
        
        embedding_function = EmbeddingFunction(**topo_dataset_settings.get("embedding_settings",{"function_name": None,"function_settings" : None}))
        nb_of_grouped_bags=topo_dataset_settings["nb_of_grouped_bags"]
        distance_function = EuclideanDistance()
        from datasets.topological_datasets.topo_dataset_desciptor import TopoDatasetDescriptor
        topo_desc = TopoDatasetDescriptor(name = "test_topo" ,dataset= dataset,nb_of_grouped_bags= nb_of_grouped_bags,embedding_function = embedding_function,distance_function =distance_function)
        
        per_bag_dist_mat, bag_instance_order = topo_desc.generate_or_get_topo_dataset_from_descriptor(normalize_distance_matricies=normalize_distance_matricies)
        assert len(per_bag_dist_mat) == len(bag_instance_order)
        assert len(per_bag_dist_mat) == len(self)
        for bag_index in range(len(self)):
            dist_mat_instance_order = bag_instance_order[bag_index]
            dataset_bag_instance_order = self.get_image_paths_from_index(bag_index)
            assert dist_mat_instance_order == dataset_bag_instance_order  # make sure the instances in the bag have the same order, this is importqnat because we are not using wasserstein (no point matching or correspondence)
            pass
        self.topo_labels = per_bag_dist_mat
        
        
    def __len__(self):
        '''returns amount of images contained in object'''
        return len(self.indicies_list)


        
        
    def get_only_pretransform_item(self,idx):
        '''returns specific item from this dataset'''
        if type(idx) is int:
            return  self.get_single_tiff_image_without_post_trans(idx)
        elif isinstance(idx,Iterable):
            images = []
            labels = []
            for index in idx:
                image , label = self.get_single_tiff_image_without_post_trans(index)
                images.append(image)
                labels.append(label)
            return images,labels
           
    def get_single_tiff_bag(self, idx):
        #if idx not in self.loaded_data:
        instances_in_bag_paths = self.get_image_paths_from_index(idx)
        bag_data = []
        for img_path in instances_in_bag_paths: 
            image = self.transforms(self.preload_transforms(img_path))
            bag_data.append(image)
        if self.topo_labels is None:
            x = self.convert_bag_list(bag_data), self.indexer.convert_from_int_to_label_bag_level(self.indicies_list[idx][1])
        else:
            x = self.convert_bag_list(bag_data), self.indexer.convert_from_int_to_label_bag_level(self.indicies_list[idx][1]), self.topo_labels[idx]
        return x
    
    def get_single_tiff_image_without_post_trans(self, idx):
        instances_in_bag_paths = self.get_image_paths_from_index(idx)
        image = self.preload_transforms(instances_in_bag_paths[0])
        return image,self.indexer.convert_from_int_to_label_bag_level(self.indicies_list[idx][1])

    def get_image_paths_from_index(self,idx):
        patient_path = self.indicies_list[idx][0]
        instances_in_bag_paths = [f for f in glob.glob(os.path.join(patient_path, '*.tif'))]
        instances_in_bag_paths.sort()
        return instances_in_bag_paths
        
    def get_and_dino_encode_tif_bag(self,idx):
        if idx not in self.loaded_data:
            instances_in_bag_paths = self.get_image_paths_from_index(idx)
            bag_data = []
            for img_path in instances_in_bag_paths:
                image = self.preload_transforms(img_path)
                dino_features = self.dino_enc(image)
                image_features = self.transforms(dino_features[0].cpu().detach())
                bag_data.append(image_features)
                
            if self.topo_settings is None:
                x = self.convert_bag_list(bag_data), self.indexer.convert_from_int_to_label_bag_level(self.indicies_list[idx][1])
            else:
                x = self.convert_bag_list(bag_data), self.indexer.convert_from_int_to_label_bag_level(self.indicies_list[idx][1]), self.topo_labels[idx]
        return x
    
if __name__ == '__main__':
    db = SCEMILA_base(balance_dataset_classes = 100,gpu= False)
    x = db[889]
    pass

