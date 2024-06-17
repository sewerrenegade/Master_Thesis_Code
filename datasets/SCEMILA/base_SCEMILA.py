import sys
sys.path.append('/home/milad/Desktop/Master_Thesis/code/Master_Thesis_Code')
from datasets.base_dataset_abstraction import baseDataset
from datasets.SCEMILA.SEMILA_indexer import SCEMILA_Indexer
import numpy as np
import os
import torch
import random
from models.DinoBloom.dinobloom_hematology_feature_extractor import get_dino_bloom,DINOBLOOM_TRANSFORMS,DINOBLOOM_NETWORKS_INFOS,DEFAULT_PATCH_NUM,DEFAULT_IMAGE_DIM
from collections.abc import Iterable

INDEXER = SCEMILA_Indexer()

class SCEMILAfeature_MIL_base(baseDataset):

    '''MLL mil dataset class. Can be used by pytorch DataLoader '''

    def __init__(
            self,
            data_indicies,
            prefix = "fnl34_",
            aug_im_order=True
            ):
        '''dataset constructor. Accepts parameters:
        - folds: list of integers or integer in range(NUM_FOLDS) which are set in beginning of this file.
                Used to define split of data this dataset should countain, e.g. 0-7 for train, 8 for val,
                9 for test
        - aug_im_order: if True, images in a bag are shuffled each time during loading
        - split: store information about the split within object'''
        super().__init__("SCEMILA/feature_data")
        self.aug_im_order = aug_im_order
        self.prefix = prefix
        # grab data split for corresponding folds
        self.data_indicies = data_indicies
        self.features_loaded = {}

        
    def __len__(self):
        '''returns amount of images contained in object'''
        return len(self.data_indicies)

    def __getitem__(self, idx):
        '''returns specific item from this dataset'''

        # grab images, patient id and label
        path = self.data_indicies[idx][0]
        added_folder = "processed"
        # only load if object has not yet been loaded
        if (path not in self.features_loaded):
            bag = np.load(
                os.path.join(
                    path,added_folder,
                    self.prefix +
                    'bn_features_layer_7.npy'))
            self.features_loaded[path] = bag
        else:
            bag = self.features_loaded[path].copy()

        label = self.data[idx][1]
        pat_id = path

        # shuffle features by image order in bag, if desired
        if(self.aug_im_order):
            num_rows = bag.shape[0]
            new_idx = torch.randperm(num_rows)
            bag = bag[new_idx, :]

        # # prepare labels as one-hot encoded
        # label_onehot = torch.zeros(len(self.data))
        # label_onehot[label] = 1

        label_regular = torch.Tensor([label]).long()

        return bag, label_regular, pat_id
    
class SCEMILA_fnl34_feature_base(baseDataset):

    '''MLL mil dataset class. Can be used by pytorch DataLoader '''

    def __init__(
            self,
            training=True,#dummy variable for now
            transforms_list = None,
            prefix = "fnl34_",
            numpy = False,
            gpu =True,
            flatten = False,
            to_tensor = False
            ):
        '''dataset constructor. Accepts parameters:
        - folds: list of integers or integer in range(NUM_FOLDS) which are set in beginning of this file.
                Used to define split of data this dataset should countain, e.g. 0-7 for train, 8 for val,
                9 for test
        - aug_im_order: if True, images in a bag are shuffled each time during loading
        - split: store information about the split within object'''
        super().__init__("SCEMILA/feature_data")
        self.prefix = prefix   
        self.preload_transforms,self.transforms = self.get_transform_function(extra_transforms=transforms_list,numpy=numpy,to_gpu=gpu,flatten=flatten,to_tensor=to_tensor)
        self.data_indicies,self.class_indicies = INDEXER.get_feature_balanced_class_structure_from_indexer_instance_level()
        self.classes = list(self.class_indicies.keys())
        self.features_loaded = {}

    def get_n_random_instances_from_class(self, class_name, number_of_instances):
        indicies = self.class_indicies[class_name]
        random_indicies = random.sample(indicies,number_of_instances)
        return self[random_indicies]
        
    def __len__(self):
        '''returns amount of images contained in object'''
        return len(self.data_indicies)

    def __getitem__(self, idx):
        '''returns specific item from this dataset'''
        if type(idx) is int:
            return self.get_singe_cell_feature(idx)
        elif isinstance(idx,Iterable):
            images = []
            labels = []
            for index in idx:
                image_feature , label = self.get_singe_cell_feature(index)
                images.append(self.transforms(image_feature))
                labels.append(label)
            return np.array(images),labels
    
    def get_singe_cell_feature(self,index):
        # grab images, patient id and label
        patient_path,cell_index = self.data_indicies[index][0]
        added_folder = "processed"
        # only load if object has not yet been loaded
        if (patient_path not in self.features_loaded):
            bag = np.load(
                os.path.join(
                    patient_path,added_folder,
                    self.prefix +
                    'bn_features_layer_7.npy'))
            self.features_loaded[patient_path] = bag
        else:
            bag = self.features_loaded[patient_path].copy()

        label = INDEXER.convert_from_int_to_label_instance_level(self.data_indicies[index][1])


        return bag[cell_index,:], label
    
class SCEMILA_DinoBloom_feature_base(baseDataset):

    '''MLL mil dataset class. Can be used by pytorch DataLoader '''

    def __init__(
            self,
            dino_bloom_size = "small",
            training=True,#dummy variable
            transforms_list = None,
            numpy = False,
            gpu =True,
            flatten = False,
            to_tensor = True
            ):
        super().__init__("SCEMILA/dinobloom_feature_data")
        self.preload_transforms,self.transforms = self.get_transform_function(load_tiff=True,extra_transforms=transforms_list,numpy=numpy,to_gpu=False,flatten=False,to_tensor=False)
        self.data_indicies,self.class_indicies = INDEXER.get_image_balanced_class_structure_from_indexer_instance_level()
        self.classes = list(self.class_indicies.keys())
        self.dino_bloom_encoder = get_dino_bloom(dino_bloom_size)
        self.encoder_saved_values = {}
    
    def preprocess_image_for_dinobloom(self,image):
        return DINOBLOOM_TRANSFORMS(image.convert('RGB').resize((DEFAULT_IMAGE_DIM,DEFAULT_IMAGE_DIM)))

    
    def __len__(self):
        '''returns amount of images contained in object'''
        return len(self.data_indicies)

    def __getitem__(self, idx):
        '''returns specific item from this dataset'''
        if type(idx) is int:
            imgage_feaures = self.get_encoded_image(idx)
            return imgage_feaures,INDEXER.convert_from_int_to_label_instance_level(self.data_indicies[idx][1])
        elif isinstance(idx,Iterable):
            images = []
            labels = []
            for index in idx:
                images.append(self.get_encoded_image(index))
                labels.append(INDEXER.convert_from_int_to_label_instance_level(self.data_indicies[index][1]))
            return images,labels
        
    def get_encoded_image(self,idx):
        if idx not in self.encoder_saved_values:
            image_path = self.data_indicies[idx][0]
            image = self.preload_transforms(image_path)
            image = torch.stack([self.preprocess_image_for_dinobloom(image)]).cuda()
            feautres_dict = self.dino_bloom_encoder(image)
            imgage_feaures = feautres_dict["x_norm_clstoken"]
            imgage_feaures = self.transforms(imgage_feaures[0].cpu().detach())
            self.encoder_saved_values[idx] = imgage_feaures
            return imgage_feaures
        else:
            return self.encoder_saved_values[idx]


    def get_n_random_instances_from_class(self, class_name, number_of_instances):
        indicies = self.class_indicies[class_name]
        random_indicies = random.sample(indicies,number_of_instances)
        return self[random_indicies]


class SCEMILAimage_base(baseDataset):

    '''MLL mil dataset class. Can be used by pytorch DataLoader '''

    def __init__(
            self,
            training=True,#dummy variable
            transforms_list = None,
            numpy = False,
            gpu =True,
            flatten = False,
            to_tensor = True
            ):
        super().__init__("SCEMILA/image_data")
        self.preload_transforms,self.transforms = self.get_transform_function(load_tiff=True,extra_transforms=transforms_list,numpy=numpy,to_gpu=gpu,flatten=flatten,to_tensor=to_tensor)
        self.data_indicies,self.class_indicies = INDEXER.get_image_balanced_class_structure_from_indexer_instance_level()
        self.classes = list(self.class_indicies.keys())
        pass
    
    
    def __len__(self):
        '''returns amount of images contained in object'''
        return len(self.data_indicies)

    def __getitem__(self, idx):
        '''returns specific item from this dataset'''
        if type(idx) is int:
            image_path = self.data_indicies[idx][0]
            image = self.transforms(self.preload_transforms(image_path))
            return image,INDEXER.convert_from_int_to_label_instance_level(self.data_indicies[idx][1])
        elif isinstance(idx,Iterable):
            images = []
            labels = []
            for index in idx:
                image_path = self.data_indicies[index][0]
                image = self.preload_transforms(image_path)
                label = INDEXER.convert_from_int_to_label_instance_level(self.data_indicies[index][1])
                images.append(self.transforms(image))
                labels.append(label)
            return images,labels

    def get_n_random_instances_from_class(self, class_name, number_of_instances):
        indicies = self.class_indicies[class_name]
        random_indicies = random.sample(indicies,number_of_instances)
        return self[random_indicies]
    
if __name__ == '__main__':
    db = SCEMILAimage_base()
    x = db[0]
    pass