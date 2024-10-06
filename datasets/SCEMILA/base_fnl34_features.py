import sys

from datasets.base_dataset_abstraction import BaseDataset
import random
from collections.abc import Iterable
import numpy as np
import os

class SCEMILA_fnl34_feature_base(BaseDataset):

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
        super().__init__("SCEMILA/fnl34_feature_data")
        self.prefix = prefix   
        self.preload_transforms,self.transforms = self.get_transform_function(extra_transforms=transforms_list,numpy=numpy,to_gpu=gpu,flatten=flatten,to_tensor=to_tensor)
        self.data_indicies,self.class_indicies = self.indexer.get_feature_balanced_class_structure_from_indexer_instance_level()
        self.classes = list(self.class_indicies.keys())
        self.features_loaded = {}
    
    def get_random_samples_from_class(self, class_name, number_of_instances):
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

        label = self.indexer.convert_from_int_to_label_instance_level(self.data_indicies[index][1])


        return bag[cell_index,:], label
    