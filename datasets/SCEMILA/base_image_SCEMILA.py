import sys
sys.path.append('/home/milad/Desktop/Master_Thesis/code/Master_Thesis_Code')
from datasets.base_dataset_abstraction import BaseDataset
import random
from collections.abc import Iterable

class SCEMILAimage_base(BaseDataset):

    '''MLL mil dataset class. Can be used by pytorch DataLoader '''

    def __init__(
            self,
            training=True,#dummy variable
            transforms_list = None,
            numpy = False,
            gpu =True,
            flatten = False,
            to_tensor = True,
            number_of_per_class_instances = 100,
            augmentation_settings = None
            ):
        super().__init__("SCEMILA/image_data")
        if number_of_per_class_instances is None:
            self.augment_data = False
        else:
            self.augment_data = True
            self.augment_all = True
            self.number_of_per_class_instances = number_of_per_class_instances
        self.preload_transforms,self.transforms = self.get_transform_function(load_tiff=True,extra_transforms=transforms_list,numpy=numpy,to_gpu=gpu,flatten=flatten,to_tensor=to_tensor)
        self.augmented_preload_transform,_ = self.get_transform_function(load_tiff=True,augmentation_settings= augmentation_settings,extra_transforms=transforms_list,numpy=numpy,to_gpu=False,flatten=False,to_tensor=False)
        self.data_indicies,self.class_indicies = self.indexer.get_image_class_structure_from_indexer_instance_level()
        self.classes = list(self.class_indicies.keys())
        if self.augment_data:
            self.data_indicies,self.class_indicies, self.augmentation_list  = self.upsample_downsample_to_balance_classes()
        
      
    def upsample_downsample_to_balance_classes(self):
        if self.augment_data:
            
            new_class_indicies = {}
            paths = []
            labels = []
            augmentation_list = []
            for key,value in self.class_indicies.items():
                if len(value)>self.number_of_per_class_instances:
                    paths.extend([self.data_indicies[i][0] for i in value[:self.number_of_per_class_instances]])
                    augmentation_list.extend([self.augment_data]*self.number_of_per_class_instances)
                elif len(value)<self.number_of_per_class_instances:
                    class_list_of_paths = value * (self.number_of_per_class_instances // len(value) + 1)
                    class_list_of_paths = class_list_of_paths[:self.number_of_per_class_instances]
                    paths.extend([self.data_indicies[i][0] for i in class_list_of_paths])
                    augmentation_list.extend([self.augment_data]*len(value))
                    augmentation_list.extend([True]*(self.number_of_per_class_instances - len(value)))
                else:
                    assert len(value) == self.number_of_per_class_instances
                    paths.extend([self.data_indicies[i][0] for i in value])
                    augmentation_list.extend([self.augment_data]*self.number_of_per_class_instances)
                new_class_indicies[key] = list(range(len(labels),len(labels)+self.number_of_per_class_instances))
                labels.extend([key]*self.number_of_per_class_instances)
            return list(zip(paths,labels)),new_class_indicies , augmentation_list
                
    def __len__(self):
        '''returns amount of images contained in object'''
        return len(self.data_indicies)

    def __getitem__(self, idx):
        '''returns specific item from this dataset'''
        if type(idx) is int:
            return self.get_single_item(idx)
        elif isinstance(idx,Iterable):
            images = []
            labels = []
            for index in idx:
                image , label = self.get_single_item(index)
                images.append(self.transforms(image))
                labels.append(label)
            return images,labels
        
    def get_single_item(self, idx):
        image_path = self.data_indicies[idx][0]
        if not self.augment_data:
            image = self.transforms(self.preload_transforms(image_path))
            return image,self.indexer.convert_from_int_to_label_instance_level(self.data_indicies[idx][1])
        else:
            if self.augmentation_list[idx]:
                aug_image= self.augmented_preload_transform(image_path)
                image = self.transforms(aug_image)
                return image,self.indexer.convert_from_int_to_label_instance_level(self.data_indicies[idx][1])
            else:
                image = self.transforms(self.preload_transforms(image_path))
                return image,self.indexer.convert_from_int_to_label_instance_level(self.data_indicies[idx][1])
            
    def get_random_samples_from_class(self, class_name, number_of_instances):
        indicies = self.class_indicies[class_name]
        random_indicies = random.sample(indicies,number_of_instances)
        return self[random_indicies]
    
if __name__ == '__main__':
    db = SCEMILAimage_base(number_of_per_class_instances = 100,gpu= False)
    x = db[889]
    pass

