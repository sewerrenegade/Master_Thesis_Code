import pytorch_lightning as pl
import os
from sklearn.model_selection import KFold
import pandas as pd
import torchvision.transforms as transforms
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import Dataset
from torchvision.datasets import MNIST
import json
from collections.abc import Iterable
import random

class SCEMILA_Indexer:
    def __init__(self,SCEMILA_Path = "data/SCEMILA/extracted_features/mll_mil_extracted"):
        self.path_data = SCEMILA_Path
        self.dict_path = f"{SCEMILA_Path}/metadeata.csv"
        self.indicies = self.define_dataset()
        self.classes= list(self.indicies.keys())
        self.train_indicies,self.test_indicies = self.seperate_test_train_data()
        self.train_class_count ,self.test_class_count = self.get_class_count(self.classes,self.train_indicies,self.test_indicies)


    def get_class_int(self,class_name):
        return self.classes.index(class_name)
    

    def seperate_test_train_data(self):
        random.seed(42)# ensures it is split the same way
        train_data = {}
        test_data = {}
        for label in self.classes:
            train_data[label] , test_data[label] = self.split_list(self.indicies[label])
        return train_data,test_data



    # Function to split a list into two parts
    def split_list(self,data, split_ratio=0.15):
        # Determine the number of elements for the 20% portion
        num_elements_15 = int(len(data) * split_ratio)        
        # Randomly select indices for the 20% portion
        indices_15 = random.sample(range(len(data)), num_elements_15)
        
        # Split the data based on the selected indices
        list_15 = [data[i] for i in indices_15]
        list_85 = [data[i] for i in range(len(data)) if i not in indices_15]        
        return list_85,list_15

    def define_dataset(
        self,
        filter_diff_count=-1):

        # load patient data
        df_data_master = pd.read_csv(
            '{}/metadata.csv'.format(self.path_data)).set_index('patient_id')

        print("")
        print("Filtering the dataset...")
        print("")

        # iterate over all patients in the df_data_master sheet
        merge_dict_processed = {}
        for idx, row in df_data_master.iterrows():

            # filter if patient has not enough malign cells (only if an AML patient)
            # define filter criterion by which to filter the patients by annotation
            annotations_exclude_by = [
                'pb_myeloblast',
                'pb_promyelocyte',
                'pb_myelocyte']
            annotation_count = sum(row[annotations_exclude_by])
            if annotation_count < filter_diff_count and (
                    not row['bag_label'] == 'control'):
                print("Not enough malign cells, exclude: ", row.name,
                    " with ", annotation_count, " malign cells ")
                continue

            label = self.process_label(row)
            if label is None:
                continue

            # store patient for later loading
            if label not in merge_dict_processed.keys():
                merge_dict_processed[label] = []
            patient_path = os.path.join(
                self.path_data, 'data', row['bag_label'], row.name)
            merge_dict_processed[label].append(patient_path)
        return merge_dict_processed

    
    def process_label(self,row):
        ''' This function receives the patient row from the mll_data_master, and
        computes the patient label. If it returns None, the patient is left out
        --> this can be used for additional filtering.
        '''

        lbl_out = row.bag_label

        # alternative labelling, e.g. by sex:
        # lbl_out = ['female', 'male'][int(row.sex_1f_2m)-1]

        return lbl_out


    
    def get_class_count(self,classes,train_indicies,test_indicies = []):
        train_class_count={}
        test_class_count={}
        for class_name in classes:
            train_class_count[class_name] = len(train_indicies[class_name])
            test_class_count[class_name] = len(test_indicies[class_name])
        
        return train_class_count,test_class_count
    

if __name__ == "__main__":
    print(list(SCEMILA_Indexer().classes)[2])


