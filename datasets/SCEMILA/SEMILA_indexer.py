import pytorch_lightning as pl
import os
from sklearn.model_selection import KFold
import pandas as pd
import random
import re

class SCEMILA_Indexer:
    def __init__(self,SCEMILA_Path = "data/SCEMILA/"):
        self.path_data = SCEMILA_Path
        self.dict_path = f"{SCEMILA_Path}meta_files/metadata.csv"
        self.image_level_annotations_file = f"{SCEMILA_Path}meta_files/image_annotation_master.csv"
        self.tiff_image_pattern = re.compile(r"(.*\/)image_(\d+)\.tif$")
        self.indicies, self.bag_meta_df = self.define_dataset()
        self.classes = list(self.indicies.keys())
        
        #bag level indexing
        self.train_indicies,self.test_indicies = self.seperate_test_train_data()
        self.train_class_count ,self.test_class_count = self.get_class_count(self.classes,self.train_indicies,self.test_indicies)

        #instance level indexing
        self.instance_level_annotations_by_class,self.instance_level_class_count,self.instance_classes = self.read_csv_instance_level_annotations()
        pass
   
    def get_image_class_structure_from_indexer_instance_level(self):
        paths = []
        labels = []
        dict_struct = {}
        for key in self.instance_classes:
            class_paths = self.instance_level_annotations_by_class[key]
            nb_paths_in_class = len(class_paths)
            paths.extend(class_paths)
            labels.extend([key]*nb_paths_in_class)
            dict_struct[key] = list(range(len(labels)-nb_paths_in_class,len(labels)))#class_paths[:nb_of_instances_to_take]
        assert len(paths) == len(labels)
        return list(zip(paths,labels)),dict_struct
    
    def get_feature_balanced_class_structure_from_indexer_instance_level(self):
        paths_to_patients = []
        cell_indecies = []
        labels = []
        dict_struct = {}
        for key in self.instance_classes:
            class_paths = self.instance_level_annotations_by_class[key]
            nb_paths_in_class = len(class_paths)
            paths, cell_idxs = self.extract_cell_id_from_paths(class_paths)
            paths_to_patients.extend(paths)
            cell_indecies.extend(cell_idxs)
            labels.extend([key]*nb_paths_in_class)
            dict_struct[key] = list(range(len(labels)-nb_paths_in_class,len(labels)))#class_paths[:nb_of_instances_to_take]
        return list(zip(list(zip(paths_to_patients,cell_indecies)),labels)),dict_struct
    
    def extract_cell_id_from_paths(self,paths):
        cell_indices = []
        paths_to_patients = []
        for path in paths:
            match = self.tiff_image_pattern.match(path)
            if match:
                paths_to_patients.append(match.group(1).replace("image_data", "fnl34_feature_data"))
                cell_indices.append(int(match.group(2)))
            else:
                raise ValueError("The provided path does not match the expected pattern.")
        return paths_to_patients,cell_indices
    
    def convert_from_int_to_label_instance_level(self,int_or_label):#TODO HANDLE multiple inputs
        if type(int_or_label) is int:
            return self.instance_classes[int_or_label]
        elif type(int_or_label) is str:
            return self.instance_classes.index(int_or_label)

    def get_class_int(self,class_name):
        return self.classes.index(class_name)
    

    def seperate_test_train_data(self):
        random.seed(42)# ensures it is split the same way
        train_data = {}
        test_data = {}
        for label in self.classes:
            train_data[label] , test_data[label] = self.split_list(self.indicies[label])
        return train_data,test_data
    
    def read_csv_instance_level_annotations(self,number_of_classes  =10,include_other = False):
        df = pd.read_csv(self.image_level_annotations_file)
        labels = df["mll_annotation"] 
        patient_ids = df["ID"]
        image_names = df["im_tiffname"]
        class_paths = {}
        paths = []
        for label,id,image_name in zip(labels,patient_ids,image_names):
            path = self.get_image_path_from_ID_image_name(id,image_name)
            paths.append(path)
            try:
                class_paths[label].append(path)
            except KeyError:
                class_paths[label]=[path]
        length_dict = {key: len(value) for key, value in class_paths.items()}
        class_paths,length_dict,class_order = self.reformulate_dataset_k_class_including_other(class_paths,length_dict,number_of_classes,include_other)

        return class_paths,length_dict,class_order

    def get_image_path_from_ID_image_name(self,ID,image_name):
        label_folder_name = self.bag_meta_df.loc[ID,"bag_label"]
        return os.path.join(self.path_data,"image_data",label_folder_name,ID,f"image_{image_name}")

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
        df_data_master = pd.read_csv(self.dict_path).set_index('patient_id')

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
        return merge_dict_processed,df_data_master

    
    def process_label(self,row):
        ''' This function receives the patient row from the mll_data_master, and
        computes the patient label. If it returns None, the patient is left out
        --> this can be used for additional filtering.
        '''

        lbl_out = row.bag_label

        # alternative labelling, e.g. by sex:
        # lbl_out = ['female', 'male'][int(row.sex_1f_2m)-1]

        return lbl_out

    def top_k_keys_with_highest_values(self,dict, k):
        # Sort the dictionary items by their values in descending order
        sorted_items = sorted(dict.items(), key=lambda item: item[1], reverse=True)
        
        # Extract the top k keys
        top_k_keys = [item[0] for item in sorted_items[:k]]
        
        return top_k_keys
    
    def reformulate_dataset_k_class_including_other(self,class_paths,length_dict,number_of_classes,include_other_class):
        sorted_classes = sorted(length_dict.items(), key=lambda item: item[1], reverse=True)
        sorted_classes = [item[0] for item in sorted_classes]
        other_classes = ["other"]
        if include_other_class:
            other_classes.extend(sorted_classes[number_of_classes+1:])
            included_classes = sorted_classes[:number_of_classes]
            new_class_paths = {"other":[]}
            for _class in sorted_classes:
                if _class in included_classes:
                    new_class_paths[_class] = class_paths[_class]
                else:
                    new_class_paths["other"].extend(class_paths[_class])
            length_dict = {key: len(value) for key, value in new_class_paths.items()}
        else:
            other_classes.extend(sorted_classes[number_of_classes+2:])
            included_classes = sorted_classes[1:number_of_classes+1]
            new_class_paths = {"other":[]}
            for _class in sorted_classes:
                if _class in included_classes:
                    new_class_paths[_class] = class_paths[_class]
                else:
                    new_class_paths["other"].extend(class_paths[_class])
            del new_class_paths["other"]
            length_dict = {key: len(value) for key, value in new_class_paths.items()}
        return new_class_paths,length_dict,included_classes



    
    def get_class_count(self,classes,train_indicies,test_indicies = []):
        train_class_count={}
        test_class_count={}
        for class_name in classes:
            train_class_count[class_name] = len(train_indicies[class_name])
            test_class_count[class_name] = len(test_indicies[class_name])
        
        return train_class_count,test_class_count
    



if __name__ == "__main__":
    x = SCEMILA_Indexer()
    a = x.get_image_class_structure_from_indexer_instance_level()
    print(list(x.classes)[2])


