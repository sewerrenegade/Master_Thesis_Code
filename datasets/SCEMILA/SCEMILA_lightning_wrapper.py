import typing
import pytorch_lightning as pl
from datasets.SCEMILA.base_SCEMILA import baseSCEMILAdataset,INDEXER
import torchvision.transforms as transforms
import os
from sklearn.model_selection import KFold

from  models.SCEMILA import label_converter

from torch.utils.data import Dataset, random_split, Subset
from torch.utils.data.dataloader import DataLoader

class SCEMILA(pl.LightningDataModule):
    def __init__(
            self,
            data_path: str ='data/SCEMILA/extracted_features/mll_mil_extracted',
            num_workers: int = 4,
            val_split = 0.2,
            k_fold = -1,
            patient_bootstrap_exclude=None
            ):
        super().__init__()
        self.data_dir = data_path
        self.num_workers = num_workers
        self.val_split = val_split
        self.k_fold = k_fold
        self.indexer = INDEXER
        self.train_indicies = self.process_indicies(INDEXER.train_indicies,patient_bootstrap_exclude)
        self.test_indicies = self.process_indicies(INDEXER.test_indicies,patient_bootstrap_exclude)
        self.create_train_dataset()
        self.create_test_dataset()
        if self.k_fold != -1:
            self.fold = 0
            kf = KFold(n_splits=k_fold, shuffle=True)
            self.all_splits = [k for k in kf.split(self.train_indicies)]

    
    def setup(self, stage=None):
        if self.k_fold != -1:
            if stage == 'fit':
                train_indexes, test_indexes = self.all_splits[self.fold]
                self.train_dataset = [self.dataset_full[i] for i in train_indexes]
                self.test_dataset = [self.dataset_full[i] for i in test_indexes]
            elif stage == 'test':
                self.fold = self.fold + 1
            

    def create_train_dataset(self):
        train_dataset = baseSCEMILAdataset(self.train_indicies)
        if self.k_fold == -1:
            train_dataset_size = len(train_dataset)
            val_size = int(self.val_split * train_dataset_size)
            train_size = train_dataset_size - val_size
            self.train_dataset, self.val_dataset = random_split(train_dataset, [train_size, val_size])
        else:
            self.full_dataset = train_dataset

    def create_test_dataset(self):
        self.test_dataset = baseSCEMILAdataset(self.test_indicies)
    
    
    def process_indicies(self,data,patient_bootstrap_exclude):
        paths = []
        labels = []
        for key, val in data.items():
            if patient_bootstrap_exclude is not None:
                if(0 <= patient_bootstrap_exclude < len(val)):
                    path_excluded = val.pop(patient_bootstrap_exclude)
                    patient_bootstrap_exclude = -1
                    print("Bootstrapping. Excluded: ", path_excluded)
                else:
                    patient_bootstrap_exclude -= len(val)

            paths.extend(val)

            label =  INDEXER.get_class_int(key)
            labels.extend([label] * len(val))
        return paths,labels
        

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset, 
            num_workers=self.num_workers,
            shuffle=True,
            drop_last=True,
            )

    def val_dataloader(self):
        data_temp =  DataLoader(
            self.val_dataset, 
            num_workers=self.num_workers,
            shuffle=True,
            drop_last=True,
            )
        return data_temp

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset, 
            num_workers=self.num_workers,
            drop_last=True,
            )

    def predict_dataloader(self):
        return DataLoader(
            self.train_dataset, 
            num_workers=self.num_workers,
            drop_last=True,
            )