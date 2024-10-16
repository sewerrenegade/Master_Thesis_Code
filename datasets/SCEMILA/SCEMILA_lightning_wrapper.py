import numpy
import pytorch_lightning as pl
from sklearn.model_selection import  train_test_split,StratifiedKFold
from torch.utils.data.dataloader import DataLoader
from datasets.SCEMILA.base_image_SCEMILA import SCEMILA_MIL_base
from datasets.image_augmentor import AugmentationSettings
from torch.utils.data import Subset
from datasets.indexer_scripts.indexer_utils import get_dataset_indexer

class SCEMILA(pl.LightningDataModule):
    def __init__(
            self,
            input_type = "images",#images      
            num_workers: int = 2,
            val_split = 0.1,
            k_fold = -1,
            patient_bootstrap_exclude=None,
            encode_with_dino_bloom = False,
            balance_dataset_classes = None,
            gpu = True, grayscale= False,
            numpy = False,flatten = False,
            to_tensor = True,
            augmentation_settings = None,
            topo_settings = None,
            shuffle_training = True                    
            ):
        super().__init__()
        self.name = "SCEMILA_lighting"
        self.input_type = input_type
        self.shuffle_training =  shuffle_training
        self.num_workers = num_workers
        self.val_split = val_split
        self.k_fold = k_fold
        self.encode_with_dino_bloom = encode_with_dino_bloom
        self.balance_dataset_classes = balance_dataset_classes
        self.gpu = gpu
        self.grayscale = grayscale
        self.numpy = numpy
        self.flatten = flatten
        self.to_tensor = to_tensor
        self.topo_settings = topo_settings
        self.persistent_workers = True
        self.augmentation_settings = AugmentationSettings.get_instance_from_unknown_struct(augmentation_settings)
        self.indexer = get_dataset_indexer(self.name)
        self.data = self.create_train_dataset()
        self.test_dataset = self.create_test_dataset()
    
    def setup(self, stage=None):
        if self.k_fold > 1:
            if stage == 'fit':
                train_indexes, val_indexes = self.all_splits[self.fold]
                self.train_dataset = Subset(self.full_dataset, train_indexes)
                self.val_dataset = Subset(self.full_dataset, val_indexes)
            elif stage == 'test':
                print("Incrementing Fold Index")
                self.fold = self.fold + 1
            

    def create_train_dataset(self):
        assert isinstance(self.k_fold,int)
        train_dataset =  SCEMILA_MIL_base(training_mode= True,input_type=self.input_type,
            encode_with_dino_bloom = self.encode_with_dino_bloom,
            balance_dataset_classes = self.balance_dataset_classes,
            gpu = self.gpu, grayscale= self.grayscale,
            numpy = self.numpy,flatten = self.flatten,
            to_tensor = self.to_tensor,
            augmentation_settings = self.augmentation_settings,topo_settings=self.topo_settings)
        if self.k_fold <= 1:
            targets = train_dataset.get_targets()
            train_dataset_size = len(train_dataset)
            val_size = max(int(self.val_split * train_dataset_size),len(set(targets)))
            train_size = train_dataset_size - val_size
            
            
            train_idx, val_idx = train_test_split(
                range(len(targets)), 
                test_size=val_size, 
                stratify=targets, 
                random_state=42
            )
            
            self.train_dataset = Subset(train_dataset, train_idx)
            self.val_dataset = Subset(train_dataset, val_idx)
        elif self.k_fold > 1:
            self.fold = 0
            self.full_dataset = train_dataset
            paths,labels = zip(*self.full_dataset.indicies_list)
            indicies = list(range(len(labels)))
            skf = StratifiedKFold(n_splits=self.k_fold, shuffle=self.shuffle_training, random_state = 42) # this needs to be shuffled if resampling is used, otherwise a big portion of disproportionate amount of reampled images will go into validation split, in some folds/splits
            self.all_splits = [(train_idx, val_idx) for train_idx, val_idx in skf.split(indicies, labels)]
            # kf = KFold(n_splits=self.k_fold, shuffle=self.shuffle_training, random_state= 42)
            # all_splits = [k for k in kf.split(self.full_dataset)]
            pass
            
    def create_test_dataset(self):
        test_dataset = SCEMILA_MIL_base(training_mode= False,input_type=self.input_type,
            encode_with_dino_bloom = self.encode_with_dino_bloom,
            balance_dataset_classes = None,
            gpu = self.gpu, grayscale= self.grayscale,
            numpy = self.numpy,flatten = self.flatten,
            to_tensor = self.to_tensor,
            augmentation_settings = None)#dont augment test dataset as a general rule
        return test_dataset
    
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
            label =  self.indexer.convert_from_int_to_label_bag_level(key)
            labels.extend([label] * len(val))
        return list(zip(paths,labels))
        

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset, 
            num_workers=self.num_workers,
            shuffle=self.shuffle_training,
            drop_last=False, # does not matter because we are always on a batch_size 1, controlling batch size is through acc_grad
            persistent_workers=self.persistent_workers,
            batch_size=1
            )

    def val_dataloader(self):
        data_temp =  DataLoader(
            self.val_dataset, 
            num_workers=0,
            shuffle=False,
            persistent_workers=False,
            batch_size=1
            )
        return data_temp

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset, 
            num_workers=0,
            persistent_workers=False,
            batch_size=1,
            shuffle= False
            )

    def predict_dataloader(self):
        return DataLoader(
            self.train_dataset, 
            num_workers=self.num_workers,
            drop_last=True,
            )
