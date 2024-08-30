import pytorch_lightning as pl
from sklearn.model_selection import KFold
from torch.utils.data import random_split
from torch.utils.data.dataloader import DataLoader
from datasets.SCEMILA.base_image_SCEMILA import SCEMILA_MIL_base
from datasets.image_augmentor import AugmentationSettings
from torch.utils.data import Subset
from datasets.indexer_scripts.indexer_utils import get_dataset_indexer
from sklearn.model_selection import train_test_split

class SCEMILA(pl.LightningDataModule):
    def __init__(
            self,
            input_type = "fnl34",#images      
            num_workers: int = 1,
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
        if self.k_fold > -1:
            if stage == 'fit':
                train_indexes, test_indexes = self.all_splits[self.fold]
                self.train_dataset = Subset(self.full_dataset, train_indexes)
                self.test_dataset = Subset(self.full_dataset, test_indexes)
            elif stage == 'test':
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
            train_dataset_size = len(train_dataset)
            val_size = int(self.val_split * train_dataset_size)
            train_size = train_dataset_size - val_size
            
            targets = train_dataset.get_targets()
            train_idx, test_idx = train_test_split(
                range(len(targets)), 
                test_size=val_size, 
                stratify=targets, 
                random_state=42
            )
            
            self.train_dataset = Subset(train_dataset, train_idx)
            self.val_dataset = Subset(train_dataset, test_idx)
        elif self.k_fold > 1:
            self.fold = 0
            val_percentage = 0.05
            dataset_size = len(train_dataset)
            val_size = int(val_percentage * dataset_size)
            train_size = dataset_size - val_size
            self.full_dataset,self.val_dataset = random_split(train_dataset, [train_size, val_size])
            kf = KFold(n_splits=self.k_fold, shuffle=True)
            self.all_splits = [k for k in kf.split(self.full_dataset)]
            
    def create_test_dataset(self):
        if self.k_fold <= 1:
            test_dataset = SCEMILA_MIL_base(training_mode= False,input_type=self.input_type,
                encode_with_dino_bloom = self.encode_with_dino_bloom,
                balance_dataset_classes = self.balance_dataset_classes,
                gpu = self.gpu, grayscale= self.grayscale,
                numpy = self.numpy,flatten = self.flatten,
                to_tensor = self.to_tensor,
                augmentation_settings = None)#dont augment test dataset as a general rule
        else:
            test_dataset = None # to be filled by kfold validation
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
            drop_last=True,
            persistent_workers=self.persistent_workers,
            batch_size=1
            )

    def val_dataloader(self):
        data_temp =  DataLoader(
            self.val_dataset, 
            num_workers=self.num_workers,
            shuffle=False,
            drop_last=True,
            persistent_workers=self.persistent_workers,
            batch_size=1
            )
        return data_temp

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset, 
            num_workers=self.num_workers,
            drop_last=True,
            persistent_workers=self.persistent_workers,
            batch_size=1
            )

    def predict_dataloader(self):
        return DataLoader(
            self.train_dataset, 
            num_workers=self.num_workers,
            drop_last=True,
            )
