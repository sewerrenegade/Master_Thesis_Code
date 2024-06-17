import typing
import pytorch_lightning as pl
from datasets.MNIST.MNIST_base import MNIST_MIL_base
import torchvision.transforms as transforms
import os
from sklearn.model_selection import KFold
from torch.utils.data import Dataset, random_split, Subset
from torch.utils.data.dataloader import DataLoader

class MNIST(pl.LightningDataModule):
    def __init__(
            self,
            synthesizer,
            data_path: str = "data/",
            num_workers: int = 4,
            img_dim: int = 28,
            val_split = 0.2,
            k_fold = -1
            ):
        super().__init__()
        self.synth_params = synthesizer
        self.data_dir = data_path
        self.img_dim = img_dim
        self.num_workers = num_workers
        self.val_split = val_split
        self.k_fold = k_fold
        self.transfroms = transforms.Compose([transforms.Grayscale(),transforms.Normalize((0.1307,), (0.3081,))])#transforms.ToTensor(),
        self.create_test_dataset()
        self.create_train_dataset()

    def set_up_kfold(self):
        if self.k_fold != -1:
            kf = KFold(n_splits=self.k_fold, shuffle=True)
            self.all_split =  [k for k in kf.split(self.train_dataloader)]

    def setup(self, stage=None):
        if self.k_fold == -1:
            if stage == 'fit':
                self.create_train_dataset()
            elif stage == 'test':
                self.create_test_dataset()
            

    def create_train_dataset(self):
        train_dataset = MNIST_MIL_base(data_synth=self.synth_params, root_dir=self.data_dir, transforms=self.transfroms,training= True)
        train_dataset_size = len(train_dataset)
        val_size = int(self.val_split * train_dataset_size)
        train_size = train_dataset_size - val_size
        self.train_dataset, self.val_dataset = random_split(train_dataset, [train_size, val_size])

    def create_test_dataset(self):
        self.test_dataset = MNIST_MIL_base(data_synth=self.synth_params,root_dir=self.data_dir, transforms=self.transfroms,training= False)


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
