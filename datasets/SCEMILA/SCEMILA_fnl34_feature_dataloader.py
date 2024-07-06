from datasets.base_dataset_abstraction import BaseDataset
import numpy as np
import os
import torch

class SCEMILAfeature_MIL_base(BaseDataset):

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
    