# this file will contain data loading utilities such as loading h5 files, splitting data into train val and test sets, data augmentation, etc.

import os
import h5py
import numpy as np
import pandas as pd
from config import *
import torch
from torch.utils.data import WeightedRandomSampler, Dataset, ConcatDataset, DataLoader

class DataLoadingUtils():

    def __init__(self):

        # data related params
        self.input_data_key = input_data_key
        self.label_data_key = label_data_key
        self.data_info_key = data_info_key
        self.preprocessed_data_path =  preprocessed_data_path

        # deep learning related params
        self.seed = seed
        self.train_frac = train_frac
        self.val_frac = val_frac
        self.test_frac = test_frac


        # set the seeds for reproducibility
        self.set_reproducibility()

    def set_reproducibility(self):
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)

    def load_h5_dataset(self, file_path):
        # this function will load the h5 file using the defined key parameters

        data = {}

        with h5py.File(file_path, "r") as file:
            for k in [self.input_data_key, self.label_data_key]:
                if k in file:
                    data[k] = file[k][:]

            # loading info from h5 file is a little bit different because it is a "scalar" and reqiures [()]
            data[self.data_info_key] = file[self.data_info_key][()]
        return data
    
    def data_split_indices(self, n):
        # this function will provide a list of indices for each dataset to be used when splitting the individual h5 input and label datasets

        # define indices list and shuffle it
        idx_list = np.arange(n)
        np.random.shuffle(idx_list)  # shuffle so that as much of the data can be seen in all datasets

        # define total number of entries for each dataset
        n_test = int(self.test_frac * n)
        n_val = int(self.val_frac * n)

        # define indices list for each dataset
        test_idx = idx_list[:n_test]
        valid_idx = idx_list[n_test:n_test + n_val]
        train_idx = idx_list[n_test + n_val:]

        return train_idx, valid_idx, test_idx
    
    def split_dataset(self, X, Y):
        train_idxs, valid_idxs, test_idxs = self.data_split_indices(len(X))

        print(f"Successfully split data!")

        return X[train_idxs], Y[train_idxs], X[valid_idxs], Y[valid_idxs], X[test_idxs],  Y[test_idxs]
    
    def load_all_data(self):

        datasets = {}

        for file in os.listdir(self.preprocessed_data_path)[1:4]:

            if file.endswith(".h5"):
                name = file.replace(".h5", "")
                path = os.path.join(self.preprocessed_data_path, file)

                print(f"Loading {name}...")
                datasets[name] = self.load_h5_dataset(path)

        return datasets
    
    def split_all_data(self, dict_datasets, to_debug=True):

        #  this function unfortunately does the heavy lifting due to time constraints.
        #  it is splitting the data AND converting it into a pytorch Dataset object

        # define lists to store all the train, val and test datasets from 
        train_datasets = []
        val_datasets = []
        test_datasets = []

        for name, data in dict_datasets.items():
            X = data[self.input_data_key]
            Y = data[self.label_data_key]

            X_train, Y_train, X_val, Y_val, X_test, Y_test = self.split_dataset(X, Y)
            train_datasets.append(MedicalDataset(X_train, Y_train, name))
            val_datasets.append(MedicalDataset(X_val, Y_val, name))
            test_datasets.append(MedicalDataset(X_test, Y_test, name))

        if to_debug:
            for ds in train_datasets:
                x, y = ds[0]
                print(f"{ds.name}: len={len(ds)}, x={x.shape}, y={y.shape}")

        return train_datasets, val_datasets, test_datasets

    def concat_all_data(self, dataset):
        #  this function just concatenates a list containing all the data
        #  this required pytorch Dataset object

        return ConcatDataset(dataset)
    
    def build_sampling_weights(self, dataset, mode='root-proportional'):
        """
        Build per-sample weights for a concatenated dataset.
        
        Args:
            concat_dataset: ConcatDataset
            mode: how to weight datasets. Options:
                - 'proportional': equal weighting
                - 'root-proportional': weight inversely to sqrt(n_samples)
                - 'log-proportional': weight inversely to log(n_samples)
        Returns:
            torch.DoubleTensor of weights
        """

        weights = []

        # concat_dataset.datasets is the list of original datasets
        for ds in dataset.datasets:
            n_samples = len(ds)
            
            if mode == 'proportional':
                w = 1.0 / n_samples
            elif mode == 'root-proportional':
                w = 1.0 / np.sqrt(n_samples)
            elif mode == 'log-proportional':
                w = 1.0 / np.log(n_samples)
            else:
                raise ValueError(f"Unknown mode: {mode}")

            # assign this weight to all samples in the sub-dataset
            weights.extend([w] * n_samples)

        # convert to tensor
        return torch.DoubleTensor(weights)

class MedicalDataset(Dataset):
    def __init__(self, X, Y, name=None, transform=None):
        self.X = X
        self.Y = Y
        self.name = name
        self.transform = transform

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx]
        y = self.Y[idx]

        if self.transform:
            x, y = self.transform(x, y)

        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

    

if __name__ == "__main__":
    data_loading_utils = DataLoadingUtils()
    dict_datasets = data_loading_utils.load_all_data()

    # split the data
    train_datasets, val_datasets, test_datasets = data_loading_utils.split_all_data(dict_datasets)

    # ocncatenate the datasets
    train_combined = data_loading_utils.concat_all_data(train_datasets)
    val_combined = data_loading_utils.concat_all_data(val_datasets)
    test_combined = data_loading_utils.concat_all_data(test_datasets)

    # get the sampling weights
    weights = data_loading_utils.build_sampling_weights(train_combined, mode='root-proportional')

    # crate weighted random sampler
    sampler = WeightedRandomSampler(weights=weights, num_samples=len(weights), replacement=True)

    # create data loader
    train_loader = DataLoader(train_combined, batch_size=2, sampler=sampler)
    
    # x, y = train_combined[0]
    # print(f"len={len(train_combined)}, x={x.shape}, y={y.shape}")
    # print(train_combined.datasets)
    # # print(weights)
    # print(len(weights))


    

        
