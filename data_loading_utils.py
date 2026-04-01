# this file will contain data loading utilities such as loading h5 files, splitting data into train val and test sets, data augmentation, etc.

import os
import json
import h5py
import numpy as np
import pandas as pd
from config import *
import torch
from torch.utils.data import WeightedRandomSampler, Dataset, ConcatDataset, DataLoader
from pre_processing import PreProcessor

class DataLoadingUtils():

    def __init__(self):

        # data related params
        self.input_data_key = input_data_key
        self.label_data_key = label_data_key
        self.data_info_key = data_info_key
        self.preprocessed_data_path =  preprocessed_data_path

        # deep learning related params
        self.keep_keys = keep_keys
        self.seed = seed
        self.train_frac = train_frac
        self.val_frac = val_frac
        self.test_frac = test_frac


        # set the seeds for reproducibility
        self.set_reproducibility()

    def set_reproducibility(self):
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)

    def load_h5_dataset(self, file_path, to_debug=False):
        # this function will load the h5 file using the defined key parameters

        data = {}

        with h5py.File(file_path, "r") as file:
            for k in [self.input_data_key, self.label_data_key]:
                if k in file:

                    if to_debug:
                        data[k] = file[k][:1]
                    else:
                        data[k] = file[k][:]

            # loading info from h5 file is a little bit different because it is a "scalar" and reqiures [()]
            info_bytes = file[self.data_info_key][()]  # scalar from HDF5

            # Check type
            if isinstance(info_bytes, bytes):
                # Single bytes object
                info_list = [json.loads(info_bytes.decode('utf-8'))]

            elif isinstance(info_bytes, (np.ndarray, list)):
                # Array of bytes
                info_list = [json.loads(x.decode('utf-8')) for x in info_bytes]

            else:
                raise TypeError(f"Unexpected type {type(info_bytes)} for HDF5 info")

            # info is now inside a list with len=1 so need to use [0] to access it
            data[self.data_info_key] = info_list[0]

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
    
    def split_dataset(self, X, Y, info, to_debug=False):
        train_idxs, val_idxs, test_idxs = self.data_split_indices(len(X))

        all_case_keys = list(info.keys())

        # split keys by indices
        train_keys = [all_case_keys[i] for i in train_idxs]
        val_keys   = [all_case_keys[i] for i in val_idxs]
        test_keys  = [all_case_keys[i] for i in test_idxs]

        # create a list of cleaned info dicts for each split
        train_info = [{k: info[case_key][k] for k in self.keep_keys} for case_key in train_keys]
        val_info   = [{k: info[case_key][k] for k in self.keep_keys} for case_key in val_keys]
        test_info  = [{k: info[case_key][k] for k in self.keep_keys} for case_key in test_keys]

        if to_debug:
            print(f"{train_info=}, {val_info=}, {test_info=}")


        return X[train_idxs], Y[train_idxs], train_info, X[val_idxs], Y[val_idxs], val_info, X[test_idxs],  Y[test_idxs], test_info
    
    def load_all_data(self, to_debug=False):

        datasets = {}

        for file in os.listdir(self.preprocessed_data_path)[:]:

            if to_debug:
                if "sliver" not in file.lower():
                    continue

            if file.endswith(".h5"):
                name = file.replace(".h5", "")
                path = os.path.join(self.preprocessed_data_path, file)

                print(f"Loading {name}...")
                datasets[name] = self.load_h5_dataset(path)
            

        return datasets
    
    def split_all_data(self, dict_datasets, to_debug=False):

        #  this function unfortunately does the heavy lifting due to time constraints.
        #  it is splitting the data AND converting it into a pytorch Dataset object

        # define lists to store all the train, val and test datasets from 
        train_datasets = []
        val_datasets = []
        test_datasets = []

        # Iterate over each dataset
        for name, data in dict_datasets.items():

            # get the input and label arrays
            X = data[self.input_data_key]
            Y = data[self.label_data_key]
            info = data[self.data_info_key]

            # split each dataset so that each dataset can be in all stages
            # the returned split dataset is unordinary in terms of shape: its like a list of various datasets and each dataset component contains input array, label array = ds[0]
            X_train, Y_train, info_train, X_val, Y_val, info_val, X_test, Y_test, info_test = self.split_dataset(X, Y, info)
            
            # here the medical dataset class chunks and converts to tensor
            train_datasets.append(MedicalDataset(X_train, Y_train, info_train, name, mode="train"))
            val_datasets.append(MedicalDataset(X_val, Y_val, info_val, name, mode='val'))
            test_datasets.append(MedicalDataset(X_test, Y_test, info_test, name, mode='test'))

        if to_debug:
            for ds, orig_ds in zip(train_datasets, dict_datasets.values()):
                for idx, vol in enumerate(ds):
                    x, y, info_ind = vol

                    
                    print(f"Line 142 {ds.name}-{idx} train: len={len(ds)}, x={x.shape}, y={y.shape}")
                    print(f"{len(orig_ds[self.input_data_key])=}")

            for ds in val_datasets:
                x, y, info_ind = ds[0]
                print(f"{ds.name} val: len={len(ds)}, x={x.shape}, y={y.shape} (orig:{len(X_val)})")

            for ds in test_datasets:
                x, y, info_ind = ds[0]
                print(f"{ds.name} test: len={len(ds)}, x={x.shape}, y={y.shape} (orig:{len(X_test)})")

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
    def __init__(self, X, Y, info, name=None, transform=None, mode='train'):
        self.X = X
        self.Y = Y
        self.info = info
        self.name = name
        self.transform = transform
        self.chunk_x = chunk_x
        self.chunk_y = chunk_y
        self.chunk_z = chunk_z
        self.stride = stride
        self.mode = mode

        # get the chunk indices for validation and test
        if self.mode != "train":
            self.patch_indices = self.build_patch_index()

        # initiate class
        pre_processor_cls = PreProcessor()
        self.debugging_plotting = pre_processor_cls.save_plots_for_debugging

    def build_patch_index(self):
        """
        The input arrays were stored in h5 files as 1,Z,Y,X while the label arrays either had 2 or 3 classes and were stored as 2,Z,Y,X or 3,Z,Y,X.
        When defining the chunks, I need to ignore the channel dimension.
        
        The for loop will define how many total slides the original volume can be chunked into. 
        For example for volums with dimensions of 1,160,160,160, the channel part will be skipped. For the depth, there can be 3 chunks with stride of 96//4=24 whose starting indices are 0, 24, 64 and ending indices are 0+96,24+96,64+96.

        This function is run 3x because on lines 104-106, the class is called for each split type.

        This function and method will only be used for validation and test datasets. All of the patches for these two datasets will be used.

       """
        
        indices = []
        for i in range(len(self.X)):  # each volume
            SZ, SY, SX = self.X[i].shape[1:]  # skip Channel dimension

            for z in range(0, SZ - self.chunk_z + 1, self.stride): # start with z dimension because it comes after channel
                for y in range(0, SY - self.chunk_y + 1, self.stride): # continue with y dimension because it comes after z
                    for x in range(0, SX - self.chunk_x + 1, self.stride): # end with x dimension because it comes after y
                        indices.append((i, z, y, x))
        return indices

    def __len__(self):
        # This function calculates the total length of the dataset.
        # because my dataset is chunked, the len(self.X) does not represent the true dataset now.
        # after getting the chunking indices, the true data length is of that of chunk indices number
        if self.mode == "train" or self.mode == "val" or self.mode == "test":
            return len(self.X)
        elif self.mode == 'chunk':
            return len(self.patch_indices)
        else:
            print("Define the mode.")

    def __getitem__(self, idx, to_debug=False):
        if self.mode == "train":

            # only one patch will be saved for the train
            x = self.X[idx]
            y = self.Y[idx]
            info_indiv = self.info[idx]

            _, SZ, SY, SX = x.shape

            # center
            z_c, y_c, x_c = SZ//2, SY//2, SX//2

            # compute jitter range
            # make sure it doesnt go negative
            jitter_z = (SZ - self.chunk_z)//2 - 5
            jitter_y = (SY - self.chunk_y)//2 - 5
            jitter_x = (SX - self.chunk_x)//2 - 5

            # apply jitter
            z_c += np.random.randint(-jitter_z, jitter_z + 1)
            y_c += np.random.randint(-jitter_y, jitter_y + 1)
            x_c += np.random.randint(-jitter_x, jitter_x + 1)

            # convert center → start index
            z0 = np.clip(z_c - self.chunk_z//2, 0, SZ - self.chunk_z)
            y0 = np.clip(y_c - self.chunk_y//2, 0, SY - self.chunk_y)
            x0 = np.clip(x_c - self.chunk_x//2, 0, SX - self.chunk_x)

            patch_x = x[:, z0:z0+self.chunk_z, y0:y0+self.chunk_y, x0:x0+self.chunk_x]
            patch_y = y[:, z0:z0+self.chunk_z, y0:y0+self.chunk_y, x0:x0+self.chunk_x]

        elif self.mode == 'val' or self.mode == 'test':
            #  the validation and test datasets will not be processed here and will be saved as is. 
            #  in the validation and testing stages, each volume will get chunked, ran thru model, stitched back together
            patch_x = self.X[idx]
            patch_y = self.Y[idx]
            info_indiv = self.info[idx]

        # apply transforms if any (on the patch, not the full volume)
        if self.transform:
            patch_x, patch_y = self.transform(patch_x, patch_y)

        # debugging
        if to_debug:
            self.debugging_plotting(patch_x[0,:,:,:].transpose(2,1,0), patch_y[1,:,:,:].transpose(2,1,0), slice_number=37, processing_type="jitter", dataset_name=info_indiv['dataset'], case_number=info_indiv['case_number'])

        # convert to PyTorch tensors
        patch_x = torch.tensor(patch_x, dtype=torch.float32)
        patch_y = torch.tensor(patch_y, dtype=torch.float32)

        if patch_x is None or patch_y is None or info_indiv is None:
            print(f"Warning {idx} has None data: {patch_x}, {patch_y}, {info_indiv}")

            print(f"Returning shapes: {patch_x.shape if patch_x is not None else None}, "
                f"{patch_y.shape if patch_y is not None else None}, info={info_indiv}")
        
        return patch_x, patch_y, info_indiv

def get_dataloaders(to_debug=False):
    
    data_loading_utils = DataLoadingUtils()
    dict_datasets = data_loading_utils.load_all_data()

    # split the data
    train_datasets, val_datasets, test_datasets = data_loading_utils.split_all_data(dict_datasets)

    # ocncatenate the datasets
    # the provided datasets have the type of pytorch dataset 
    # to get the shape of the ConCat dataset type, do the following: train_combined.datasets[0].X.shape
    train_combined = data_loading_utils.concat_all_data(train_datasets)
    val_combined = data_loading_utils.concat_all_data(val_datasets)
    test_combined = data_loading_utils.concat_all_data(test_datasets)

    # the below print function may not print the processed shape but instead the original shape since get item is storing both possibly
    if to_debug:
        print(f"""
        {train_combined.datasets[0].X.shape=} and {train_combined.datasets[0].Y.shape=})
        {val_combined.datasets[0].X.shape=} and {val_combined.datasets[0].Y.shape=})
        {test_combined.datasets[0].X.shape=} and {test_combined.datasets[0].Y.shape=}""")

    # get the sampling weights
    weights = data_loading_utils.build_sampling_weights(train_combined, mode='root-proportional')

    # crate weighted random sampler
    sampler = WeightedRandomSampler(weights=weights, num_samples=len(weights), replacement=True)

    # create data loader
    train_loader = DataLoader(train_combined, batch_size=1, sampler=sampler)
    val_loader = DataLoader(val_combined, batch_size=1, shuffle=False)
    test_loader = DataLoader(test_combined, batch_size=1, shuffle=False)

    return train_loader, val_loader, test_loader

# if __name__ == "__main__":
#     data_loading_utils = DataLoadingUtils()
#     dict_datasets = data_loading_utils.load_all_data()

#     # split the data
#     train_datasets, val_datasets, test_datasets = data_loading_utils.split_all_data(dict_datasets)

#     # ocncatenate the datasets
#     # the provided datasets have the type of pytorch dataset 
#     # to get the shape of the ConCat dataset type, do the following: train_combined.datasets[0].X.shape
#     train_combined = data_loading_utils.concat_all_data(train_datasets)
#     val_combined = data_loading_utils.concat_all_data(val_datasets)
#     test_combined = data_loading_utils.concat_all_data(test_datasets)

#     # for debuggin
#     # sample = train_combined[0]

#     print(f"{train_combined.datasets[0].X.shape=} and {train_combined.datasets[0].Y.shape=}")
#     print(f"{val_combined.datasets[0].X.shape=} and {val_combined.datasets[0].Y.shape=}")
#     print(f"{test_combined.datasets[0].X.shape=} and {test_combined.datasets[0].Y.shape=}")

#     # get the sampling weights
#     weights = data_loading_utils.build_sampling_weights(train_combined, mode='root-proportional')

#     # crate weighted random sampler
#     sampler = WeightedRandomSampler(weights=weights, num_samples=len(weights), replacement=True)

#     # create data loader
#     train_loader = DataLoader(train_combined, batch_size=2, sampler=sampler)
#     val_loader = DataLoader(val_combined, batch_size=1, shuffle=False)
#     test_loader = DataLoader(test_combined, batch_size=1, shuffle=False)
    


    

        
