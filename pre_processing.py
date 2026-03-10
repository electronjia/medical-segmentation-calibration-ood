# Use the code to pre-process the data for training, validation, and testing.
# The output of the code will be a ready-to-use tensor datasets for pytorch.

from config import *
import SimpleITK as sitk
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import h5py
from tqdm import tqdm 

class PreProcessor:
    def __init__(self):
        
        # excel filepath and column names
        self.excel_paths = excel_paths
        self.dataset_name_col = dataset_name_col
        self.case_col = case_col
        self.object_col = object_col
        self.input_col = input_col
        self.label_col = label_col

        # pre-processing parameters
        self.desired_spacing_dict = desired_spacing_dict
        self.crop_dict = crop_dict
        self.n_channel_dict = n_channel_dict
        self.n_class_dict = n_class_dict

    def get_segmentation_extent(self, label_array):
        # get the extent of the segmentation in the label array and return a tuple

        # find the non-zero indices in the label array
        non_zero_label = np.where(label_array > 0)

        # get the min and max indices in each dimension
        seg_extent =(
            non_zero_label[0].min(), non_zero_label[0].max(),
            non_zero_label[1].min(), non_zero_label[1].max(),
            non_zero_label[2].min(), non_zero_label[2].max()
        )

        return seg_extent

    def get_volume_info(self, volume):
        # get the spacing, size, and origin of the volume
        spacing = volume.GetSpacing()
        size = volume.GetSize()

        return spacing, size

    def load_volume(self, volume_path):
        
        # Load the volume using simpleITK
        # simpleITK loads volumes as (x,y,z) where x=width, y=height, z=depth
        volume = sitk.ReadImage(volume_path)

        # the dimensions change to (z,y,x) where z=depth, y=height, x=width
        array = sitk.GetArrayFromImage(volume)

        # transpose to (x,y,z) 
        array_np = np.transpose(array, (2,1,0))

        return volume, array_np
    
    def get_pos_volume_array_stats(self, volume):

        # get the mean and standard deviation of the positive volume array > 0

        array_new = volume[volume > 0]
        mean = np.mean(array_new)
        std = np.std(array_new)

        return mean, std

    def resample_volume(self, volume, orig_spacing, desired_spacing, resampling_mode):
        # Resample the provided volume to the desired spacing using simpleITK
        # Deep learning models work better using uniform spacing
        # it is important to set resampling_mode to correct function nearestneighbor for label volumes

        # define original size and new size
        orig_size = volume.GetSize()
        new_size = [ int(old_spacing / new_spacing * size) for old_spacing, new_spacing, size in zip(orig_spacing, desired_spacing, orig_size)]
        
        # pre-process the input volume
        new_metadata = sitk.Image(new_size, volume.GetPixelIDValue())
        new_metadata.SetSpacing(desired_spacing)
        new_metadata.SetOrigin(volume.GetOrigin())
        new_metadata.SetDirection(volume.GetDirection())

        # define resampler
        resampler = sitk.ResampleImageFilter()
        resampler.SetReferenceImage(new_metadata)
        resampler.SetInterpolator(resampling_mode)
        resampler.SetTransform(sitk.Transform())

        # resample the input volume
        resampled_volume = resampler.Execute(volume)

        # get array from the resampled input volume
        resampled_array = sitk.GetArrayFromImage(resampled_volume)

        # transpose to (x,y,z)
        resampled_array = np.transpose(resampled_array, (2,1,0))

        return resampled_volume, resampled_array

    def final_channel_dimension(self, array, transpose_function):
        # the final channel dimension should be z,y,x to fit with the pytorch
        return transpose_function(array)
    
    def display_for_debugging(self, input_array, label_array, axes=['z'], num_slices=1, slice_number=None, processing_type="resampling"):
        
        # index for dimension
        idx_dict = {'x':0, 'y':1, 'z':2}

        for axis in axes:
            # select axis index
            shape = input_array.shape
            for i in range(num_slices):
                # pick slice indices evenly
                if axis in idx_dict:
                    idx = shape[idx_dict[axis]] * (i+1) // (num_slices+1) if slice_number is None else slice_number
                    input_slice = input_array[:,:,idx]
                    label_slice = label_array[:,:,idx]
                else:
                    raise ValueError("Axis must be 'x', 'y', or 'z'.")

                plt.imsave(f"debugging/input_{processing_type}_{axis}_{idx}.png", input_slice, cmap='gray')
                plt.imsave(f"debugging/label_{processing_type}_{axis}_{idx}.png", label_slice, cmap='jet')

    def crop_volume(self, input_array, label_array, new_shape, slice_number=None, to_print=True):

        old_shape = input_array.shape

        # find segmentation extend
        seg_extent = self.get_segmentation_extent(label_array)
        x_min, x_max = seg_extent[0], seg_extent[1]
        y_min, y_max = seg_extent[2], seg_extent[3]
        z_min, z_max = seg_extent[4], seg_extent[5]

        mins = [x_min, y_min, z_min]
        maxs = [x_max, y_max, z_max]

        # initialize slices for cropping
        slices = []

        for idx, (min_idx, max_idx, oldie_shape, desired_shape) in enumerate(zip(mins, maxs, old_shape, new_shape)):

                center = (min_idx + max_idx) // 2
                start = max(0, center - desired_shape // 2)
                end = start + desired_shape

                # if crop exceeds dimension, shift
                if end > oldie_shape:
                    start = max(0, oldie_shape - desired_shape)
                    end = start + desired_shape
                slices.append(slice(start, end))

        # crop the input and label arrays
        cropped_input_array = input_array[slices[0], slices[1], slices[2]]
        cropped_label_array = label_array[slices[0], slices[1], slices[2]]

        # pad if needed
        pad_width = [(0, max(0, desired - cropped)) for desired, cropped in zip(new_shape, cropped_input_array.shape)]
        if any(pad > 0 for _, pad in pad_width):
            cropped_input_array = np.pad(cropped_input_array, pad_width, mode='constant', constant_values=0)
            cropped_label_array = np.pad(cropped_label_array, pad_width, mode='constant', constant_values=0)

        # debugging
        if to_print:
            self.display_for_debugging(cropped_input_array, cropped_label_array, slice_number=slice_number, processing_type="cropping")

        return cropped_input_array, cropped_label_array

    def process_all_volumes(self, to_print=True, to_debug=False):
        # Read the excel file containing the file paths
        df = pd.read_excel(self.excel_paths)

        # unique datasets in the excel file
        unique_datasets = df[self.dataset_name_col].unique()

        # 🐞 for debugging
        if to_debug:
            unique_datasets = unique_datasets[0:1] # only take the first dataset for now, for testing purposes

        # Iterate through each row in the dataframe and load the volumes
        for unique_dataset in unique_datasets:
            df_subset = df[df[self.dataset_name_col] == unique_dataset]
            object = df_subset[self.object_col].iloc[0].lower() # get the object name for the dataset, and make it lowercase for consistency
            
            # 🐞 for debugging
            if to_debug:
                df_subset = df_subset[0:1] # only take the first row of each dataset for now, for testing purposes
                    
            # get the pre-processing parameters for the dataset
            desired_spacing = self.desired_spacing_dict[object]
            n_class = self.n_class_dict[object]
            new_shape = self.crop_dict[object]
            n_channel = self.n_channel_dict[object]
            N = len(df_subset)

            # only for sliver, use the object=sliver spacing and number of classes
            if unique_dataset.lower() == 'sliver':
                desired_spacing = self.desired_spacing_dict['sliver']
                n_class = self.n_class_dict['sliver']


            # arrays to keep track of stuff
            volume_info_dict = {} # create an array to store the info of the volumes
            X_vol = np.zeros((N, n_channel, *new_shape), np.float32) # create an array to store the input volumes
            Y_vol = np.zeros((N, n_class, *new_shape), np.float32) # create an array to store the label volumes
            # the above Y_vol does not assume one-hot encoding, however this may change if the loss function required one hot encoding

            for index, row in tqdm(df_subset.iterrows(), total=len(df_subset), desc=f"Processing {unique_dataset}"):
                input_path = row[self.input_col]
                label_path = row[self.label_col]
                case_number = row[self.case_col]

                # 1️⃣ load the input and label volumes and arrays
                input_volume, input_array = self.load_volume(input_path)
                label_volume, label_array = self.load_volume(label_path)

                # get spacing and size
                input_spacing, input_size = self.get_volume_info(input_volume)
                label_spacing, label_size = self.get_volume_info(label_volume)

                # get the segmentation extent
                seg_extent = self.get_segmentation_extent(label_array)

                # get volume array stats
                input_mean, input_std = self.get_pos_volume_array_stats(input_array)

                # 2️⃣ resample the input and label volumes to the desired spacing
                resampled_input_volume, resampled_input_array = self.resample_volume(input_volume, input_spacing, desired_spacing, sitk.sitkBSpline)
                resampled_label_volume, resampled_label_array = self.resample_volume(label_volume, label_spacing, desired_spacing, sitk.sitkNearestNeighbor)

                resampled_input_spacing, resampled_input_size = self.get_volume_info(resampled_input_volume) # get spacing and size of the resampled input volume
                resampled_label_spacing, resampled_label_size = self.get_volume_info(resampled_label_volume) # get spacing and size of the resampled label volume
                resampled_seg_extent = self.get_segmentation_extent(resampled_label_array) # get segmentation extent of the resampled label volume

                #  🐞 define a slice to debug            
                old_z_slice = resampled_input_array.shape[2] // 2+10
                phys_z_slice = input_spacing[2] * old_z_slice
                new_z_slice = int(round(phys_z_slice / desired_spacing[2]))

                if to_print:
                    self.display_for_debugging(input_array, label_array, slice_number=old_z_slice, processing_type="")
                    self.display_for_debugging(resampled_input_array, resampled_label_array, slice_number=new_z_slice, processing_type="resampling")


                # 3️⃣crop and pad the resampled input and label arrays to the desired crop size
                cropped_input, cropped_label = self.crop_volume(resampled_input_array, resampled_label_array, new_shape, slice_number=new_z_slice, to_print=to_print)

                cropped_seg_extent = self.get_segmentation_extent(cropped_label) # get segmentation extent of the cropped label volume

                # 4️⃣ normalize image                
                cropped_input[cropped_input < 0] = 0 # set the negative volume to 0
                cropped_input = cropped_input / (input_std + 1e-8)

                # 5️⃣ final channel dimension and add channel dimension for pytorch
                final_input = self.final_channel_dimension(cropped_input, lambda x: np.transpose(x, (2,1,0))) # transpose to ( z, y, x)
                final_input = final_input[np.newaxis, ...] # add channel dimension to the input array, so the shape becomes (1, z, y, x) for pytorch
                final_label = self.final_channel_dimension(cropped_label, lambda x: np.transpose(x, (2,1,0))) # transpose to ( z, y, x)

                # populate X_vol and Y_vol arrays
                X_vol[index, 0, ...] = final_input
                # one hot encode labels
                for indiv_class in range(n_class):
                    Y_vol[index, indiv_class, ...] = (final_label == indiv_class).astype(np.float32)

                # populate info_volumes array
                volume_info_dict[f'{unique_dataset.lower()}_{object.lower()}_case{case_number}'] = {
                    'dataset': unique_dataset.lower(),
                    'object': object.lower(),
                    'case_number': case_number,
                    'input_path': input_path,
                    'label_path': label_path,

                    'n_channel': n_channel,
                    'n_class': n_class,
                    'desired_spacing': desired_spacing,
                    'new_shape': new_shape,

                    'input_spacing': input_spacing,
                    'input_size': input_size,
                    'input_physical_size': np.array(input_spacing) * np.array(input_size),
                    'label_spacing': label_spacing,
                    'label_size': label_size,
                    'label_extent': seg_extent,
                    'label_physical_size': np.array(input_spacing) * (np.array(seg_extent[1::2])- np.array(seg_extent[0::2])),
                    'label_bounding_box_size': (np.array(seg_extent[1::2])- np.array(seg_extent[0::2])),

                    'resampeled_input_spacing': resampled_input_spacing,
                    'resampled_input_size': resampled_input_size,
                    'resampled_input_physical_size': np.array(resampled_input_spacing) * np.array(resampled_input_size),
                    'resampled_label_spacing': resampled_label_spacing,
                    'resampled_label_size': resampled_label_size,
                    'resampled_label_extent': resampled_seg_extent,
                    'resampled_label_physical_size': np.array(resampled_input_spacing) * (np.array(resampled_seg_extent[1::2])- np.array(resampled_seg_extent[0::2])),
                    'resampled_label_bounding_box_size': (np.array(resampled_seg_extent[1::2])- np.array(resampled_seg_extent[0::2])),

                    'cropped_input_shape': cropped_input.shape,
                    'cropped_label_shape': cropped_label.shape,
                    'cropped_label_extent': cropped_seg_extent,
                    'cropped_label_bounding_box_size': (np.array(cropped_seg_extent[1::2])- np.array(cropped_seg_extent[0::2])),

                    'final_input_shape': X_vol[index].shape,
                    'final_label_shape': Y_vol[index].shape,
                }

                if to_print:
                    print(f"\n\n 🐞 Volume info for {unique_dataset} case {case_number}:")
                    for key, value in volume_info_dict[f'{unique_dataset.lower()}_{object.lower()}_case{case_number}'].items():
                        print(f"   {key}: {value}")
                    print("\n\n")



            # save to hpf5 file
            save_path = f"preprocessed_data/{unique_dataset.lower()}_{object.lower()}.h5"
            with h5py.File(save_path, "w") as f:
                f.create_dataset("X", data=X_vol, compression="gzip")
                f.create_dataset("Y", data=Y_vol, compression="gzip")
                f.create_dataset("info", data=np.string_(str(volume_info_dict)), compression="gzip")

            if to_print:
                print(f"✅ Saved {N} cases of {unique_dataset} ({object_name}) to {save_path}")


if __name__ == "__main__":
    pre_processor = PreProcessor()
    pre_processor.process_all_volumes()