# Use the code to pre-process the data for training, validation, and testing.
# The output of the code will be a ready-to-use tensor datasets for pytorch.

from config import *
import SimpleITK as sitk
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

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
    
    def get_volume_array_stats(self, volume):

        # get the mean and standard deviation of the volume array > 0

        array_new = volume[volume > 0]
        mean = np.mean(array_new)
        std = np.std(array_new)

        return mean, std

    def resample_volume(self, input_volume, orig_spacing, desired_spacing, resampling_mode, to_print=True):
        # Resample the input volume to the desired spacing using simpleITK
        # Deep learning models work better using uniform spacing
        # it is important to set resampling_mode to correct function nearestneighbor for label volumes

        # define original size and new size
        orig_size = input_volume.GetSize()
        new_size = [ int(old_spacing / new_spacing * size) for old_spacing, new_spacing, size in zip(orig_spacing, desired_spacing, orig_size)]
        
        # pre-process the input volume
        new_metadata = sitk.Image(new_size, input_volume.GetPixelIDValue())
        new_metadata.SetSpacing(desired_spacing)
        new_metadata.SetOrigin(input_volume.GetOrigin())
        new_metadata.SetDirection(input_volume.GetDirection())

        # define resampler
        resampler = sitk.ResampleImageFilter()
        resampler.SetReferenceImage(new_metadata)
        resampler.SetInterpolator(resampling_mode)
        resampler.SetTransform(sitk.Transform())

        # resample the input volume
        resampled_input_volume = resampler.Execute(input_volume)

        # get array from the resampled input volume
        resampled_input_array = sitk.GetArrayFromImage(resampled_input_volume)

        # transpose to (x,y,z)
        resampled_input_array = np.transpose(resampled_input_array, (2,1,0))

        if to_print:    
            print(f"🔁 Original spacing: {orig_spacing}, New spacing: {desired_spacing}")
            print(f"🔁 Original size: {orig_size}, New size: {new_size}")
            print(f"🔁 Resampled input array shape: {resampled_input_array.shape}")

        return resampled_input_array

    def display_for_debugging(self, input_array, label_array, axes=['z'], num_slices=1, slice_number=None, processing_type="resampling"):

        print(f"🐞 Input shape: {input_array.shape}, label shape: {label_array.shape}")

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
                
                print(np.unique(label_slice))

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
            print(f"🌾 Cropped input array shape: {cropped_input_array.shape}, Cropped label array shape: {cropped_label_array.shape}")
        
            self.display_for_debugging(cropped_input_array, cropped_label_array, slice_number=slice_number, processing_type="cropping")

        return cropped_input_array, cropped_label_array


    def process_all_volumes(self, to_print=True):
        # Read the excel file containing the file paths
        df = pd.read_excel(self.excel_paths)

        # unique datasets in the excel file
        unique_datasets = df[self.dataset_name_col].unique()
        unique_datasets = unique_datasets[0:1] # only take the first dataset for now, for testing purposes

        # Iterate through each row in the dataframe and load the volumes
        for unique_dataset in unique_datasets:
            df_subset = df[df[self.dataset_name_col] == unique_dataset]
            df_subset = df_subset[0:1] # only take the first row of each dataset for now, for testing purposes
            object = df_subset[self.object_col].iloc[0]

            info_volumes = np.zeros((len(df_subset), 24)) # create an array to store the info of the volumes

            # only for sliver, use the sliver spacing and crop size, for all other datasets, use the same spacing and crop size defined in the config file
            if unique_dataset.lower() == 'sliver':
                desired_spacing = self.desired_spacing_dict['sliver']
                new_shape = self.crop_dict[object]
            
            else:
                desired_spacing = self.desired_spacing_dict[object]
                new_shape = self.crop_dict[object]

            for index, row in df_subset.iterrows():
                input_path = row[self.input_col]
                label_path = row[self.label_col]

                # 1️⃣ load the input and label volumes and arrays
                input_volume, input_array = self.load_volume(input_path)
                label_volume, label_array = self.load_volume(label_path)

                # get spacing and size
                input_spacing, input_size = self.get_volume_info(input_volume)
                label_spacing, label_size = self.get_volume_info(label_volume)

                # get the segmentation extent
                seg_extent = self.get_segmentation_extent(label_array)

                # get volume array stats
                input_mean, input_std = self.get_volume_array_stats(input_array)

                # 2️⃣ resample the input and label volumes to the desired spacing
                resampled_input_array = self.resample_volume(input_volume, input_spacing, desired_spacing, sitk.sitkBSpline)
                resampled_label_array = self.resample_volume(label_volume, label_spacing, desired_spacing, sitk.sitkNearestNeighbor)

                #  🐞 define a slice to debug
                old_z_slice = resampled_input_array.shape[2] // 2+10
                phys_z_slice = input_spacing[2] * old_z_slice
                new_z_slice = int(round(phys_z_slice / desired_spacing[2]))

                if to_print:
                    self.display_for_debugging(input_array, label_array, slice_number=old_z_slice, processing_type="")
                    self.display_for_debugging(resampled_input_array, resampled_label_array, slice_number=new_z_slice, processing_type="resampling")


                # get segmentation extent of the resampled label volume
                resampled_seg_extent = self.get_segmentation_extent(resampled_label_array)

                # 3️⃣crop the resampled input and label arrays to the desired crop size
                cropped_input, cropped_label = self.crop_volume(resampled_input_array, resampled_label_array, new_shape, slice_number=new_z_slice)

                # 4️⃣ normalize image


                # populate info_volumes array
                info_volumes[index, 0:3] = input_spacing
                info_volumes[index, 3:6] = input_size
                info_volumes[index, 6:12] = seg_extent
                info_volumes[index, 12:15] = np.array(input_spacing) * np.array(input_size)
                info_volumes[index, 15:18]= np.array(input_spacing) * (np.array(seg_extent[1::2])- np.array(seg_extent[0::2]))
                info_volumes[index,18:21]= (np.array(resampled_seg_extent[1::2])- np.array(resampled_seg_extent[0::2]))


if __name__ == "__main__":
    pre_processor = PreProcessor()
    pre_processor.process_all_volumes()