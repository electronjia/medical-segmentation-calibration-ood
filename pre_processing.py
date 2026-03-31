# Use the code to pre-process the data for training, validation, and testing.
# The output of the code will be a ready-to-use tensor datasets for pytorch.

from config import *
import SimpleITK as sitk
import pandas as pd
import numpy as np
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import h5py
from tqdm import tqdm 
import matplotlib.image as mpimg
import cv2
import json

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

    def resample_volume(self, input_volume, label_volume, input_orig_spacing, label_orig_spacing, desired_spacing, input_resampling_mode, label_resamping_mode):
        # Resample the provided volume to the desired spacing using simpleITK
        # Deep learning models work better using uniform spacing
        # it is important to set resampling_mode to correct function nearestneighbor for label volumes
        # It was found that some of the input and label data pairs may have mismatched spacing and this required using the same image grid. usually this image grid comes from the input array and is used for the label resampling.

        # resample the input array first______________________________________________________________________________________________________
        # define original size and new size. use round() to avoid flooring the values which caused the error with dimension mismatch
        input_orig_size = input_volume.GetSize()
        new_size = [ round(int(old_spacing / new_spacing * size)) for old_spacing, new_spacing, size in zip(input_orig_spacing, desired_spacing, input_orig_size)]
        
        # pre-process the input volume
        new_metadata = sitk.Image(new_size, input_volume.GetPixelIDValue())
        new_metadata.SetSpacing(desired_spacing)  # copyinformation() function does not work
        new_metadata.SetOrigin(input_volume.GetOrigin())
        new_metadata.SetDirection(input_volume.GetDirection())
        new_metadata.SetSpacing(desired_spacing)

        # define resampler
        resampler = sitk.ResampleImageFilter()
        resampler.SetReferenceImage(new_metadata)
        resampler.SetInterpolator(input_resampling_mode)
        resampler.SetTransform(sitk.Transform())

        # resample the input volume
        resampled_input_volume = resampler.Execute(input_volume)

        # get array from the resampled input volume
        resampled_input_array = sitk.GetArrayFromImage(resampled_input_volume)

        # transpose to (x,y,z)
        resampled_input_array = np.transpose(resampled_input_array, (2,1,0))

        # force label to use th input image geometry before resampling___________________________________________________________________________________________
        label_volume.SetOrigin(input_volume.GetOrigin())
        label_volume.SetSpacing(input_volume.GetSpacing())
        label_volume.SetDirection(input_volume.GetDirection())

        # resample the label next using the image grid from input array
        resampler.SetReferenceImage(resampled_input_volume)

        # resample the label volume (xyz) using the proper resampling mode for the label
        resampler.SetInterpolator(label_resamping_mode)
        resampled_label_volume = resampler.Execute(label_volume)

        # get array from resampled label volume -> zyx
        resampled_label_array = sitk.GetArrayFromImage(resampled_label_volume)
        
        # transpose back ->  xyz
        resampled_label_array = np.transpose(resampled_label_array, (2,1,0))

        assert resampled_input_array.shape == resampled_label_array.shape , f'The resampled input and label shape do not match: {resampled_input_array.shape=} and {resampled_label_array.shape}'
        assert resampled_input_volume.GetSpacing() == resampled_label_volume.GetSpacing(), f"The resampled input and label spacing do not match: {resampled_input_volume.GetSpacing()=} and {resampled_label_volume.GetSpacing()=}"

        return resampled_input_volume, resampled_input_array, resampled_label_volume, resampled_label_array

    def final_channel_dimension(self, array, transpose_function):
        # the final channel dimension should be z,y,x to fit with the pytorch
        return transpose_function(array)
    
    def save_overlay_for_debugging(self, input_path, label_path):
        # load the newly saved input and label arrays using matplotlib and create an overlay figure

        # laod images as RGBA (RGB+alpha)
        input_img = mpimg.imread(input_path)
        label_img = mpimg.imread(label_path)

        # convert input RGBA → grayscale
        if input_img.ndim == 3:
            input_img = input_img[..., :3].mean(axis=2)

        # convert label RGBA → mask
        # if label was saved as RGBA (using cmap='jet), convert to a grayscale mask
        if label_img.ndim == 3:
            label_mask = label_img[..., :3].mean(axis=2)
        else:
            label_mask = label_img

        overlay_path = "_".join(["debugging/overlay"] + os.path.basename(input_path).replace("_input", "").split("_"))

        # normalize input to 0-255
        input_img = (input_img * 255).astype(np.uint8)

        # convert grayscale → BGR
        input_img = cv2.cvtColor(input_img, cv2.COLOR_GRAY2BGR)

        # create colored mask
        mask = np.zeros_like(input_img)
        mask[label_mask > 0] = [0,0,255]   # red mask

        # blend
        overlay = cv2.addWeighted(input_img, 1.0, mask, 0.5, 0)

        cv2.imwrite(overlay_path, overlay)

    def save_plots_for_debugging(self, input_array, label_array, axis='z', slice_number=None, processing_type="resampling", dataset_name=None, case_number=None):
        # saves a single slice from z axis in the middle

        # get middle z slice to save
        shape = input_array.shape
        idx = shape[2]  // 2 if slice_number is None else slice_number
        input_slice = input_array[:,:,idx]
        label_slice = label_array[:,:,idx]
        
        input_path = f"debugging/{dataset_name}_case{case_number}_{processing_type}_input_{axis}_{idx}.png"
        label_path = f"debugging/{dataset_name}_case{case_number}_{processing_type}_label_{axis}_{idx}.png"

        plt.imsave(input_path, input_slice, cmap='gray')
        plt.imsave(label_path, label_slice, cmap='gray')
        plt.close("all")

        self.save_overlay_for_debugging(input_path=input_path, label_path=label_path)

        return input_path, label_path

    def crop_volume(self, input_array, label_array, new_shape):

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

                # tried to get the center is the old shape > desired shape
                center = (min_idx + max_idx) // 2
                start = max(0, center - desired_shape // 2)
                end = start + desired_shape

                # gets the top-left part is the old shape < desired shape, then pads
                if end > oldie_shape:
                    start = max(0, oldie_shape - desired_shape)
                    end = start + desired_shape
                slices.append(slice(start, end))

        # crop the input and label arrays
        cropped_input_array = input_array[slices[0], slices[1], slices[2]]
        cropped_label_array = label_array[slices[0], slices[1], slices[2]]

        # pad if needed
        # print(f"{cropped_input_array.shape=}, {cropped_label_array.shape=}") # if troubleshooting the sizes
        pad_width = []

        for desired, cropped in zip(new_shape, cropped_input_array.shape):
            diff = max(0, desired - cropped)
            pad_before = diff // 2
            pad_after = diff - pad_before
            pad_width.append((pad_before, pad_after))

        cropped_input_array = np.pad(cropped_input_array, pad_width, mode='constant')
        cropped_label_array = np.pad(cropped_label_array, pad_width, mode='constant')

        assert cropped_input_array.shape == new_shape and cropped_label_array.shape == new_shape

        return cropped_input_array, cropped_label_array, slices, pad_width

    def convert_for_json(self, obj):

        if isinstance(obj, np.integer):
            return int(obj)

        if isinstance(obj, np.floating):
            return float(obj)

        if isinstance(obj, np.ndarray):
            return obj.tolist()

        if isinstance(obj, tuple):
            return list(obj)

        if isinstance(obj, slice):
            return {"start": obj.start, "stop": obj.stop, "step": obj.step}

        if isinstance(obj, list):
            return [self.convert_for_json(i) for i in obj]

        if isinstance(obj, dict):
            return {k: self.convert_for_json(v) for k, v in obj.items()}

        return obj

    def process_all_volumes(self, to_print=True, to_debug=True):
        # Read the excel file containing the file paths
        df = pd.read_excel(self.excel_paths)

        # unique datasets in the excel file
        unique_datasets = df[self.dataset_name_col].unique()

        if to_debug:
            unique_datasets = unique_datasets[:1]
        # task 3 liver did not work out due to data issues. task 5 spleen [index 8] did not work due to

        # Iterate through each row in the dataframe and load the volumes
        for unique_dataset in unique_datasets:
            df_subset = df[df[self.dataset_name_col] == unique_dataset]
            df_subset = df_subset.reset_index(drop=True)

            if to_debug:
                df_subset = df_subset[:1]
            
            object = df_subset[self.object_col].iloc[0].lower() # get the object name for the dataset, and make it lowercase for consistency
               
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
            # when defining the shape, i am following pytorch's suggestions on having zyx format
            volume_info_dict = {} # create an array to store the info of the volumes
            X_vol = np.zeros((N, n_channel, new_shape[2], new_shape[1], new_shape[0]), np.float32) # create an array to store the input volumes
            Y_vol = np.zeros((N, n_class, new_shape[2], new_shape[1], new_shape[0]), np.float32) # create an array to store the label volumes
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
                resampled_input_volume, resampled_input_array, resampled_label_volume, resampled_label_array  = self.resample_volume(input_volume, label_volume, input_spacing, label_spacing, desired_spacing, sitk.sitkBSpline,sitk.sitkNearestNeighbor )
                resampled_input_spacing, resampled_input_size = self.get_volume_info(resampled_input_volume) # get spacing and size of the resampled input volume
                resampled_label_spacing, resampled_label_size = self.get_volume_info(resampled_label_volume) # get spacing and size of the resampled label volume
                resampled_seg_extent = self.get_segmentation_extent(resampled_label_array) # get segmentation extent of the resampled label volume

                # make sure the correct number of classes exist in resampled label array
                assert len(np.unique(label_array)) == len(np.unique(resampled_label_array)), f"The unique values in original and resampled label arrays do not match: {len(np.unique(label_array))} and {len(np.unique(resampled_label_array))}"

                # 3️⃣crop and pad the resampled input and label arrays to the desired crop size
                cropped_input_array, cropped_label_array, slices, pad_width = self.crop_volume(resampled_input_array, resampled_label_array, new_shape)

                cropped_seg_extent = self.get_segmentation_extent(cropped_label_array) # get segmentation extent of the cropped label volume

                # 🐞 define a slice to debug

                # center of crop in resampled volume
                resampled_z_slice = resampled_input_array.shape[2] // 2

                # convert to cropped coordinates
                cropped_z_slice = cropped_input_array.shape[2] // 2

                # convert to original coordinates
                phys_slice = resampled_z_slice * desired_spacing[2]
                orig_z_slice = int(round(phys_slice / input_spacing[2]))

                # 🐞 to debug the z slice definition for orig input, resampled input, cropped input
                if to_debug:               
                    print(f"{slices=}")
                    print(f"{input_array.shape=}, {resampled_input_array.shape=}, {cropped_input_array.shape=}")
                    print(f"{label_array.shape=}, {resampled_label_array.shape=}, {cropped_label_array.shape=}")
                    print(f"{orig_z_slice=}, {resampled_z_slice=}, {cropped_z_slice=}")
                
                # 4️⃣ normalize image                
                cropped_input_array[cropped_input_array < 0] = 0 # set the negative volume to 0
                cropped_input_array = cropped_input_array / (input_std + 1e-8)

                if to_print:
                    self.save_plots_for_debugging(input_array, label_array, slice_number=orig_z_slice, processing_type=f"", dataset_name=unique_dataset.lower(), case_number=case_number)
                    self.save_plots_for_debugging(resampled_input_array, resampled_label_array, slice_number=resampled_z_slice, processing_type="resampling", dataset_name=unique_dataset.lower(), case_number=case_number)
                    self.save_plots_for_debugging(cropped_input_array, cropped_label_array, slice_number=cropped_z_slice, processing_type=f"cropping", dataset_name=unique_dataset.lower(), case_number=case_number)


                # 5️⃣ final channel dimension and add channel dimension for pytorch
                final_input = self.final_channel_dimension(cropped_input_array, lambda x: np.transpose(x, (2,1,0))) # transpose to ( z, y, x)
                final_input = final_input[np.newaxis, ...] # add channel dimension to the input array, so the shape becomes (1, z, y, x) for pytorch
                final_label = self.final_channel_dimension(cropped_label_array, lambda x: np.transpose(x, (2,1,0))) # transpose to ( z, y, x)

                # populate X_vol and Y_vol arrays
                X_vol[index, 0, ...] = final_input
                # one hot encode labels
                for indiv_class in range(n_class):
                    Y_vol[index, indiv_class, ...] = (final_label == indiv_class).astype(np.float32)

                # populate info_volumes array
                volume_info_dict[f'{unique_dataset.lower()}_{object.lower()}_case{case_number}'] = {
                    'dataset': unique_dataset.lower(),
                    'object': object.lower(),
                    'case_number': self.convert_for_json(case_number),
                    'input_path': input_path,
                    'label_path': label_path,

                    'n_channel': self.convert_for_json(n_channel),
                    'n_class': self.convert_for_json(n_class),
                    'desired_spacing': self.convert_for_json(desired_spacing),
                    'new_shape': self.convert_for_json(new_shape),

                    'input_spacing': self.convert_for_json(input_spacing),
                    'input_size': self.convert_for_json(input_size),
                    'input_physical_size':self.convert_for_json( np.array(input_spacing) * np.array(input_size)),
                    'label_spacing': self.convert_for_json(label_spacing),
                    'label_size': self.convert_for_json(label_size),
                    'label_extent': self.convert_for_json(seg_extent),
                    'label_physical_size': self.convert_for_json(np.array(input_spacing) * (np.array(seg_extent[1::2])- np.array(seg_extent[0::2]))),
                    'label_bounding_box_size': self.convert_for_json((np.array(seg_extent[1::2])- np.array(seg_extent[0::2]))),

                    'resampeled_input_spacing': self.convert_for_json(resampled_input_spacing),
                    'resampled_input_size': self.convert_for_json(resampled_input_size),
                    'resampled_input_physical_size': self.convert_for_json(np.array(resampled_input_spacing) * np.array(resampled_input_size)),
                    'resampled_label_spacing': self.convert_for_json(resampled_label_spacing),
                    'resampled_label_size': self.convert_for_json(resampled_label_size),
                    'resampled_label_extent': self.convert_for_json(resampled_seg_extent),
                    'resampled_label_physical_size': self.convert_for_json(np.array(resampled_input_spacing) * (np.array(resampled_seg_extent[1::2])- np.array(resampled_seg_extent[0::2]))),
                    'resampled_label_bounding_box_size': self.convert_for_json((np.array(resampled_seg_extent[1::2])- np.array(resampled_seg_extent[0::2]))),

                    'cropped_input_shape': self.convert_for_json(cropped_input_array.shape),
                    'cropped_label_shape': self.convert_for_json(cropped_label_array.shape),
                    'cropping_slices': self.convert_for_json(slices),
                    'cropping_pad_width': self.convert_for_json(pad_width),
                    'cropped_label_extent': self.convert_for_json(cropped_seg_extent),
                    'cropped_label_bounding_box_size': self.convert_for_json((np.array(cropped_seg_extent[1::2])- np.array(cropped_seg_extent[0::2]))),

                    'final_input_shape': self.convert_for_json(X_vol[index].shape),
                    'final_label_shape': self.convert_for_json(Y_vol[index].shape),
                }

                if to_print and to_debug:
                    print(f"\n\n 🐞 Volume info for {unique_dataset} case {case_number}:")
                    for key, value in volume_info_dict[f'{unique_dataset.lower()}_{object.lower()}_case{case_number}'].items():
                        print(f"   {key}: {value}")
                    print("\n\n")

            # save to hpf5 file
            save_path = f"preprocessed_data/{unique_dataset.lower()}_{object.lower()}.h5"
            info_json = json.dumps(self.convert_for_json(volume_info_dict))
            with h5py.File(save_path, "w") as f:
                f.create_dataset("X", data=X_vol, compression="gzip")
                f.create_dataset("Y", data=Y_vol, compression="gzip")
                f.create_dataset("info", data=np.string_(info_json))

                
            

            if to_print:
                print(f"✅ Saved {N} cases of {unique_dataset} ({object}) to {save_path}")

if __name__ == "__main__":
    pre_processor = PreProcessor()
    pre_processor.process_all_volumes()