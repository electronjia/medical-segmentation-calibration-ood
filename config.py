import torch
# The code will include the file paths needed for this project
# excel filepath and column names
excel_paths = "medical_segmentation_datasets_filepaths.xlsx"
dataset_name_col = 'dataset' 
case_col = 'case'
object_col = 'object'
input_col = 'input'
label_col = 'label'

# pre-processing parameters_______________________________________________________________
desired_spacing_dict = {
    'spleen': (1.5, 1.5, 1.5), # decathlon, task 5 spleen data 34 had mismatched label array size

    'heart': (1.5, 1.5, 1.5), # decathlon

    'hippocampus': (0.75, 0.75, 0.75), # decathlon

    'pancreas': (2.0, 2.0, 2.0), # decathlon

    'sliver': (2.5, 2.5, 2.5), #sliver

    'liver': (1.5,1.5,1.5), # decathlon, was not used. i was not able to use task liver due to data issues.

    'kidney': (1.0,1.0,1.0), # kits19, was not defined

    'lung': (1.5,1.5,1.5), # decathlon, was not defined

    'hepaticvessel': (1.5,1.5,1.5), # decathlon, was not defined

    'colon': (1.5,1.5,1.5), # decathlon, was not defined
}

# define chunk size
# make sure it is not bigger than the new shape dimensions
chunk_x = 96
chunk_y = 96
chunk_z = 96
stride = chunk_x // 4

# define Sx, Sy, Sz
crop_dict = {
    'spleen': (160, 160, 160), # in the paper it is (135, 189, 155)

    'heart': (160, 160, 160),  # in the paper it is (135, 189, 155)

    'hippocampus': (160, 160, 160), # in the paper it is (135, 189, 155)

    'pancreas': (160, 160, 160),

    'liver': (160, 160, 160), # sliver, decathlon, was not defined. i was not able to use task liver due to data issues. only sliver was used.

    'kidney': (160, 160, 160), # kits19, was not defined

    'lung': (160, 160, 160), # decathlon, was not defined

    'hepaticvessel': (160, 160, 160), # decathlon, was not defined

    'colon': (160, 160, 160), # decathlon, was not defined

}

# define n channel for each dataset. i believe all the CT datasets only have 1 channel
n_channel_dict = {
    'spleen': 1,
    'heart': 1,
    'hippocampus': 1,
    'pancreas': 1,
    'liver': 1, # i was not able to use task liver due to data issues. only sliver was used.
    'kidney': 1,
    'lung': 1,
    'hepaticvessel': 1,
    'colon': 1,

    }

# define number of classes in segmentation for each dataset
# this is important because not every slice in a single image from each dataset contains all the classes
n_class_dict = {
    'spleen': 2,
    'heart': 2,
    'hippocampus': 3,
    'pancreas': 3,
    'liver': 3, # i was not able to use task liver due to data issues. only sliver was used.
    'sliver': 2,
    'kidney': 3,
    'lung': 2,
    'hepaticvessel': 3,
    'colon': 2,

    }


# h5 dataset related parameters_______________________________________________
preprocessed_data_path = r"preprocessed_data/"
input_data_key = "X"
label_data_key = "Y"
data_info_key = "info"


# deep learning related parameters_____________________________________________
keep_keys = ['dataset', 'object', 'case_number'] # i can only keep certain keys because there are some keys that are None which are breaking the dataloader.
seed = 42
train_frac = 0.3
val_frac = 0.4
test_frac = 0.3
device = 'cuda' if torch.cuda.is_available() else 'cpu'
n_classes = 3
lr = 10.0e-5
num_epochs = 150
experiment_details = 'all_segresnet'