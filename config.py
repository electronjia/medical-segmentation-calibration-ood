# The code will include the file paths needed for this project
# excel filepath and column names
excel_paths = "medical_segmentation_datasets_filepaths.xlsx"
dataset_name_col = 'dataset' 
case_col = 'case'
object_col = 'object'
input_col = 'input'
label_col = 'label'

# pre-processing parameters
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

# define sx, sy, sz
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
