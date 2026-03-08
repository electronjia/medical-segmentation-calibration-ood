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
    'spleen': (1.5, 1.5, 1.5), # decathlon

    'heart': (1.5, 1.5, 1.5), # decathlon

    # 'prostate': (1.25, 1.25, 1.25), # decathlon, not used

    'hippocampus': (0.75, 0.75, 0.75), # decathlon

    'pancreas': (2.0, 2.0, 2.0), # decathlon

    'sliver': (2.5, 2.5, 2.5), #sliver

    'liver': (1.5,1.5,1.5), # decathlon, was not defined

    'kidney': (1.0,1.0,1.0), # kits19, was not defined

    'lung': (1.5,1.5,1.5), # decathlon, was not defined

    'hepaticvessel': (1.5,1.5,1.5), # decathlon, was not defined

    'colon': (1.5,1.5,1.5), # decathlon, was not defined
}

# define sx, sy, sz
crop_dict = {
    'spleen': (135, 189, 155),

    'heart': (135, 189, 155),

    # 'prostate': (135, 189, 155), # decathlon, not used

    'hippocampus': (135, 189, 155),

    'pancreas': (160, 160, 160),

    'liver': (160, 160, 160), # sliver, decathlon, was not defined

    'kidney': (160, 160, 160), # kits19, was not defined

    'lung': (160, 160, 160), # decathlon, was not defined

    'hepaticvessel': (160, 160, 160), # decathlon, was not defined
    
    'colon': (160, 160, 160), # decathlon, was not defined

}
