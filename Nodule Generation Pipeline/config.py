import random

'''
"PARAMS" INFO IS TAKEN FROM-
https://pubmed.ncbi.nlm.nih.gov/21371772/
'''

PARAMS = {
    "Cancer_solid": {
        "mean_HU": -11,
        "std_HU": 28,
        "sphere_r": 40,
        "euler_num": random.gauss(-50, 10),
        "hole_r": 6,
        "volume_shape": (128, 128, 128),
        "target_compactness": 0.55,
        "compactness_std_dev": 0.1,
        "hu_thresh": -100
    },
    "Cancer_BAC": {
        "mean_HU": -376,
        "std_HU": 33,
        "sphere_r": 40,
        "euler_num": random.gauss(-4, 2),
        "hole_r": 6,  #
        "volume_shape": (128, 128, 128),
        "target_compactness": 0.48,
        "compactness_std_dev": 0.1,
        "hu_thresh": 20
    },
    "Active_Fibrosis": {
        "mean_HU": -17,
        "std_HU": 28,
        "sphere_r": 40,
        "euler_num": random.gauss(-10, 5),
        "hole_r": 6,
        "volume_shape": (128, 128, 128),
        "target_compactness": 0.53,
        "compactness_std_dev": 0.05,
        "hu_thresh": -50
    },
     "Inactive_Fibrosis": {
        "mean_HU": 26,
        "std_HU": 28,
        "sphere_r": 40,
        "euler_num": random.gauss(-55, 20),
        "hole_r": 6,
        "volume_shape": (128, 128, 128),
        "target_compactness": 0.45,
        "compactness_std_dev": 0.15,
        "hu_thresh": -50
    },
     "RBC": {
        "mean_HU": -75,
        "std_HU": 29,
        "sphere_r": 40,
        "euler_num": random.gauss(2, 0),
        "hole_r": 6,
        "volume_shape": (128, 128, 128),
        "target_compactness": 0.8,
        "compactness_std_dev": 0.1,
        "hu_thresh": -50
    },
     "Necrosis": {
        "mean_HU": -3,
        "std_HU": 32,
        "sphere_r": 40,
        "euler_num": random.gauss(0, 0),
        "hole_r": 6,
        "volume_shape": (128, 128, 128),
        "target_compactness": 0.66,
        "compactness_std_dev": 0.02,
        "hu_thresh": -50
    }
}

flow_data_csv = "3d_flow_data_final.csv"
image_size  = 128
grid_size = (image_size, image_size, image_size)
num_boundary_pts  = 300  
num_ctrl_x, num_ctrl_y, num_ctrl_z =8, 8, 8
compactness_mean_tolerance = 0.03
number_of_hollow_points_to_find = 5
min_distance_between_target_points = 20


tissue_type = "Inactive_Fibrosis"
mean_HU = PARAMS[tissue_type]["mean_HU"]  
std_HU = PARAMS[tissue_type]["std_HU"]
sphere_r = PARAMS[tissue_type]["sphere_r"]
euler_num = PARAMS[tissue_type]["euler_num"]
hole_r = PARAMS[tissue_type]["hole_r"]
volume_shape = PARAMS[tissue_type]["volume_shape"]
target_compactness = PARAMS[tissue_type]["target_compactness"]
compactness_std_dev = PARAMS[tissue_type]["compactness_std_dev"]
hu_thresh = PARAMS[tissue_type]["hu_thresh"]

alphas = {
    "Active_Fibrosis": 0.13,
    "Cancer_BAC": 0.22,
    "Cancer_solid": 0.46,
    "Inactive_Fibrosis": 0.13,
    "Necrosis": 0.03,
    "RBC": 0.03
}

'''
just setting some default value, will change with modify_params() from class Nodule()
'''

tissue_type = "Inactive_Fibrosis"
mean_HU = PARAMS[tissue_type]["mean_HU"]  
std_HU = PARAMS[tissue_type]["std_HU"]
sphere_r = PARAMS[tissue_type]["sphere_r"]
euler_num = PARAMS[tissue_type]["euler_num"]
hole_r = PARAMS[tissue_type]["hole_r"]
volume_shape = PARAMS[tissue_type]["volume_shape"]
target_compactness = PARAMS[tissue_type]["target_compactness"]
compactness_std_dev = PARAMS[tissue_type]["compactness_std_dev"]
hu_thresh = PARAMS[tissue_type]["hu_thresh"]