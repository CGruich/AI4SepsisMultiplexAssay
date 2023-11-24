import cv2
import os
import os.path as osp
from tqdm import tqdm
import helper_functions
from math import floor, ceil
import pandas as pd
import numpy as np

# Path to training-ready positive/negative samples
sample_path = '/home/cameron/Dropbox (University of Michigan)/DL_training/data/classifier_training_samples/Protein_Assay_Training_Samples/new_set'
neg_sample_path = osp.join(sample_path, 'negative')

# Get positive sample filenames
neg_code_file_names = os.listdir(neg_sample_path)
# We'll load image data as numpy arrays and store in a list
sample_ndarray_list = []
# We'll save image metadata from the filename here
sample_metadata = []

# For each positive
for code_file_name_ind in tqdm(range(len(neg_code_file_names))):
    code_file_name = neg_code_file_names[code_file_name_ind]
    if 'code ' in code_file_name.split("_")[0] and '.png' in code_file_name.split("_")[-1]:
        file_info  = code_file_name.split('_')
        
        sample_metadata.append(file_info)
        
        # Load region.
        region = cv2.imread(
            os.path.join(neg_sample_path, code_file_name), cv2.IMREAD_ANYDEPTH
        )

        sample_ndarray_list.append(region)

neg_intensities = helper_functions.get_particle_intensities(particle_image_list=sample_ndarray_list)
neg_min_intensity = floor(min(neg_intensities))
neg_max_intensity = ceil(max(neg_intensities))
bin_width = 200

intensities_df = pd.DataFrame(neg_intensities)
intensities_df.columns = ['normalized_pixel_intensity']
bins = list(range(neg_min_intensity, neg_max_intensity, bin_width))
bins = [-np.inf, 0] + bins + [np.inf]

intensities_df['normalized_pixel_intensity_bin'] = pd.cut(intensities_df['normalized_pixel_intensity'], bins=bins, labels=False).astype('int32')

print(intensities_df['normalized_pixel_intensity_bin'].value_counts())

intensities_df.to_csv(osp.join('negative_sample_normalized_pixel_intensity_to_bin_label.csv'), index=False)

bin_labels = intensities_df['normalized_pixel_intensity_bin'].to_list()

assert len(bin_labels) == len(sample_metadata)

for sample_metadatum, bin_label, neg_code_file_name in tqdm(zip(sample_metadata, bin_labels, neg_code_file_names)):
    sample_metadatum[1] = str(bin_label)
    new_file_info = '_'.join(sample_metadatum)
    os.rename(osp.join(neg_sample_path, neg_code_file_name), osp.join(neg_sample_path, new_file_info))