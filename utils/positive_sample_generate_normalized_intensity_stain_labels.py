import cv2
import os
import os.path as osp
from tqdm import tqdm
import helper_functions
from math import floor, ceil
import pandas as pd
import numpy as np

# Path to training-ready positive/negative samples
sample_path = '/home/cameron/Dropbox (University of Michigan)/DL_training/data/classifier_training_samples/'
pos_sample_path = osp.join(sample_path, 'positive_CORRECT_LABELS')

# Get positive sample filenames
pos_code_file_names = os.listdir(pos_sample_path)
# We'll load image data as numpy arrays and store in a list
sample_ndarray_list = []
# We'll save image metadata from the filename here
sample_metadata = []

skipped_samples = []
# For each positive
for code_file_name_ind in tqdm(range(len(pos_code_file_names))):
    code_file_name = pos_code_file_names[code_file_name_ind]
    if 'code ' in code_file_name.split("_")[0] and '.png' in code_file_name.split("_")[-1]:
        file_info  = code_file_name.split('_')
        
        sample_metadata.append(file_info)
        
        # Load region.
        region = cv2.imread(
            os.path.join(pos_sample_path, code_file_name), cv2.IMREAD_ANYDEPTH
        )
        
        if region is not None:
            sample_ndarray_list.append(region)
        else:
            print(f'ERROR WHEN LOADING: {code_file_name}')

pos_intensities = helper_functions.get_particle_intensities(particle_image_list=sample_ndarray_list)
pos_min_intensity = floor(min(pos_intensities))
pos_max_intensity = ceil(max(pos_intensities))

intensities_df = pd.DataFrame(pos_intensities)
intensities_df.columns = ['normalized_pixel_intensity']

min_bin_count = 2
num_bins = len(intensities_df) // min_bin_count
while True:
    bins = pd.qcut(intensities_df['normalized_pixel_intensity'], q=num_bins, duplicates='drop')
    if bins.value_counts().min() >= min_bin_count:
        break
    num_bins -= 1

# assign a label to each bin for each datapoint in the original dataframe
intensities_df['normalized_pixel_intensity_bin'] = bins.cat.codes.apply(lambda x: x)
pd.DataFrame({'positive_sample_bin_intervals': bins.cat.categories}).to_csv(osp.join(sample_path, 'positive_sample_bin_intervals.csv'), index=False)

print(intensities_df)
print(bins)
print(f"Num Bins: {num_bins}")
print(intensities_df['normalized_pixel_intensity_bin'].value_counts())
print(intensities_df['normalized_pixel_intensity_bin'].value_counts().describe())

intensities_df.to_csv(osp.join(sample_path, 'positive_sample_normalized_pixel_intensity_to_bin_label.csv'), index=False)

bin_labels = intensities_df['normalized_pixel_intensity_bin'].to_list()

assert len(bin_labels) == len(sample_metadata)

for sample_metadatum, bin_label, pos_code_file_name in tqdm(zip(sample_metadata, bin_labels, pos_code_file_names)):
    old_file_info = '_'.join(sample_metadatum)
    sample_metadatum[1] = str(bin_label)
    new_file_info = '_'.join(sample_metadatum)
    print((old_file_info, bin_label, new_file_info))
    os.rename(osp.join(pos_sample_path, pos_code_file_name), osp.join(pos_sample_path, new_file_info))