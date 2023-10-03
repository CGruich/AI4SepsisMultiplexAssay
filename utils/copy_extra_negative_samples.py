import os
import random
import shutil
from tqdm import tqdm

# Extra negative regions
extra_sample_path = '/home/cameron/Dropbox (University of Michigan)/DL_training/data/classifier_training_samples/Protein_Assay_Training_Samples/negative'
# Region classifier training data path
copy_dest_path = '/home/cameron/Dropbox (University of Michigan)/DL_training/data/classifier_training_samples/composite/negative'

# Get filenames of samples
extra_sample_files = os.listdir(extra_sample_path)
# Train on code 1-5 so exclude regions labeled with code 6
extra_sample_files = [sample for sample in extra_sample_files]

# Choose samples to make negative/positive samples a 1:1 class balance
chosen_samples = random.sample(extra_sample_files, 16)

# Copy files over
for sample in tqdm(chosen_samples):
    if '.png' in sample:
        src = os.path.join(extra_sample_path, sample)
        dst = os.path.join(copy_dest_path, sample)
        shutil.copyfile(src, dst)
