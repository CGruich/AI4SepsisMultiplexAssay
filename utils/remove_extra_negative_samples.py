import os
import random
import shutil
from tqdm import tqdm

# Extra negative regions
extra_sample_path = '/home/cameron/Dropbox (University of Michigan)/DL_training/data/classifier_training_samples/composite/negative'

# Get filenames of samples
extra_sample_files = os.listdir(extra_sample_path)
# Train on code 1-5 so exclude regions labeled with code 6
extra_sample_files = [sample for sample in extra_sample_files]

# Choose samples to make negative/positive samples a 1:1 class balance
chosen_samples = random.sample(extra_sample_files, 219)

# Copy files over
for sample in tqdm(chosen_samples):
    if '.png' in sample:
        os.remove(os.path.join(extra_sample_path, sample))
