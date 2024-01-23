import os
from tqdm import tqdm

directory = '/home/cameron/Dropbox (University of Michigan)/DL_training/data/classifier_training_samples/positive_CORRECT_LABELS_MINT'
substring = 'code 6_'
filetype = '.png'

for filename in tqdm(os.listdir(directory)):
    if substring in filename and filetype in filename:
        os.remove(os.path.join(directory, filename))
