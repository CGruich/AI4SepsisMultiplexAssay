import os

dir_path = '/mnt/c/Users/CGrui/Dropbox (University of Michigan)/DL_training/data/classifier_training_samples/composite/positive'

# Get available stain levels
levels = []

for file in os.listdir(dir_path):
    if file.endswith('.png'):
        code, stain_level = file.split('_')[0:2]
        code = code.replace('code ', '')
        levels.append(code + '_' + stain_level)

stains = {stain_level: 0 for stain_level in levels}

for file in os.listdir(dir_path):
    if file.endswith('.png'):
        file = os.path.basename(file)
        
        for stain_level in stains.keys():
            stains[stain_level] += file.count(stain_level)

print(stains)
