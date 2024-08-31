import os
from tqdm import tqdm

sample_root_path = "/home/cameron/Dropbox (University of Michigan)/DL_training/data/classifier_training_samples/composite/negative"

old_code_file_names_abs = []
new_code_file_names_abs = []

code_file_names = os.listdir(sample_root_path)

for code_file_name_ind in tqdm(range(len(code_file_names))):
    code_file_name = code_file_names[code_file_name_ind]
    if '(' in code_file_name.split("_")[0] and ')' in code_file_name.split("_")[0]:
        code, ref, initial_region_label, sample_id = code_file_name.split('_')
        code = code[1]

        new_filename = f'code {code}_0_{ref}_UNDEFsid{initial_region_label}{sample_id}_UNDEFsid{initial_region_label}{sample_id}_128x128.png'
        new_code_file_names_abs.append(os.path.join(sample_root_path, new_filename))
        old_code_file_names_abs.append(os.path.join(sample_root_path, code_file_name))

for old_file, new_file in zip(old_code_file_names_abs, new_code_file_names_abs):
    os.rename(old_file, new_file)