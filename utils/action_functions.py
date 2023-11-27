from model_training import (
    RegionClassifierTrainerGPU,
    MSEROptimizer,
    CodeClassifierTrainerGPU,
)
from object_detection import RegionDetector
from sklearn.model_selection import StratifiedKFold, train_test_split
from tqdm import tqdm
import cv2
import numpy as np
import pandas as pd
import os
from pathlib import Path
import json
from datetime import datetime
from . import (
    helper_functions,
    bayesian,
)
# For setting RNG
import torch
import random

# Hyperparameter optimization
import optuna
from optuna.trial import TrialState
from optuna.samplers import TPESampler
# Reloading checkpointed Bayesian optimization studies
import joblib

def set_experimental_rng(pipeline_inputs: dict,
                         exact: bool = False):
    random_state = pipeline_inputs.get("strat_kfold", {}).get("random_state")
    verbose = pipeline_inputs.get("verbose")

    assert type(random_state) is int
    assert type(verbose) is bool
    if random_state is not None:
        torch.manual_seed(random_state)
        torch.cuda.manual_seed(random_state)
        # Improves exact reproducibility but at the cost of performance
        if exact:
            # Pick a deterministic convolutional algorithm
            # By default, this is nondeterministic algorithm selection
            # across different hardware
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        np.random.seed(random_state)
        # Probably not needed but just in-case
        random.seed(random_state)
    
        if verbose is not None:
            if verbose:
                print(f'Random Seed Set: {random_state}')

def find_mser_params(pipeline_inputs: dict):

    # Set random state if available...
    set_experimental_rng(pipeline_inputs=pipeline_inputs,
                         exact=False)
    
    raw_directory = pipeline_inputs['raw_directory']
    code_list = pipeline_inputs['code_list']

    for code in code_list:
        # If we are processing colored images of the barcoded particles,
        if pipeline_inputs['color']:
            # The directory of all the raw images for a particular code color (e.g., (1))
            code_raw_directory = os.path.join(raw_directory, 'code ' + code + ' color')
            print(f'Examining Code {code}\n{code_raw_directory}')
        # Or if we are processing greyscale images of the barcoded particles,
        else:
            # The directory of all the raw images for a particular code (e.g., (1))
            code_raw_directory = os.path.join(raw_directory, 'code ' + code)
            print(f'Examining Code {code}\n{code_raw_directory}')

        # Load
        # Image naming convention: 1.tiff or (for a reference image) 1_ref.tiff
        raw_img_names = []
        reference_img_names = []
        particle_location_names = []
        for file_name in os.listdir(code_raw_directory):
            if 'amp' in file_name or 'phase' in file_name or 'MSER' in file_name:
                continue

            # Old code made MSER params for each image, but this is now changed
            # Here, we restrict MSER param searching to '1.tiff' and '1_ref.tiff' with each code.
            # We then take the MSER params from these images and use them as MSER params for the code.
            if ('1_ref.tiff' == file_name) or ('1_1_ref.tiff' == file_name):
                reference_img_names.append(file_name)
            elif 'particle_locations' in file_name:
                particle_location_names.append(file_name)
            elif ('1.tiff' == file_name) or ('1_1.tiff' == file_name):
                raw_img_names.append(file_name)

        # Image filenames will be loaded in parallel. e.g., "1.tiff" will be loaded with its own reference image "1_ref.tiff"
        # Here we ensure the file names are sorted alphanumerically so each file name is paired with the appropriate reference
        # and particle position list.
        helper_functions.sort_alphanumeric(raw_img_names)
        helper_functions.sort_alphanumeric(particle_location_names)
        helper_functions.sort_alphanumeric(reference_img_names)

        print(f'Loading Raw Images:\n{raw_img_names}')
        print(f'Loading Reference Images:\n{reference_img_names}')
        print(f'Loading Particle Locations:\n{particle_location_names}')

        # Ensure we have as many reference images as we do raw images
        assert len(raw_img_names) == len(reference_img_names)

        holograms = []
        references = []
        grayscales = []
        particle_locations = []
        for i in range(len(raw_img_names)):
            raw_img_path = os.path.join(code_raw_directory, raw_img_names[i])
            reference_img_path = os.path.join(code_raw_directory, reference_img_names[i])
            particle_location_path = os.path.join(
                code_raw_directory, particle_location_names[i]
            )

            print(f'Raw Image Path: {raw_img_path}')
            print(f'Reference Image Path: {reference_img_path}')
            print(f'Particle Locations Path: {particle_location_path}\n')

            assert Path(raw_img_path).is_file() and Path(reference_img_path).is_file()

            holograms.append(cv2.imread(raw_img_path, cv2.IMREAD_ANYDEPTH))
            references.append(cv2.imread(reference_img_path, cv2.IMREAD_ANYDEPTH))

            with open(particle_location_path, 'r') as particle_file:
                particle_locations_json = dict(json.load(particle_file))
            particle_locations_list = list(particle_locations_json['particle_locations'])
            particle_locations.append(particle_locations_list)

        for hologram_image, reference_image in zip(holograms, references):
            grayscale_hologram = helper_functions.normalize_by_reference(
                hologram_image, reference_image
            )
            grayscales.append(grayscale_hologram)

        opt = MSEROptimizer(
            normalized_images=grayscales,
            particle_locations=particle_locations,
            num_iterations=pipeline_inputs['number_iterations'],
        )

        save_directory = os.path.join(code_raw_directory, 'MSER_Parameters.json')
        opt.train(save_directory=save_directory)
        print(f'\nMSER Parameters Saved To:\n{save_directory}\n')


def train_region_classifier(
    pipeline_inputs: dict = None,
    load_data_path: str = 'data/classifier_training_samples',
    model_save_path: str = 'data/models/region',
    test_size: float = 0.20,
    cross_validate: bool = False,
    k: int = 5,
    random_state: int = 100,
    stratify_by_stain: bool = False,
    verbose: bool = True,
    log: bool = True,
    timestamp: str = None,
    save_every_n: int = 1,
    # Hyperparameters
    batch_size: int = 192,
    lr: float = 3e-4,
    fc_size: int = 64,
    fc_num: int = 2,
    dropout_rate: float = 0.3,
    patience: int = 10,
    # Only used for Bayesian hyperparameter optimization
    hyper_dict: dict = None
):

    # Set random state if available...
    set_experimental_rng(pipeline_inputs=pipeline_inputs,
                         exact=False)
    
    # Train a new classifier with the data located under
    # data/classifier_training_samples/positive and
    # data/classifier_training_samples/negative

    if pipeline_inputs is not None:
        load_data_path = pipeline_inputs.get("sample_parent_directory")
        model_save_path = pipeline_inputs.get("model_save_parent_directory")
        test_size = pipeline_inputs.get("test_size")
        cross_validate = pipeline_inputs.get("strat_kfold", {}).get("activate")
        k = pipeline_inputs.get("strat_kfold", {}).get("num_folds")
        random_state = pipeline_inputs.get("strat_kfold", {}).get("random_state")
        stratify_by_stain = pipeline_inputs.get("strat_kfold", {}).get("stratify_by_stain", False)
        verbose = pipeline_inputs.get("verbose")
        log = pipeline_inputs.get("log")
        timestamp = pipeline_inputs.get("timestamp")
        save_every_n = pipeline_inputs.get("save_every_n")

        # Hyperparameters
        batch_size = pipeline_inputs.get("batch_size")
        lr = pipeline_inputs.get("lr")
        fc_size = pipeline_inputs.get("fc_size")
        fc_num = pipeline_inputs.get("fc_num")
        dropout_rate = pipeline_inputs.get("dropout_rate")
        patience = pipeline_inputs.get("patience")

    if timestamp is None:
        timestamp = datetime.now().strftime('%m_%d_%y_%H:%M')

    # Returns list of lists
    data_list = helper_functions.load_data(load_data_path, verbose=verbose, stratify_by_stain=stratify_by_stain)

    # Based on the format of the return result of .load_data(),
    # Extract all the targets of the training samples
    targets = np.array(list(zip(*data_list))[-1])

    # All the samples
    dataset = np.asarray(
        data_list, dtype=object
    )

    # Do a stratified train/test split of all samples into training and test datasets
    # Returns the actual samples, not the indices of the samples.
    training_data, test_data = train_test_split(
        dataset, 
        test_size=test_size, 
        stratify=targets, 
        random_state=random_state
    )

    if stratify_by_stain:
        training_targets_stain = np.asarray(list(zip(*training_data))[-1])
        test_targets_stain = np.asarray(list(zip(*test_data))[-1])
        training_targets = helper_functions.stain_labels_to_training_labels(data=training_targets_stain)
        test_targets = helper_functions.stain_labels_to_training_labels(data=test_targets_stain)

        for i, new_label in enumerate(training_targets):
            training_data[i, -1] = new_label

        for i, new_label in enumerate(test_targets):
            test_data[i, -1] = new_label
    else:
        training_targets = np.asarray(list(zip(*training_data))[-1])
        test_targets = np.asarray(list(zip(*test_data))[-1])
        training_targets_stain = training_targets
        test_targets_stain = test_targets

    # CG: Stratified k-Fold cross-validation
    # Object for stratified k-fold cross-validation splitting of training dataset into a new training dataset and validation dataset
    splits = StratifiedKFold(n_splits=k, shuffle=True, random_state=random_state)

    training_data_idx = np.arange(len(training_data))
    cross_val_scores = {
        'Val_Loss': [],
        'Val_Acc': [],
        'Test_Loss': [],
        'Test_Acc': [],
    }
    fold_index = 1
    # For each fold, define training data indices and validation data indices from the input training dataset
    for fold, (train_idx, val_idx) in enumerate(
        splits.split(training_data_idx, y=training_targets_stain)
    ):
        if verbose:
            print('\n\nFold {}'.format(fold + 1))
        
        if hyper_dict is None:
            # Define a region classifier
            trainer = RegionClassifierTrainerGPU(
                model_save_path=model_save_path,
                save_every_n=save_every_n,
                batch_size=batch_size,
                lr=lr,
                fc_size=fc_size,
                fc_num=fc_num,
                dropout_rate=dropout_rate,
                k=fold_index,
                patience=patience,
                verbose=verbose,
                log=log,
                timestamp=timestamp,
            )
        # Else, if we are Bayesian optimizing
        else:
            # CG: I avoid using .get() here because the hyperparameter dictionary should be strictly and completely specified with values
            batch_size = hyper_dict['bs']
            lr = hyper_dict['lr']
            fc_size = hyper_dict['fc_size']
            fc_num = hyper_dict['fc_num']
            dropout_rate = hyper_dict['dr']

            trainer = RegionClassifierTrainerGPU(
                model_save_path=model_save_path,
                save_every_n=save_every_n,
                batch_size=batch_size,
                lr=lr,
                fc_size=fc_size,
                fc_num=fc_num,
                dropout_rate=dropout_rate,
                k=fold_index,
                patience=patience,
                verbose=verbose,
                log=log,
                timestamp=timestamp,
            )

        trainer.load_data(
            training_data,
            training_targets,
            train_idx,
            val_idx,
            test_dataset_np=test_data,
            test_targets_np=test_targets,
        )
        # Cross-validation is coded into the trainer, which will add and return cross-validation scores for each fold
        cross_val_scores = trainer.train(
            cross_validation=cross_validate, cross_validation_scores=cross_val_scores
        )
        # Keep track of what k-fold we are on for book-keeping
        fold_index = fold_index + 1

    if pipeline_inputs['verbose']:
        print('\nTRAINING COMPLETE.\nCross-Validation Dictionary:')
        print(cross_val_scores)
        # Average cross-validation scores
        for key, value in cross_val_scores.items():
            print('Avg. ' + str(key) + ': ' + str(np.array(value).mean()))
    return cross_val_scores


def classify_regions(pipeline_inputs: dict = None):
    assert pipeline_inputs is not None

    # Set random state if available...
    set_experimental_rng(pipeline_inputs=pipeline_inputs,
                         exact=False)
    
    # Get the parent directory of raw images for all codes
    raw_directory = pipeline_inputs['raw_directory']
    # Get the parent directory to save visual representation of all regions detected by MSER
    hull_directory = pipeline_inputs['hull_directory']
    # Indicates if we draw rectangles around stable regions of particles/not-particles found by MSER and save to hull_directory
    draw_blobs = pipeline_inputs['draw_blobs']

    # Get the list of codes to process from the control panel pipeline inputs
    code_list = pipeline_inputs['code_list']

    # For each code to process
    for code_num in tqdm(code_list):
        # Get the raw image directory for that code
        raw_code_dir = os.path.join(raw_directory, 'code ' + code_num)
        # Define positive/negative sample save paths
        pos_save_dir = os.path.join(
            'data/classifier_training_samples/', 'code ' + code_num, 'positive'
        )
        neg_save_dir = os.path.join(
            'data/classifier_training_samples/', 'code ' + code_num, 'negative'
        )
        if not os.path.exists(pos_save_dir):
            os.makedirs(pos_save_dir)
            os.makedirs(neg_save_dir)
        print(f'Processing and saving data to:\n{pos_save_dir}')
        print(f'Processing and saving data to:\n{neg_save_dir}')

        # Load MSER parameters for an initial division of positive/negative samples
        # This should be a list parallel with the ordering of images (1.tiff, 2.tiff, etc.)
        # Each entry in the list is the filename of the MSER parameters to be used on that particular image
        # (e.g., entry 0 in the list refers to MSER parameters used for 1.tiff, entry 1 refers to 2.tiff, etc.)
        MSER_params = pipeline_inputs['MSER_params_per_code'][code_num]

        # We'll store the final processed images to be fed to the region classifier in this list
        grayscale_holograms = []
        
        # We are using a region detector with an assumed flexibility to switch out the MSER params per raw image of barcoded particles
        # In reality, we are using one set of MSER parameters for each code to bootstrap training samples for all raw images of barcoded particles for a code
        # Nevertheless, we keep this flexibility here by storing separate copies of the region detector in a list, parallel with 'MSER_params' list
        region_detectors = []

        # For each raw image for a particular code,
        for MSER_index in range(len(MSER_params)):
            # These counters for positive/negative samples are counted per-reference image, not per-code
            sum_pos = 0
            sum_neg = 0
            raw_image_id = str(MSER_index + 1)
            with open(os.path.join(raw_code_dir, MSER_params[MSER_index]), 'r') as MSERObj:
                mser_dict = dict(json.load(MSERObj)['optimizer.max']['params'])
            # Define a region detector for positive/negative division of samples
            # This region detector is not optimized to be the most accurate
            # Rather, we use the region detector as a means of initializing positive/negative samples.
            # Later, we will actually hyperparameter optimize the region detectors for accurate positive/negative sample separation
            region_detector = RegionDetector(MSER_parameters=mser_dict)
            region_detectors.append(region_detector)

            image_name = raw_image_id + ".tiff"
            reference_name = raw_image_id + "_ref.tiff"

            # Read the raw code image
            hologram = cv2.imread(
                '{}/{}'.format(raw_code_dir, image_name), cv2.IMREAD_ANYDEPTH,
            )

            # Read the reference image
            reference = cv2.imread(
                '{}/{}'.format(raw_code_dir, reference_name), cv2.IMREAD_ANYDEPTH,
            )

            # Normalize the raw code image by the reference image for robustness
            # to lighting conditions/noise/etc.
            grayscale_hologram = helper_functions.normalize_by_reference(
                hologram, reference)
            
            grayscale_hologram = grayscale_hologram.astype(np.float32)
            # Append the raw image data with the reference image data.
            grayscale_holograms.append(
                (hologram, raw_image_id, reference)
            )

        # For each normalized image representing a collection of barcoded particles, along with its' corresponding MSER parameter-loaded region detector,
        for hologram, region_detector in tqdm(zip(grayscale_holograms, region_detectors)):
            holo, raw_image_id, reference = hologram
            save_img_file = '{}_{}_regions.png'.format(code_num, raw_image_id)
            save_img_name = os.path.join(hull_directory, f'code {code_num}')
            os.makedirs(save_img_name, exist_ok=True)
            save_img_name = os.path.join(save_img_name, save_img_file)

            (positive_regions, negative_regions,) = region_detector.detect_regions(
                holo, reference, draw_blobs=draw_blobs, save_img_name=save_img_name
            )
            for i in range(len(positive_regions)):
                file_path = '{}/{}_{}_positive_{}.png'.format(
                    pos_save_dir, code_num, raw_image_id, i + sum_pos
                )
                region = positive_regions[i]
                shape = region.shape
                new_shape = (*shape[1:], shape[0])
                region = region.reshape(new_shape)
                cv2.imwrite(file_path, region)
            sum_pos = sum_pos + len(positive_regions)

            for i in range(len(negative_regions)):
                file_path = '{}/{}_{}_negative_{}.png'.format(
                    neg_save_dir, code_num, raw_image_id, i + sum_neg
                )
                region = negative_regions[i]
                shape = region.shape
                new_shape = (*shape[1:], shape[0])
                region = region.reshape(new_shape)
                cv2.imwrite(file_path, region)
            sum_neg = sum_neg + len(negative_regions)


def train_code_classifier(
    pipeline_inputs: dict = None,
    load_data_path: str = 'data/classifier_training_samples',
    model_save_path: str = 'data/models/code',
    codes: list = None,
    test_size: float = 0.20,
    cross_validate: bool = False,
    k: int = 5,
    random_state: int = 100,
    stratify_by_stain: bool = False,
    verbose: bool = True,
    log: bool = True,
    timestamp: str = None,
    save_every_n: int = 1,
    # Hyperparameters
    batch_size: int = 192,
    lr: float = 3e-4,
    fc_size: int = 64,
    fc_num: int = 2,
    dropout_rate: float = 0.3,
    patience: int = 10,
    warmup: int = 200,
    # Only used for Bayesian hyperparameter optimization
    hyper_dict: dict = None,
    bayes_trial = None,
):

    if pipeline_inputs is not None:
        
        # Set random state if available...
        set_experimental_rng(pipeline_inputs=pipeline_inputs,
                            exact=False)
        
        load_data_path = pipeline_inputs.get("sample_parent_directory")
        model_save_path = pipeline_inputs.get("model_save_parent_directory")
        codes = pipeline_inputs.get("code_list")
        test_size = pipeline_inputs.get("test_size")
        cross_validate = pipeline_inputs.get("strat_kfold", {}).get("activate")
        k = pipeline_inputs.get("strat_kfold", {}).get("num_folds")
        random_state = pipeline_inputs.get("strat_kfold", {}).get("random_state")
        stratify_by_stain = pipeline_inputs.get("strat_kfold", {}).get("stratify_by_stain", False)
        verbose = pipeline_inputs.get("verbose")
        log = pipeline_inputs.get("log")
        timestamp = pipeline_inputs.get("timestamp")
        save_every_n = pipeline_inputs.get("save_every_n")

        # Hyperparameters
        batch_size = pipeline_inputs.get("batch_size")
        lr = pipeline_inputs.get("lr")
        weight_decay = pipeline_inputs.get("weight_decay")
        fc_size = pipeline_inputs.get("fc_size")
        fc_num = pipeline_inputs.get("fc_num")
        dropout_rate = pipeline_inputs.get("dropout_rate")
        patience = pipeline_inputs.get("patience")
        warmup = pipeline_inputs.get("warmup")
        
    if timestamp is None:
        timestamp = datetime.now().strftime('%m_%d_%y_%H:%M')

    assert codes is not None and len(codes) != 0

    code_data_composite = []
    for code in codes:
        code_path = os.path.join(load_data_path, 'code ' + code)
        code_data = helper_functions.load_code(code_folder_path=code_path, verbose=verbose, stratify_by_stain=stratify_by_stain)
        code_data_composite = code_data_composite + code_data

    if stratify_by_stain:
        # Extract targets
        targets = list(zip(*code_data_composite))[-1]
        #print('517 targets')
        #print(targets)
        
        # Define a dictionary of {code_number: normalized_pixel_intensity_bin_ID}
        # This denotes the binned intensity of each picture of each code, in a dictionary.
        norm_stain_label_per_code_dict = dict([(str(code_num), []) for code_num in range(1, len(codes) + 1)])
        #print('520 norm_stain_label_per_code_dict')
        #print(norm_stain_label_per_code_dict)
        # For each target,
        for target in targets:
            # Go to that particular code in the dictionary ([0]) and add the bin ID of the normalized pixel intensity ([1])
            norm_stain_label_per_code_dict[target.split('_')[0]].append((int(target.split('_')[1],)))
        #print('524 norm_stain_label_per_code_dict')
        #print(norm_stain_label_per_code_dict)
        
        # Define a similar dictionary in the format of {code number: ...}
        # Currently, our labels are a result of binning normalized pixel intensity across all particle images
        # However, when stratifying, we need a minimum of 2 datapoints.
        # There may be 2 datapoints at a particular stain level across all codes, but not within one code.
        # Thus, our stratification will crash.

        # To correct for this, instead of:
        # pixel intensity -> normalized pixel intensity -> bin of normalized pixel intensity
        # We can do:
        # pixel intensity -> normalized pixel intensity -> bin of normalized pixel intensity -> bin-of-bin of normalized pixel intensity
        # In doing so, we still group similar-stain sampled together in the same class, but in a somewhat more coarse-grain way
        # while satisfying this >=2 datapoints-per-bin condition
        code_norm_pixel_intensity_bin_bins = {}

        # For each code,
        for code_num in range(1, len(codes) + 1):
            # Get the bins corresponding to normalized pixel intensity, convert to dataframe
            code_stain_df = pd.DataFrame(norm_stain_label_per_code_dict[str(code_num)])
            code_stain_df.columns = ['normalized_pixel_intensity_bin']

            # Now get the 'bin-of-bin' of normalized pixel intensity
            # Here, we force 20 datapoints per bin (min_bin_count)
            code_stain_df, bin_mappings = helper_functions.guarantee_bin_width_for_bin_count(df=code_stain_df, col_name='normalized_pixel_intensity_bin', min_bin_count=5)
            #print(code_stain_df.shape)
            code_stain_dict = code_stain_df.to_dict()
            
            # Create a nested dictionary representing a dataframe
            # Each row of each code has (1) the bin of normalized pixel intensity and (2) the bin-of-bin of normalized pixel intensity
            code_norm_pixel_intensity_bin_bins[str(code_num)] = code_stain_dict
            # Save the bin-of-bin data
            pd.DataFrame({f'code_{code_num}_sample_bin_intervals': bin_mappings.cat.categories}).to_csv(os.path.join(load_data_path, f'code_{code_num}_sample_bin_intervals.csv'), index=False)
        
        # Store new targets here
        # e.g.,
        # 1_172 (code, normalized_pixel_intensity_bin) -> 1_14(code, normalized_pixel_intensity_bin_of_bin)
        new_targets = []
        #print('targets')
        #print(targets)
        #print(len(targets))
        #print('540 code_norm_pixel_intensity_bin_bins')
        #print(code_norm_pixel_intensity_bin_bins)
        # For each target,
        for target in targets:
            target_info = target.split('_')
            # Get the code
            code = target_info[0]
            # Get the normalized pixel intensity bin (the old label)
            norm_pixel_intensity_bin = target_info[1]
            # For each normalized pixel intensity bin recorded in our dictionary
            for row, code_norm_pixel_intensity_bin in code_norm_pixel_intensity_bin_bins[code]['normalized_pixel_intensity_bin'].items():
                # If the current bin noted matches a bin noted in the dictionary
                if str(code_norm_pixel_intensity_bin) == norm_pixel_intensity_bin:
                    # Get the corresponding bin-of-bin for the bin noted
                    target_info[1] = str(code_norm_pixel_intensity_bin_bins[code]['normalized_pixel_intensity_bin_bin'][row])
                    # Reconstruct the new sample filename, replacing the bin stain label with the bin-of-bin stain label
                    new_target_info = '_'.join(target_info)
                    #print((counter, new_target_info))
                    new_targets.append(new_target_info)
                    # We are done with this sample, move onto the next one
                    break
        #print(code_norm_pixel_intensity_bin_bins[code]['normalized_pixel_intensity_bin'])
        #print(len(targets))
        #print(len(new_targets))
        #print(targets[0])
        #print(new_targets[0])
        #print(np.asarray(targets))
        #print(np.asarray(new_targets))
        targets = np.asarray(new_targets)

    else:
        # Based on the format of the return result of helper_functions.load_code(),
        # Extract all the targets of the training samples
        targets = np.asarray(list(zip(*code_data_composite))[-1])

    # All the samples
    dataset = np.asarray(code_data_composite, dtype=object)
    dataset[:, -1] = targets.flatten()

    #print('571 dataset')
    #print(dataset)
    #print(dataset.shape)
    
    # Do a straified train/test split of all samples into training and test datasets
    # Returns the actual samples, not the indices of the samples.
    training_data, test_data = train_test_split(
        dataset,
        test_size=test_size,
        stratify=targets,
        random_state=random_state,
    )
    
    #print('584 Training Data')
    #print(training_data)
    #print('586 Test Data')
    #print(test_data)

    if stratify_by_stain:
        training_targets_stain = np.asarray(list(zip(*training_data))[-1])
        test_targets_stain = np.asarray(list(zip(*test_data))[-1])
        
        #print('593 training targets stain')
        #print(training_targets_stain)
        #print('595 test targets stain')
        #print(test_targets_stain)

        # Strips stain info
        # e.g., label '1_100' just becomes a label '1'
        # We only stratify by stain at the dataloader level
        # We stratify-by-code within the model during training, which is why we strip this information here
        training_targets = helper_functions.stain_labels_to_training_labels(data=training_targets_stain)
        test_targets = helper_functions.stain_labels_to_training_labels(data=test_targets_stain)

        #print('601 training targets')
        #print(training_targets)
        #print('603 test targets')
        #print(test_targets)

        for i, new_label in enumerate(training_targets):
            training_data[i, -1] = new_label

        for i, new_label in enumerate(test_targets):
            test_data[i, -1] = new_label
        #print('611 training_data')
        #print(training_data)
        #print('612 test data')
        #print(test_data)

    else:
        training_targets = np.asarray(list(zip(*training_data))[-1])
        test_targets = np.asarray(list(zip(*test_data))[-1])
        training_targets_stain = training_targets
        test_targets_stain = test_targets
        #print('training targets')
        #print(training_targets)
        #print('test targets')
        #print(test_targets)

    # CG: Stratified k-Fold cross-validation
    # Define a class to do the stratified splitting into folds
    splits = StratifiedKFold(
        n_splits=k,
        shuffle=True,
        random_state=random_state,
    )

    # Get the indices of the training dataset
    training_data_idx = np.arange(len(training_data))
    # Dictionary to hold cross-validation scores
    cross_val_scores = {
        'Val_Loss': [],
        'Val_Acc': [],
        'Test_Loss': [],
        'Test_Acc': [],
    }
    fold_index = 1
    # For each fold in the training dataset, define a new training dataset and validation dataset based off the training targets
    for fold, (train_idx, val_idx) in enumerate(
        splits.split(training_data_idx, y=training_targets_stain)
    ):
        if verbose:
            print('\n\nFold {}'.format(fold + 1))
        # Define a code classifier
        # If we are not Bayesian optimizing,
        if hyper_dict is None:
            trainer = CodeClassifierTrainerGPU(
                codes=codes,
                model_save_path=model_save_path,
                save_every_n=save_every_n,
                batch_size=batch_size,
                lr=lr,
                weight_decay=weight_decay,
                fc_size=fc_size,
                fc_num=fc_num,
                dropout_rate=dropout_rate,
                k=fold_index,
                patience=patience,
                warmup=warmup,
                verbose=verbose,
                log=log,
                timestamp=timestamp,
            )
        # Else, if we are Bayesian optimizing,
        else:
            # CG: I avoid using .get() here because the hyperparameter dictionary should be strictly and completely specified with values
            batch_size = hyper_dict['bs']
            lr = hyper_dict['lr']
            fc_size = hyper_dict['fc_size']
            fc_num = hyper_dict['fc_num']
            dropout_rate = hyper_dict['dr']
            weight_decay = hyper_dict['wd']

            trainer = CodeClassifierTrainerGPU(
                codes=codes,
                model_save_path=model_save_path,
                save_every_n=save_every_n,
                batch_size=batch_size,
                lr=lr,
                weight_decay=weight_decay,
                fc_size=fc_size,
                fc_num=fc_num,
                dropout_rate=dropout_rate,
                k=fold_index,
                patience=patience,
                warmup=warmup,
                verbose=verbose,
                log=log,
                timestamp=timestamp,
            )
        # Load code classifier training data and validation data
        # Validation data is taken from the training dataset and targets (training_data, training_targets)
        # Test dataset and test targets are inputted separately
        trainer.load_data(
            training_data,
            training_targets,
            train_idx,
            val_idx,
            test_dataset_np=test_data,
            test_targets_np=test_targets
        )
        # Train
        if bayes_trial is not None:
            cross_val_scores = trainer.train(
                cross_validation=cross_validate,
                cross_validation_scores=cross_val_scores,
            )
        else:
            cross_val_scores = trainer.train(
                cross_validation=cross_validate,
                cross_validation_scores=cross_val_scores,
            )
        # Keep track of what k-fold we are on for book-keeping
        fold_index = fold_index + 1

        if bayes_trial is not None:
            intermediate_accuracy = np.array(cross_val_scores['Val_Acc']).mean()
            bayes_trial.report(intermediate_accuracy, fold_index)

            if bayes_trial.should_prune():
                raise optuna.TrialPruned()

    # Print out the average cross-validation results
    if pipeline_inputs['verbose']:
        print('\nTRAINING COMPLETE.\nCross-Validation Dictionary:')
        print(cross_val_scores)
        for key, value in cross_val_scores.items():
            print('Avg. ' + str(key) + ': ' + str(np.array(value).mean()))
    return cross_val_scores


def bayesian_optimize_code_classifer(pipeline_inputs: dict = None):
    # Currently only implemented for the Jupyter notebook pipeline,
    assert pipeline_inputs is not None

    # Set random state if available...
    set_experimental_rng(pipeline_inputs=pipeline_inputs,
                         exact=False)
    random_state = pipeline_inputs.get("strat_kfold", {}).get("random_state")

    # Defaults to None if not specified in pipeline inputs
    checkpoint = pipeline_inputs.get('checkpoint')

    # If restarting a study from a saved checkpoint,
    if checkpoint is not None:
        study = joblib.load(checkpoint)
    else:
        # Create an OpTuna study, maximize the accuracy
        study = optuna.create_study(direction='maximize', 
                                    sampler=TPESampler(seed=random_state),
                                    pruner=optuna.pruners.HyperbandPruner())

    # By default, OpTuna objective functions for objective minimization/maximization does not accept custom input variables
    # However, we can easily accomodate custom input variables in this way with some lambda operations,

    def objective_with_custom_input(trial):
        return bayesian.objective_code_classifier(trial, pipeline_inputs)

    study = bayesian.checkpoint_study(
        study,
        objective_with_custom_input,
        num_trials=pipeline_inputs['num_hpo'],
        checkpoint_every=pipeline_inputs['save_every_n'],
        checkpoint_path=os.path.join(
            pipeline_inputs['checkpoint_path'], pipeline_inputs['timestamp']
        ),
    )

    # Get the pruned trials (trials pruned prematurely)
    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    # Get the completed trials
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

    # Summarize
    print('\n\nStudy statistics: ')
    print('  Number of finished trials: ', len(study.trials))
    print('  Number of pruned trials: ', len(pruned_trials))
    print('  Number of complete trials: ', len(complete_trials))

    print('Best trial:')
    trial = study.best_trial

    print('  Value: ', trial.value)

    print('  Params: ')
    for key, value in trial.params.items():
        print('    {}: {}'.format(key, value))


def bayesian_optimize_region_classifer(pipeline_inputs: dict = None):
    # Currently only implemented for the Jupyter notebook pipeline,
    assert pipeline_inputs is not None

    # Set random state if available...
    set_experimental_rng(pipeline_inputs=pipeline_inputs,
                         exact=False)
    random_state = pipeline_inputs.get("strat_kfold", {}).get("random_state")

    # Defaults to None if not specified in pipeline inputs
    checkpoint = pipeline_inputs.get('checkpoint')

    # If restarting a study from a saved checkpoint,
    if checkpoint is not None:
        study = joblib.load(checkpoint)
    else:
        # Create an OpTuna study, maximize the accuracy
        study = optuna.create_study(direction='maximize', 
                                    sampler=TPESampler(seed=random_state))
    
    # By default, OpTuna objective functions for objective minimization/maximization does not accept custom input variables
    # However, we can easily accomodate custom input variables in this way with some lambda operations,

    def objective_with_custom_input(trial):
        return bayesian.objective_region_classifier(trial, pipeline_inputs)

    study = bayesian.checkpoint_study(
        study,
        objective_with_custom_input,
        num_trials=pipeline_inputs['num_hpo'],
        checkpoint_every=pipeline_inputs['save_every_n'],
        checkpoint_path=os.path.join(
            pipeline_inputs['checkpoint_path'], pipeline_inputs['timestamp']
        ),
    )

    # Get the pruned trials (trials pruned prematurely)
    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    # Get the completed trials
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

    # Summarize
    print('\n\nStudy statistics: ')
    print('  Number of finished trials: ', len(study.trials))
    print('  Number of pruned trials: ', len(pruned_trials))
    print('  Number of complete trials: ', len(complete_trials))

    print('Best trial:')
    trial = study.best_trial

    print('  Value: ', trial.value)

    print('  Params: ')
    for key, value in trial.params.items():
        print('    {}: {}'.format(key, value))


def find_particle_intensities():
    from scipy.signal import convolve2d

    conv_window_size = 10
    convolution_kernel = np.ones((conv_window_size, conv_window_size)) / (
        conv_window_size * conv_window_size
    )

    dropbox_path = (
        'E:/Dropbox/Dropbox (Partners HealthCare)/DL_training/data/raw/Protein assay'
    )
    code_paths = [
        'code 1_IL-1/230529_IL-1 [Dz 9~9.5]',
        'code 2_IL-3/230520_IL-3 [Dz 9~9.5]',
        'code 3_IL-6/230601_IL-6 [Dz 9~9.5]',
        'code 4_IL-10/230606_IL-10 [Dz 9.5]',
        'code 5_TNF-a/230615_TNF-a [Dz 9.5]',
        'code 6_ANG-2/230425_ANG-2 [Dz 10]',
    ]
    # code_paths = ["code 1_IL-1/230529_IL-1 [Dz 9~9.5]"]

    code_data = {}
    for code_path in code_paths:
        full_path = os.path.join(dropbox_path, code_path)

        ref = cv2.imread(os.path.join(full_path, 'ref.tiff'), cv2.IMREAD_ANYDEPTH).astype(
            np.float32
        )
        ref = convolve2d(ref, convolution_kernel, mode='same')

        code = code_path[: code_path.find('_')]
        code_data[code] = {}
        for file_name in os.listdir(full_path):
            if 'trol' in file_name or 'ref' in file_name:
                continue

            print('Loading', file_name)

            if '.json' in file_name:
                file_name_key = file_name[: file_name.rfind('_particle_locations')]
                if file_name_key not in code_data[code].keys():
                    code_data[code][file_name_key] = {}

                file = open(os.path.join(full_path, file_name), 'r')
                code_data[code][file_name_key]['particle_locations'] = dict(json.load(file))
            else:
                file_name_key = file_name[: file_name.rfind('.tiff')]
                if file_name_key not in code_data[code].keys():
                    code_data[code][file_name_key] = {}

                intensity_str = file_name[: file_name.find('_')]
                stripped = intensity_str.replace('.tif', "")
                stripped = stripped.replace('.tiff', "")
                known_intensity = int(stripped)
                hologram = cv2.imread(os.path.join(full_path, file_name), cv2.IMREAD_ANYDEPTH)
                grayscale = helper_functions.normalize_by_reference(
                    hologram, ref, scale_to_bit_depth=False, ref_already_convolved=True
                )
                code_data[code][file_name_key]['normalized_hologram'] = grayscale
                code_data[code][file_name_key]['known_intensity'] = known_intensity

    print(
        'Code Number\tProtein Concentration\tMean Pixel Intensity\tStdev Pixel Intensity\tNumber of Particles Counted'
    )
    for code, code_data in code_data.items():
        intensity_dict = {}
        for file_name, file_data in code_data.items():
            grayscale = file_data['normalized_hologram']
            locations = file_data['particle_locations']
            intensity = file_data['known_intensity']
            intensity_estimate = helper_functions.find_intensity(grayscale, locations)
            if intensity not in intensity_dict.keys():
                intensity_dict[intensity] = intensity_estimate
            else:
                intensity_dict[intensity] += intensity_estimate

        intensity_data = []
        for intensity, estimates in intensity_dict.items():
            intensity_data.append(
                (intensity, np.mean(estimates), np.std(estimates), len(estimates))
            )

        intensity_data = sorted(intensity_data, key=lambda x: x[0])
        for arg in intensity_data:
            intensity, avg_intensity_estimate, intensity_estimate_std, particle_count = arg
            print(
                '{}\t{}\t{}\t{}\t{}'.format(
                    code,
                    intensity,
                    avg_intensity_estimate,
                    intensity_estimate_std,
                    particle_count,
                )
            )
        print()
