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
import os
from pathlib import Path
import json
from datetime import datetime
from . import (
    helper_functions,
    bayesian,
)

# Hyperparameter optimization
import optuna
from optuna.trial import TrialState

# CG: Leaving here for now temporarily, in case needed
# Probably warrants deprecation.
"""def get_intensity(
    img_folder='/Users/apple/Dropbox (University of Michigan)/iMAPS_coding/Selected images for DL/1-1/1-10 (1)',
    region_detector_path='data/best/best_region_classifier.pt',
):
    region_detector = RegionDetector(model_load_path=region_detector_path)
    folder_name = img_folder[img_folder.rfind('/') + 1 :]
    holograms = []

    for ref_name in os.listdir(img_folder):
        if '.tiff' not in ref_name:
            continue
        if 'ref' in ref_name:
            print('referencing: ', ref_name)
            reference = cv2.imread('{}/{}'.format(img_folder, ref_name), cv2.IMREAD_ANYDEPTH)

            for image_name in os.listdir(img_folder):
                code = ref_name.replace('_ref.tiff', "")
                if (
                    code == image_name.replace('.tiff', "").split('_')[0]
                    and '.tiff' in image_name
                    and 'ref' not in image_name
                ):
                    hologram = cv2.imread(
                        '{}/{}'.format(img_folder, image_name), cv2.IMREAD_ANYDEPTH
                    )
                    hologram = hologram.astype(np.float32)
                    holograms.append((hologram, image_name.replace('.tiff', ""), reference))

    intensities = []
    file_names = []
    # Find intensities
    for hologram in holograms:
        holo, name, reference = hologram
        if not os.path.exists('data/hulls'):
            os.makedirs('data/hulls')
        save_img_name = 'data/hulls/{}_{}_regions.png'.format(folder_name, name)
        print('processing', name)
        intensity = region_detector.get_intensity(holo, reference, save_img_name=save_img_name)
        intensities.append(intensity)
        file_names += [folder_name + '/' + name, "", ""]

    # write to a csv file
    intensities = list(itertools.zip_longest(*intensities, fillvalue=["", "", ""]))
    intensities = [list(itertools.chain(*x)) for x in intensities]

    intensities = (
        [file_names] + [['x', 'y', 'intensity'] * int(int(len(file_names)) / 3)] + intensities
    )

    if not os.path.exists('data/intensities'):
        os.makedirs('data/intensities')
    csv_path = 'data/intensities/' + folder_name + '.csv'
    with open(csv_path, 'w') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.write_rows(intensities)"""


def find_mser_params(pipeline_inputs: dict):
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
    val_size: int = 0.20,
    cross_validate: bool = False,
    k: int = 5,
    random_state: int = 100,
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
):

    # Train a new classifier with the data located under
    # data/classifier_training_samples/positive and
    # data/classifier_training_samples/negative

    if pipeline_inputs is not None:
        load_data_path = pipeline_inputs["sample_parent_directory"]
        model_save_path = pipeline_inputs["model_save_parent_directory"]
        val_size = pipeline_inputs["val_size"]
        cross_validate = pipeline_inputs["strat_kfold"]["activate"]
        k = pipeline_inputs["strat_kfold"]["num_folds"]
        random_state = pipeline_inputs["strat_kfold"]["random_state"]
        verbose = pipeline_inputs["verbose"]
        log = pipeline_inputs["log"]
        timestamp = pipeline_inputs["timestamp"]
        save_every_n = pipeline_inputs["save_every_n"]

        # Hyperparameters
        batch_size = pipeline_inputs["batch_size"]
        lr = pipeline_inputs["lr"]
        fc_size = pipeline_inputs["fc_size"]
        fc_num = pipeline_inputs["fc_num"]
        dropout_rate = pipeline_inputs["dropout_rate"]

    if timestamp is None:
        timestamp = datetime.now().strftime('%m_%d_%y_%H:%M')

    # Returns list of lists
    data_list = helper_functions.load_data(load_data_path, verbose=verbose)
    # Based on the format of the return result of .load_data(),
    # Extract all the targets of the training samples
    targets = np.array(list(zip(*data_list))[-1])
    # All the samples
    dataset = np.asarray(
        helper_functions.load_data(load_data_path, verbose=verbose), dtype=object
    )

    # Do a stratified train/test split of all samples into training and test datasets
    # Returns the actual samples, not the indices of the samples.
    training_data, validation_data = train_test_split(
        dataset, test_size=val_size, stratify=targets
    )
    training_targets = np.asarray(list(zip(*training_data))[-1])

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
        splits.split(training_data_idx, y=training_targets)
    ):
        if verbose:
            print('\n\nFold {}'.format(fold + 1))
        # Define a region classifier
        trainer = RegionClassifierTrainerGPU(
            model_save_path=model_save_path,
            save_every_n=save_every_n,
            batch_size=batch_size,
            lr=lr,
            fc_size=fc_size,
            fc_num=fc_num,
            dropout_rate=dropout_rate,
            verbose=verbose,
            log=log,
            timestamp=timestamp,
        )
        trainer.load_data(
            load_data_path,
            training_data,
            train_idx,
            val_idx,
            test_dataset=validation_data,
        )
        # Cross-validation is coded into the trainer, which will add and return cross-validation scores for each fold
        cross_val_scores = trainer.train(
            cross_validate=cross_validate, cross_validation_scores=cross_val_scores
        )
        # Keep track of what k-fold we are on for book-keeping
        fold_index = fold_index + 1

    if verbose:
        print('\nTRAINING COMPLETE.\nCross-Validation Dictionary:')
        print(cross_val_scores)
        # Average cross-validation scores
        for key, value in cross_val_scores.items():
            print('Avg. ' + str(key) + ': ' + str(np.array(value).mean()))
    return cross_val_scores


def classify_regions(pipeline_inputs: dict = None):
    assert pipeline_inputs is not None

    # Get the parent directory of raw images for all codes
    raw_directory = pipeline_inputs['raw_directory']
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

        filenames = sorted(os.listdir(raw_code_dir))

        # Load MSER parameters for an initial division of positive/negative samples
        MSER_params = pipeline_inputs['MSER_params_per_code'][code_num]
        # For each set of MSER parameters, which corresponds to each reference image of each code,
        for MSERFile in tqdm(MSER_params):
            # These counters for positive/negative samples are counted per-reference image, not per-code
            sum_pos = 0
            sum_neg = 0
            raw_image_id = MSERFile.rstrip('_MSER.json')
            with open(os.path.join(raw_code_dir, MSERFile), 'r') as MSERObj:
                mser_dict = dict(json.load(MSERObj)['optimizer.max']['params'])
            # Define a region detector for positive/negative division of samples
            # This region detector is not optimized to be the most accurate
            # Rather, we use the region detector as a means of initializing positive/negative samples.
            # Later, we will actually hyperparameter optimize the region detectors for accurate positive/negative sample separation
            region_detector = RegionDetector(MSER_parameters=mser_dict)
            # For each reference image in the raw image directory for a particular code,
            ref_name = raw_image_id + '_ref.tiff'

            holograms = []
            # Ignore the file if it is not an image file
            if '.tiff' not in ref_name:
                continue
            # If the file is a reference image,
            elif 'ref' in ref_name:
                # Load reference
                reference = cv2.imread(
                    os.path.join(raw_code_dir, ref_name), cv2.IMREAD_ANYDEPTH
                )
                # For each image in the raw image directory of a particular code
                for image_name in filenames:
                    # The raw image file is just the name of the reference file without the reference designation
                    code = ref_name.replace('_ref.tiff', "")
                    # If we found the code image that corresponds with its' own refernece image,
                    if (
                        code == image_name.replace('.tiff', "").split('_')[0]
                        and '.tiff' in image_name
                        and 'ref' not in image_name
                    ):
                        # Read the raw code image
                        hologram = cv2.imread(
                            '{}/{}'.format(raw_code_dir, image_name), cv2.IMREAD_ANYDEPTH,
                        )
                        hologram = hologram.astype(np.float32)
                        # Append the raw image data with the reference image data.
                        holograms.append(
                            (hologram, image_name.replace('.tiff', ""), reference)
                        )

                for hologram in holograms:
                    holo, name, reference = hologram
                    save_img_name = 'data/test/{}_{}_regions.png'.format(code_num, name)
                    (positive_regions, negative_regions,) = region_detector.detect_regions(
                        holo, reference, save_img_name=save_img_name
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
    pipeline_inputs: dict = None, timestamp: str = None, hyper_dict: dict = None
):
    assert pipeline_inputs is not None

    if timestamp is None:
        timestamp = datetime.now().strftime('%m_%d_%y_%H:%M')

    # Timestamps for record-keeping
    if pipeline_inputs['timestamp'] is None:
        pipeline_inputs['timestamp'] = datetime.now().strftime('%m_%d_%y_%H:%M')

    codes = pipeline_inputs['code_list']
    trainer = CodeClassifierTrainerGPU(
        codes, model_save_path=pipeline_inputs['model_save_parent_directory']
    )
    code_data_composite = []
    for code in codes:
        code_path = os.path.join(pipeline_inputs['sample_parent_directory'], 'code ' + code)
        code_data = helper_functions.load_code(code_folder_path=code_path)
        code_data_composite = code_data_composite + code_data

    # Total composite dataset samples
    # Only print verbosely if we are not hyperparameter optimizing
    if hyper_dict is None:
        print('Total Composite Dataset Training Samples:\n{}'.format(len(code_data_composite)))

    # Based on the format of the return result of helper_functions.load_code(),
    # Extract all the targets of the training samples
    targets = np.array(list(zip(*code_data_composite))[-1])
    # All the samples
    dataset = np.asarray(code_data_composite, dtype=object)

    # Do a straified train/test split of all samples into training and test datasets
    # Returns the actual samples, not the indices of the samples.
    training_data, validation_data = train_test_split(
        dataset,
        test_size=pipeline_inputs['test_size'],
        stratify=targets,
        random_state=pipeline_inputs['strat_kfold']['random_state'],
    )
    # Train targets
    training_targets = np.asarray(list(zip(*training_data))[-1])

    # CG: Stratified k-Fold cross-validation
    # If doing strat. k-fold cross-val.,
    if pipeline_inputs['strat_kfold']['activate']:
        # Define a class to do the stratified splitting into folds
        splits = StratifiedKFold(
            n_splits=pipeline_inputs['strat_kfold']['num_folds'],
            shuffle=True,
            random_state=pipeline_inputs['strat_kfold']['random_state'],
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
            splits.split(training_data_idx, y=training_targets)
        ):
            if pipeline_inputs['verbose']:
                print('\n\nFold {}'.format(fold + 1))
            # Define a code classifier
            # If we are not Bayesian optimizing,
            if hyper_dict is None:
                trainer = CodeClassifierTrainerGPU(
                    codes=codes,
                    model_save_path=pipeline_inputs['model_save_parent_directory'],
                    verbose=pipeline_inputs['verbose'],
                    log=pipeline_inputs['log'],
                    timestamp=pipeline_inputs['timestamp'],
                )
            # Else, if we are Bayesian optimizing,
            else:
                trainer = CodeClassifierTrainerGPU(
                    codes=codes,
                    model_save_path=pipeline_inputs['model_save_parent_directory'],
                    batch_size=hyper_dict['bs'],
                    lr=hyper_dict['lr'],
                    fc_size=hyper_dict['fc_size'],
                    fc_num=hyper_dict['fc_num'],
                    dropout_rate=hyper_dict['dr'],
                    verbose=pipeline_inputs['verbose'],
                    log=pipeline_inputs['log'],
                    timestamp=pipeline_inputs['timestamp'],
                )
            # Load code classifier training data and validation data
            # Validation data is taken from the training dataset and targets (training_data, training_targets)
            # Test dataset and test targets are inputted separately
            trainer.load_data(
                pipeline_inputs['sample_parent_directory'],
                training_data,
                training_targets,
                train_idx,
                val_idx,
                test_dataset=validation_data,
            )
            # Train
            cross_val_scores = trainer.train(
                cross_validation=pipeline_inputs['strat_kfold']['activate'],
                cross_validation_scores=cross_val_scores,
            )
            # Keep track of what k-fold we are on for book-keeping
            fold_index = fold_index + 1

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
    # Create an OpTuna study, maximize the accuracy
    study = optuna.create_study(direction='maximize')
    # By default, OpTuna objective functions for objective minimization/maximization does not accept custom input variables
    # However, we can easily accomodate custom input variables in this way with some lambda operations,

    def objective_with_custom_input(trial):
        return bayesian.objective_code_classifier(trial, pipeline_inputs)

    study = bayesian.checkpoint_study(
        study,
        objective_with_custom_input,
        num_trials=pipeline_inputs['num_hpo'],
        checkpoint_every=pipeline_inputs['save_every'],
        checkpoint_path=os.path.join(
            pipeline_inputs['checkpoint_path'], pipeline_inputs['timestamp']
        ),
    )

    """# Optimize the study
    study.optimize(objective_with_custom_input, 
                    n_trials=pipeline_inputs["num_hpo"], 
                    timeout=pipeline_inputs["timeout"],
                    callbacks=[pruning_callback])"""

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
    # Create an OpTuna study, maximize the accuracy
    study = optuna.create_study(direction='maximize')
    # By default, OpTuna objective functions for objective minimization/maximization does not accept custom input variables
    # However, we can easily accomodate custom input variables in this way with some lambda operations,

    def objective_with_custom_input(trial):
        return bayesian.objective_region_classifier(trial, pipeline_inputs)

    study = bayesian.checkpoint_study(
        study,
        objective_with_custom_input,
        num_trials=pipeline_inputs['num_hpo'],
        checkpoint_every=pipeline_inputs['save_every'],
        checkpoint_path=os.path.join(
            pipeline_inputs['checkpoint_path'], pipeline_inputs['timestamp']
        ),
    )

    """# Optimize the study
    study.optimize(objective_with_custom_input, 
                    n_trials=pipeline_inputs["num_hpo"], 
                    timeout=pipeline_inputs["timeout"],
                    callbacks=[pruning_callback])"""

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
