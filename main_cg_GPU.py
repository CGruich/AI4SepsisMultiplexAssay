from model_training import RegionClassifierTrainerGPU, MSEROptimizer, CodeClassifierTrainerGPU
from object_detection import RegionDetector
from code_classification import CodeClassifier
from sklearn.model_selection import StratifiedKFold, train_test_split
from tqdm import tqdm
import cv2
import numpy as np
import pandas as pd
import os
import argparse
import csv
import itertools
import re as regex
from pathlib import Path
import json
from datetime import datetime
# Hyperparameter optimization
import optuna
from optuna.trial import TrialState
# For checkpointing hyperparameter optimization
import pickle
from utils import helper_functions

# Objective function for Bayesian optimization with OpTuna
def objective(trial,  pipeline_inputs: dict = None):
    # Learning rate
    lr = trial.suggest_float("learning_rate", 1e-8, 1e-2)
    # Batch size
    bs = trial.suggest_int("batch_size", 128, 1024, 64)
    # Fully connected layer size
    fcSize = trial.suggest_int("fully_connected_size", 128, 1024, 64)
    # Number of fully connected layers
    fcNum = trial.suggest_int("fully_connected_layers", 1, 5, 1)
    # Dropout rate
    dr = trial.suggest_float("dropout_rate", 0.0, 0.8)

    # Dictionary of hyperparameters
    hyper_dict = {"lr": lr, 
                 "bs": bs, 
                 "fcSize": fcSize, 
                 "fcNum": fcNum, 
                 "dr": dr}
    
    # Run stratified k-fold cross-validation with the hyperparameters
    # Via the pipeline functionality of the workflow,
    cross_val_scores = train_code_classifier(pipeline_inputs=pipeline_inputs,
                          timestamp=None,
                          hyper_dict=hyper_dict)
    
    # Average stratified k-fold cross-validation accuracy
    avg_val_accuracy = np.array(cross_val_scores["Val_Acc"]).mean()

    # Return this accuracy, which we rely on for the Bayesian loop
    return avg_val_accuracy

# Define a function that we can use to restart the optimization from the last trial.
# This is useful if we try a high-throughput amount of trials and don't want to start over after a crash, for example
def checkpoint_study(study: optuna.study.Study,
                     objective_function = None,
                     num_trials: int = None,
                     checkpoint_every: int = 100,
                     checkpoint_path: str = None):
    
    for trial in range(num_trials):
        # Optimize a single trial
        study.optimize(objective_function, n_trials=1)

        # Checkpoint every checkpoint_every trials
        if (trial + 1) % checkpoint_every == 0:
            # Make a directory to save checkpoints if not exist
            if not os.path.exists(checkpoint_path):
                os.makedirs(checkpoint_path)
            # Pickle the study to a file
            ckptFile = f"ckpt_{trial + 1}.pkl"
            with open(os.path.join(checkpoint_path, ckptFile), 'wb') as fileObj:
                pickle.dump(study, fileObj)
    return study

def load_data(folder_path, 
              verbose=True):
    positive_sample_folder = os.path.join(folder_path, "positive")
    negative_sample_folder = os.path.join(folder_path, "negative")
    data = []

    # For each image in the positive samples folder.
    for file_name in os.listdir(positive_sample_folder):
        if not file_name.endswith(".png"):
          continue

        # Load region.
        region = cv2.imread(os.path.join(positive_sample_folder, file_name), cv2.IMREAD_ANYDEPTH)
        label = 1

        # Append region and positive label to dataset.
        data.append([region.reshape(1, *region.shape), label])

    n_positive = len(data)

    if verbose:
        print("Loaded {} positive training samples.".format(n_positive))

    # For each image in the negative samples folder.
    for file_name in os.listdir(negative_sample_folder):
        if not file_name.endswith(".png"):
          continue

        # Load region.
        region = cv2.imread(os.path.join(negative_sample_folder, file_name), cv2.IMREAD_ANYDEPTH)
        label = 0
        # Append region and negative label to dataset.
        data.append([region.reshape(1, *region.shape), label])

    if verbose:
        print("Loaded {} negative training samples.".format(len(data) - n_positive))
    
    # Return dataset
    return data

def load_code(code_folder_path,
              verbose=True):
    code_sample_folder = os.path.join(code_folder_path, "positive")
    data = []
    try:
        code_designation = int(code_folder_path[-3:].strip("()"))
    except:
        print("\n\nFAILURE OBTAINING DESIGNATED CODE LABEL FROM THE PARENT FOLDER PATH.\nThe parent folder name may not have been correctly labelled.")

    # For each image in the code's positive samples folder,
    for file_name in os.listdir(code_sample_folder):
        if not file_name.endswith(".png"):
            continue

        # Load region.
        region = cv2.imread(os.path.join(code_sample_folder, file_name), cv2.IMREAD_ANYDEPTH)
        try:
            label = int(file_name[0:2].strip("()"))
            assert label == code_designation
        except:
            print("\n\nFAILURE OBTAINING TARGET LABEL FROM SAMPLE FILENAMES.\nThe sample filenames may not have been correctly labelled.")
        # Append region and negative label to dataset.
        data.append([region.reshape(1, *region.shape), label])
    
    n_positive = len(data)

    if verbose:
        print("Loaded {} positive training samples".format(n_positive))
    
    # Return dataset for one code
    return data

def sort_alphanumeric(string_list):
    assert hasattr(string_list, 'sort'), "ERROR! TYPE {} DOES NOT HAVE A SORT FUNCTION".format(type(string_list))
    """
    Function to sort a list of strings in alphanumeric order.
    Example: the list ['b1','a1','b2','a3','b3','a2'] will be sorted as ['a1', 'a2', 'a3', 'b1', 'b2', 'b3']

    :param string_list: list of strings to sort.
    """
    sorting_key = lambda x: [int(c) if type(c) == int else c for c in regex.split('(-*[0-9]+)', x)]
    string_list.sort(key=sorting_key)

def find_mser_params(pipeline_inputs: dict):
    raw_directory = pipeline_inputs["raw_directory"]
    code_list = pipeline_inputs["code_list"]
    bit_depth = 16
    conv_window_size = 10
    convolution_kernel = np.ones((conv_window_size, conv_window_size)) / (conv_window_size * conv_window_size)

    for code in code_list:
        # The directory of all the raw images for a particular code (e.g., (1))
        code_raw_directory = os.path.join(raw_directory, "code " + code)
        print(f"Examining Code {code}\n{code_raw_directory}")

        # Load
        # Image naming convention: 1.tiff or (for a reference image) 1_ref.tiff
        raw_img_names = []
        reference_img_names = []
        particle_location_names = []
        for file_name in os.listdir(code_raw_directory):
            if "amp" in file_name or "phase" in file_name or "MSER" in file_name:
                continue

            if 'ref' in file_name:
                reference_img_names.append(file_name)
            elif "particle_locations" in file_name:
                particle_location_names.append(file_name)
            else:
                raw_img_names.append(file_name)

        # Image filenames will be loaded in parallel. e.g., "1.tiff" will be loaded with its own reference image "1_ref.tiff"
        # Here we ensure the file names are sorted alphanumerically so each file name is paired with the appropriate reference
        # and particle position list.
        sort_alphanumeric(raw_img_names)
        sort_alphanumeric(particle_location_names)
        sort_alphanumeric(reference_img_names)

        print(f"Loading Raw Images:\n{raw_img_names}")
        print(f"Loading Reference Images:\n{reference_img_names}")
        print(f"Loading Particle Locations:\n{particle_location_names}")

        # Ensure we have as many reference images as we do raw images
        assert len(raw_img_names) == len(reference_img_names)

        holograms = []
        references = []
        grayscales = []
        particle_locations = []
        for i in range(len(raw_img_names)):
            raw_img_path = os.path.join(code_raw_directory, raw_img_names[i])
            reference_img_path = os.path.join(code_raw_directory, reference_img_names[i])
            particle_location_path = os.path.join(code_raw_directory, particle_location_names[i])

            print(f"Raw Image Path: {raw_img_path}")
            print(f"Reference Image Path: {reference_img_path}")
            print(f"Particle Locations Path: {particle_location_path}\n")

            assert Path(raw_img_path).is_file() and Path(reference_img_path).is_file()

            holograms.append(cv2.imread(raw_img_path, cv2.IMREAD_ANYDEPTH))
            references.append(cv2.imread(reference_img_path, cv2.IMREAD_ANYDEPTH))

            with open(particle_location_path, 'r') as particle_file:
                particle_locations_json = dict(json.load(particle_file))
            particle_locations_list = list(particle_locations_json["particle_locations"])
            particle_locations.append(particle_locations_list)


        for hologram_image, reference_image in zip(holograms, references):
            grayscale_hologram = helper_functions.normalize_by_reference(hologram_image, reference_image)
            grayscales.append(grayscale_hologram)

        opt = MSEROptimizer(normalized_images=grayscales,
                            particle_locations=particle_locations,
                            num_iterations=pipeline_inputs["number_iterations"])

        save_directory = os.path.join(code_raw_directory, "MSER_Parameters.json")
        opt.train(save_directory=save_directory)
        print(f"\nMSER Parameters Saved To:\n{save_directory}\n")


def train_region_classifier(pipeline_inputs: dict = None, 
                            load_data_path="data/classifier_training_samples", 
                            model_save_path="data/models/region", 
                            cross_validate=False, 
                            k=5, 
                            hpo_trial=None, 
                            random_state=100, 
                            verbose=True, 
                            log=True, 
                            timestamp: str = None):
    if timestamp is None:
        timestamp = datetime.now().strftime("%m_%d_%y_%H:%M")
    
    # Train a new classifier with the data located under
    # data/classifier_training_samples/positive and
    # data/classifier_training_samples/negative
    
    # Returns list of lists
    data_list = load_data(load_data_path, verbose=verbose)
    # Based on the format of the return result of .load_data(),
    # Extract all the targets of the training samples
    targets = np.array(list(zip(*data_list))[-1])
    # All the samples
    dataset = np.asarray(load_data(load_data_path, verbose=verbose), dtype=object)

    # Do a stratified train/test split of all samples into training and test datasets
    # Returns the actual samples, not the indices of the samples.
    training_data, validation_data = train_test_split(dataset, test_size=0.20, stratify=targets)
    training_targets = np.asarray(list(zip(*training_data))[-1])
    
    # CG: Stratified k-Fold cross-validation
    if cross_validate:
        # Object for stratified k-fold cross-validation splitting of training dataset into a new training dataset and validation dataset
        splits = StratifiedKFold(n_splits=k, shuffle=True, random_state=random_state)
        
        training_data_idx = np.arange(len(training_data))
        cross_val_scores = {"Val_Loss": [], "Val_Acc": [], "Test_Loss": [], "Test_Acc": []}
        fold_index = 1
        # For each fold, define training data indices and validation data indices from the input training dataset
        for fold, (train_idx, val_idx) in enumerate(splits.split(training_data_idx, y=training_targets)):
            if verbose:
                print('\n\nFold {}'.format(fold + 1))
            # Define a region classifier
            trainer = RegionClassifierTrainerGPU(model_save_path=model_save_path, 
                                                 hpo_trial=hpo_trial,
                                                 verbose=verbose, 
                                                 log=log, 
                                                 timestamp=timestamp, 
                                                 k=fold_index)
            trainer.load_data(load_data_path, 
                              training_data, 
                              train_idx, 
                              val_idx, 
                              test_dataset=validation_data)
            # Cross-validation is coded into the trainer, which will add and return cross-validation scores for each fold
            cross_val_scores = trainer.train(cross_validate=cross_validate, cross_val_scores=cross_val_scores)
            # Keep track of what k-fold we are on for book-keeping
            fold_index = fold_index + 1
        
        if verbose:
            print("\nTRAINING COMPLETE.\nCross-Validation Dictionary:")
            print(cross_val_scores)
            # Average cross-validation scores
            for key, value in cross_val_scores.items():
                print("Avg. " + str(key) + ": " + str(np.array(value).mean()))
        return cross_val_scores 
    # If not cross-validating,
    else:
        trainer = RegionClassifierTrainerGPU(model_save_path=model_save_path, hpo_trial=hpo_trial, verbose=verbose, log=log)
        trainer.load_data(load_data_path, dataset, train_idx=None, val_idx=None)
        trainer.train(cross_validate=False, cross_val_scores=None)

def grid_search_region_classifier(pipeline_inputs: dict = None, load_hpo_path="/home/cameron/Dropbox (University of Michigan)/DL_training/hpo/region_classifier_grid_search", load_data_path="data/classifier_training_samples", model_save_path="data/models/region", cross_validation=True, k=5, random_state=100, verbose=False, save_every=50):
    if pipeline_inputs is not None:
        load_hpo_path = pipeline_inputs.get("grid_search_hpo", None).get("hpo_file", None)
        load_data_path = pipeline_inputs.get("sample_parent_directory", None)
        model_save_path = pipeline_inputs.get("model_save_parent_directory", None)
        cross_validation = pipeline_inputs.get("strat_kfold", None).get("activate", None)
        k = pipeline_inputs.get("strat_kfold", None).get("num_folds", None)
        random_state = pipeline_inputs.get("strat_kfold", None).get("random_state", None)
        save_every = pipeline_inputs.get("grid_search_hpo", None).get("save_every", None)
        log = pipeline_inputs.get("grid_search_hpo", None).get("log", None)
        verbose = pipeline_inputs.get("grid_search_hpo", None).get("verbose", None)
        timestamp = pipeline_inputs.get("grid_search_hpo", None).get("hpo_timestamp", datetime.now().strftime("%m_%d_%y_%H:%M"))

        # Ensuring proper pipeline inputs to minimize headaches
        assert load_hpo_path is not None and type(load_hpo_path) is str and ".csv" in load_hpo_path
        assert load_data_path is not None and type(load_data_path) is str
        assert model_save_path is not None and type(model_save_path) is str
        assert cross_validation is not None and type(cross_validation) is bool
        assert k is not None and type(k) is int and k >= 1
        assert random_state is not None and type(random_state) is int
        assert save_every is not None and type(save_every) is int and save_every >= 1
        assert log is not None and type(log) is bool
        assert verbose is not None and type(verbose) is bool
        assert timestamp is not None

        # If cross-validating for each hyperparameter trial,
        if cross_validation:
            # Define dataframes to store cross_validation results
            cv_loss_columns = ["Loss_cv" + str(fold) for fold in range(1, k + 1)]
            cv_loss_columns.append("Loss_cv_Avg")
            cv_loss_columns.insert(0, "hpo_id")
            cv_acc_columns = ["Acc_cv" + str(fold) for fold in range(1, k + 1)]
            cv_acc_columns.append("Acc_cv_Avg")
            cv_acc_columns.insert(0, "hpo_id")
            
            cross_validation_loss_df = pd.DataFrame(columns=cv_loss_columns)
            cross_validation_acc_df = pd.DataFrame(columns=cv_acc_columns)
            test_loss_df = pd.DataFrame(columns=cv_loss_columns)
            test_acc_df = pd.DataFrame(columns=cv_acc_columns)
        
        # Load hyperparameter trials from "./hpo" folder
        hpo_file_path = load_hpo_path
        hpo_df = pd.read_csv(hpo_file_path)
        print(hpo_df)

        # Iterating through the dataframe as a dictionary is fast.
        
        # Keep a counter of completed trials so we can save the results in periodic intervals
        counter = 0
        # Define an error code in-case the optimization fails for a particular trial.
        # This allows the grid search to continue.
        err_write_row = ["ERR" for fold in range(k + 1)]
        err_write_row = dict(zip([fold for fold in range(k + 1)], err_write_row))
        
        # For each trial,
        for row in tqdm(hpo_df.to_dict(orient="records")):
            try:
                # Try to train
                scores = train_region_classifier(load_data_path=load_data_path, 
                                                 model_save_path=model_save_path, 
                                                 cross_validation=cross_validation, 
                                                 k=k, 
                                                 hpo_trial=row, 
                                                 random_state=random_state, 
                                                 verbose=verbose, 
                                                 log=log,
                                                 timestamp=timestamp)
                hpo_id = counter
                counter = counter + 1
            except:
                # If training failed, write the error code to the accumulated results
                scores = {"ERR": "ERR"}
                write_row = err_write_row
                hpo_id = counter
                counter = counter + 1
            if cross_validation:
                dir_head, dir_trial = os.path.split(load_hpo_path)
                cross_validation_loss_path = os.path.join(dir_head, "hpo_CVLoss_region_classifier.csv")
                cross_validation_acc_path = os.path.join(dir_head, "hpo_CVAcc_region_classifier.csv")
                testLossPath = os.path.join(dir_head, "hpo_TestLoss_region_classifier.csv")
                testAccPath = os.path.join(dir_head, "hpo_TestAcc_region_classifier.csv")
                for key, value in scores.items():
                    if value == "ERR":
                        cross_validation_loss_df.loc[len(cross_validation_loss_df)] = write_row
                        cross_validation_acc_df.loc[len(cross_validation_acc_df)] = write_row
                        test_loss_df.loc[len(test_loss_df)] = write_row
                        test_acc_df.loc[len(test_acc_df)] = write_row
                        break
                    average_val = np.array(value).mean()
                    write_row = value
                    write_row.append(average_val)
                    write_row.insert(0, hpo_id)
                    if key == "Val_Loss":
                        cross_validation_loss_df.loc[len(cross_validation_loss_df)] = write_row
                    elif key == "Val_Acc":
                        cross_validation_acc_df.loc[len(cross_validation_acc_df)] = write_row
                    elif key == "Test_Loss":
                        test_loss_df.loc[len(test_loss_df)] = write_row
                    elif key == "Test_Acc":
                        test_acc_df.loc[len(test_acc_df)] = write_row
                if counter % save_every == 0:
                    cross_validation_loss_df.to_csv(cross_validation_loss_path)
                    cross_validation_acc_df.to_csv(cross_validation_acc_path)
                    test_loss_df.to_csv(testLossPath)
                    test_acc_df.to_csv(testAccPath)
                if verbose:
                    print(row)
                    print(scores)
                    print("\n")
    # Legacy code
    else:
        # If cross-validating for each hyperparameter trial,
        if cross_validation:
            # Define dataframes to store cross_validation results
            cv_loss_columns = ["Loss_cv" + str(fold) for fold in range(1, k + 1)]
            cv_loss_columns.append("Loss_cv_Avg")
            cv_loss_columns.insert(0, "hpo_id")
            cv_acc_columns = ["Acc_cv" + str(fold) for fold in range(1, k + 1)]
            cv_acc_columns.append("Acc_cv_Avg")
            cv_acc_columns.insert(0, "hpo_id")
            cross_validation_loss_df = pd.DataFrame(columns=cv_loss_columns)
            cross_validation_acc_df = pd.DataFrame(columns=cv_acc_columns)
        
        # Load hyperparameter trials from "./hpo" folder
        hpo_file_path = os.path.join(load_hpo_path, "hpo_trials_region_classifier.csv")
        hpo_df = pd.read_csv(hpo_file_path)
        print(hpo_df)

        # Iterating through the dataframe as a dictionary is fast.
        
        # Keep a counter of completed trials so we can save the results in periodic intervals
        counter = 0
        # Define an error code in-case the optimization fails for a particular trial.
        # This allows the grid search to continue.
        err_write_row = ["ERR" for fold in range(k + 1)]
        err_write_row = dict(zip([fold for fold in range(k + 1)], err_write_row))
        
        # For each trial,
        for row in tqdm(hpo_df.to_dict(orient="records")):
            try:
                # Try to train
                scores = train_region_classifier(load_data_path=load_data_path, 
                                                 model_save_path=model_save_path, 
                                                 cross_validation=cross_validation, 
                                                 k=k, 
                                                 hpo_trial=row, 
                                                 random_state=random_state, 
                                                 verbose=False, 
                                                 log=False)
                hpo_id = counter
                counter = counter + 1
            except:
                # If training failed, write the error code to the accumulated results
                write_row = err_write_row
                hpo_id = counter
                counter = counter + 1
            if cross_validation:
                cross_validation_loss_path = os.path.join(load_hpo_path, "hpo_Loss_region_classifier.csv")
                cross_validation_acc_path = os.path.join(load_hpo_path, "hpo_Acc_region_classifier.csv")
                for key, value in scores.items():
                    if value == "ERR":
                        cross_validation_loss_df.loc[len(cross_validation_loss_df)] = write_row
                        cross_validation_acc_df.loc[len(cross_validation_acc_df)] = write_row
                        break
                    average_val = np.array(value).mean()
                    write_row = value
                    write_row.append(average_val)
                    write_row.insert(0, hpo_id)
                    if key == "Val_Loss":
                        cross_validation_loss_df.loc[len(cross_validation_loss_df)] = write_row
                    elif key == "Val_Acc":
                        cross_validation_acc_df.loc[len(cross_validation_acc_df)] = write_row
                if counter % save_every == 0:
                    cross_validation_loss_df.to_csv(cross_validation_loss_path)
                    cross_validation_acc_df.to_csv(cross_validation_acc_path)
                if verbose:
                    print(row)
                    print(scores)
                    print("\n")

def classify_regions(pipeline_inputs: dict = None, 
                     load_path=None, 
                     img_folder=None):
    # If we are using the control panel .ipynb pipeline,
    if pipeline_inputs is not None:
        # Get the parent directory of raw images for all codes
        raw_directory = pipeline_inputs["raw_directory"]
        # Get the list of codes to process from the control panel pipeline inputs 
        code_list = pipeline_inputs["code_list"]
        # For each code to process
        for code_num in tqdm(code_list):
            # Get the raw image directory for that code
            raw_code_dir = os.path.join(raw_directory, "code " + code_num)
            # Define positive/negative sample save paths
            pos_save_dir = os.path.join("data/classifier_training_samples/", "code " + code_num, "positive")
            neg_save_dir = os.path.join("data/classifier_training_samples/", "code " + code_num, "negative")
            if not os.path.exists(pos_save_dir):
                os.makedirs(pos_save_dir)
                os.makedirs(neg_save_dir)
            print(f"Processing and saving data to:\n{pos_save_dir}")
            print(f"Processing and saving data to:\n{neg_save_dir}")
            
            filenames = sorted(os.listdir(raw_code_dir))
            
            # Load MSER parameters for an initial division of positive/negative samples
            MSER_params = pipeline_inputs["MSER_params_per_code"][code_num]
            # For each set of MSER parameters, which corresponds to each reference image of each code,
            for MSERFile in tqdm(MSER_params):
                # These counters for positive/negative samples are counted per-reference image, not per-code
                sum_pos = 0
                sum_neg = 0
                raw_image_id = MSERFile.rstrip("_MSER.json")
                with open(os.path.join(raw_code_dir, MSERFile), "r") as MSERObj:
                    mser_dict = dict(json.load(MSERObj)["optimizer.max"]["params"])
                # Define a region detector for positive/negative division of samples
                # This region detector is not optimized to be the most accurate
                # Rather, we use the region detector as a means of initializing positive/negative samples.
                # Later, we will actually hyperparameter optimize the region detectors for accurate positive/negative sample separation
                region_detector = RegionDetector(MSER_parameters=mser_dict)
                # For each reference image in the raw image directory for a particular code,
                ref_name = raw_image_id + "_ref.tiff"
                
                holograms=[]
                # Ignore the file if it is not an image file
                if ".tiff" not in ref_name:
                    continue
                # If the file is a reference image,
                elif "ref" in ref_name:
                    # Load reference
                    reference = cv2.imread(os.path.join(raw_code_dir, ref_name), cv2.IMREAD_ANYDEPTH)
                    # For each image in the raw image directory of a particular code
                    for image_name in filenames:
                        # The raw image file is just the name of the reference file without the reference designation
                        code = ref_name.replace("_ref.tiff", "")
                        # If we found the code image that corresponds with its' own refernece image,
                        if code == image_name.replace(".tiff", "").split("_")[0] and ".tiff" in image_name and "ref" not in image_name:
                            # Read the raw code image
                            hologram = cv2.imread("{}/{}".format(raw_code_dir, image_name), cv2.IMREAD_ANYDEPTH)
                            hologram = hologram.astype(np.float32)
                            # Append the raw image data with the reference image data.
                            holograms.append((hologram, image_name.replace(".tiff", ""), reference))

                    for hologram in holograms:
                        holo, name, reference = hologram
                        save_img_name = "data/test/{}_{}_regions.png".format(code_num, name)
                        positive_regions, negative_regions = region_detector.detect_regions(holo,
                                                                                            reference,
                                                                                            save_img_name=save_img_name)
                        for i in range(len(positive_regions)):
                            file_path = "{}/{}_{}_positive_{}.png".format(pos_save_dir, code_num, raw_image_id, i+sum_pos)
                            region = positive_regions[i]
                            shape = region.shape
                            new_shape = (*shape[1:], shape[0])
                            region = region.reshape(new_shape)
                            cv2.imwrite(file_path, region)
                        sum_pos = sum_pos + len(positive_regions)

                        for i in range(len(negative_regions)):
                            file_path = "{}/{}_{}_negative_{}.png".format(neg_save_dir, code_num, raw_image_id, i+sum_neg)
                            region = negative_regions[i]
                            shape = region.shape
                            new_shape = (*shape[1:], shape[0])
                            region = region.reshape(new_shape)
                            cv2.imwrite(file_path, region)
                        sum_neg = sum_neg + len(negative_regions)
    # Legacy code
    else:
        if not os.path.exists("data/classifier_training_samples/positive"):
            os.makedirs("data/classifier_training_samples/positive")
            os.makedirs("data/classifier_training_samples/negative")
        # Optionally, load a trained classifier. Set this to None if you would like to initialize a new random model.
        # load_path = None # "data/best_region_classifier.pt"

        # Create an instance of the region detector object.
        region_detector = RegionDetector(model_load_path=load_path)

        # Load the hologram we want to examine, and the corresponding reference image.
        img_folder = img_folder

        folder_name = img_folder[img_folder.rfind("/"):]
        holograms = []

        for ref_name in os.listdir(img_folder):
            if ".tiff" not in ref_name:
                continue
            if "ref" in ref_name:
                print("referencing: ", ref_name)
                reference = cv2.imread("{}/{}".format(img_folder, ref_name), cv2.IMREAD_ANYDEPTH)

                for image_name in os.listdir(img_folder):
                    code = ref_name.replace("_ref.tiff", "")
                    if code == image_name.replace(".tiff", "").split("_")[0] and ".tiff" in image_name and "ref" not in image_name:
                        hologram = cv2.imread("{}/{}".format(img_folder, image_name), cv2.IMREAD_ANYDEPTH)
                        hologram = hologram.astype(np.float32)
                        holograms.append((hologram, image_name.replace(".tiff", ""), reference))

        # Use the region detector to find regions that contain objects we're interested in, and regions that don't.
        sum_pos = 0
        sum_neg = 0
        for hologram in holograms:
            holo, name, reference = hologram
            save_img_name = "data/test/{}_{}_regions.png".format(folder_name, name)
            positive_regions, negative_regions = region_detector.detect_regions(holo,
                                                                                reference,
                                                                                save_img_name=save_img_name)
            for i in range(len(positive_regions)):
                file_path = "data/classifier_training_samples/positive/{}_positive_{}.png".format(folder_name,i+sum_pos)
                region = positive_regions[i]
                shape = region.shape
                new_shape = (*shape[1:], shape[0])
                region = region.reshape(new_shape)
                cv2.imwrite(file_path, region)
            sum_pos = sum_pos + len(positive_regions)

            for i in range(len(negative_regions)):
                file_path = "data/classifier_training_samples/negative/{}_negative_{}.png".format(folder_name,i+sum_neg)
                region = negative_regions[i]
                shape = region.shape
                new_shape = (*shape[1:], shape[0])
                region = region.reshape(new_shape)
                cv2.imwrite(file_path, region)
            sum_neg = sum_neg + len(negative_regions)


def train_code_classifier(pipeline_inputs: dict = None,
                          timestamp: str = None,
                          hyper_dict: dict = None):
    
    if timestamp is None:
        timestamp = datetime.now().strftime("%m_%d_%y_%H:%M")

    # If we are using the control panel .ipynb pipeline,
    if pipeline_inputs is not None:
        
        # Timestamps for record-keeping
        if pipeline_inputs["timestamp"] is None:
            pipeline_inputs["timestamp"] = datetime.now().strftime("%m_%d_%y_%H:%M")
        
        codes = pipeline_inputs["code_list"]
        trainer = CodeClassifierTrainerGPU(codes, model_save_path=pipeline_inputs["model_save_parent_directory"])
        code_data_composite = []
        for code in codes:
            code_path = os.path.join(pipeline_inputs["sample_parent_directory"], "code " + code)
            code_data = load_code(code_folder_path=code_path)
            code_data_composite = code_data_composite + code_data
        
        # Total composite dataset samples
        # Only print verbosely if we are not hyperparameter optimizing
        if hyper_dict is None:
            print("Total Composite Dataset Training Samples:\n{}".format(len(code_data_composite)))

        # Based on the format of the return result of .load_code(),
        # Extract all the targets of the training samples
        targets = np.array(list(zip(*code_data_composite))[-1])
        # All the samples
        dataset = np.asarray(code_data_composite, dtype=object)

        # Do a straified train/test split of all samples into training and test datasets
        # Returns the actual samples, not the indices of the samples.
        training_data, validation_data = train_test_split(dataset, 
                                                     test_size=pipeline_inputs["test_size"], 
                                                     stratify=targets,
                                                     random_state=pipeline_inputs["strat_kfold"]["random_state"])
        # Train targets
        training_targets = np.asarray(list(zip(*training_data))[-1])

        # CG: Stratified k-Fold cross-validation
        # If doing strat. k-fold cross-val.,
        if pipeline_inputs["strat_kfold"]["activate"]:
            # Define a class to do the stratified splitting into folds
            splits = StratifiedKFold(n_splits=pipeline_inputs["strat_kfold"]["num_folds"], 
                                     shuffle=True, 
                                     random_state=pipeline_inputs["strat_kfold"]["random_state"])
            
            # Get the indices of the training dataset
            training_data_idx = np.arange(len(training_data))
            # Dictionary to hold cross-validation scores
            cross_val_scores = {"Val_Loss": [], "Val_Acc": [], "Test_Loss": [], "Test_Acc": []}
            fold_index = 1
            # For each fold in the training dataset, define a new training dataset and validation dataset based off the training targets
            for fold, (train_idx, val_idx) in enumerate(splits.split(training_data_idx, y=training_targets)):
                if pipeline_inputs["verbose"]:
                    print('\n\nFold {}'.format(fold + 1))
                # Define a code classifier
                # If we are not Bayesian optimizing,
                if hyper_dict is None:
                    trainer = CodeClassifierTrainerGPU(codes=codes,
                                                    model_save_path=pipeline_inputs["model_save_parent_directory"],
                                                    verbose=pipeline_inputs["verbose"],
                                                    log=pipeline_inputs["log"], 
                                                    timestamp=pipeline_inputs["timestamp"])
                # Else, if we are Bayesian optimizing,
                else:
                    trainer = CodeClassifierTrainerGPU(codes=codes,
                                                    model_save_path=pipeline_inputs["model_save_parent_directory"],
                                                    batch_size=hyper_dict["bs"],
                                                    lr=hyper_dict["lr"],
                                                    fcSize=hyper_dict["fcSize"],
                                                    fcNum=hyper_dict["fcNum"],
                                                    dropoutRate=hyper_dict["dr"],
                                                    verbose=pipeline_inputs["verbose"], 
                                                    log=pipeline_inputs["log"], 
                                                    timestamp=pipeline_inputs["timestamp"])
                # Load code classifier training data and validation data
                # Validation data is taken from the training dataset and targets (training_data, training_targets)
                # Test dataset and test targets are inputted separately
                trainer.load_data(pipeline_inputs["sample_parent_directory"], 
                                  training_data,
                                  training_targets, 
                                  train_idx, 
                                  val_idx, 
                                  test_dataset=validation_data)
                # Train
                cross_val_scores = trainer.train(cross_validation=pipeline_inputs["strat_kfold"]["activate"], 
                                               cross_val_scores=cross_val_scores)
                # Keep track of what k-fold we are on for book-keeping
                fold_index = fold_index + 1
            
            # Print out the average cross-validation results
            if pipeline_inputs["verbose"]:
                print("\nTRAINING COMPLETE.\nCross-Validation Dictionary:")
                print(cross_val_scores)
                for key, value in cross_val_scores.items():
                    print("Avg. " + str(key) + ": " + str(np.array(value).mean()))
            return cross_val_scores

def test_system(img_folder="/Users/apple/Dropbox (University of Michigan)/iMAPS_coding/Selected images for DL/1-1/1-10 (1)",
                region_detector_path="data/best/best_region_classifier.pt",
                code_classifier_path="data/best/best_code_classifier.pt"):

    codes = ["1", "2", "3"]
    code_map = {idx: code for idx, code in enumerate(codes)}

    img_folder = img_folder

    region_detector = RegionDetector(model_load_path=region_detector_path)
    code_classifier = CodeClassifier(len(codes), model_load_path=code_classifier_path)

    reference = None
    for file_name in os.listdir(img_folder):
        if "ref" in file_name:
            print("referencing: ", file_name)
            reference = cv2.imread("{}/{}".format(img_folder, file_name), cv2.IMREAD_ANYDEPTH)
            break

    for file_name in os.listdir(img_folder):
        if "ref" in file_name or ".tiff" not in file_name:
            continue

        img = cv2.imread("{}/{}".format(img_folder, file_name), cv2.IMREAD_ANYDEPTH)
        regions, _ = region_detector.detect_regions(img, reference)
        classes = code_classifier.classify_regions(regions)
        counts = {key: 0 for key in codes}

        for cls in classes:
            counts[code_map[cls.item()]] += 1

        print("/n{}".format(file_name))
        for key, value in counts.items():
            print("{}: {}".format(key, value))


def get_intensity(img_folder="/Users/apple/Dropbox (University of Michigan)/iMAPS_coding/Selected images for DL/1-1/1-10 (1)",
                  region_detector_path="data/best/best_region_classifier.pt"):

    region_detector = RegionDetector(model_load_path=region_detector_path)
    folder_name = img_folder[img_folder.rfind("/")+1:]
    holograms = []

    for ref_name in os.listdir(img_folder):
        if ".tiff" not in ref_name:
            continue
        if "ref" in ref_name:
            print("referencing: ", ref_name)
            reference = cv2.imread("{}/{}".format(img_folder, ref_name), cv2.IMREAD_ANYDEPTH)

            for image_name in os.listdir(img_folder):
                code = ref_name.replace("_ref.tiff", "")
                if code == image_name.replace(".tiff", "").split("_")[0] and ".tiff" in image_name and "ref" not in image_name:
                    hologram = cv2.imread("{}/{}".format(img_folder, image_name), cv2.IMREAD_ANYDEPTH)
                    hologram = hologram.astype(np.float32)
                    holograms.append((hologram, image_name.replace(".tiff", ""), reference))

    intensities = []
    file_names = []
    # Find intensities
    for hologram in holograms:
        holo, name, reference = hologram
        if not os.path.exists("data/hulls"):
            os.makedirs("data/hulls")
        save_img_name = "data/hulls/{}_{}_regions.png".format(folder_name, name)
        print("processing", name)
        intensity = region_detector.get_intensity(holo, reference, save_img_name=save_img_name)
        intensities.append(intensity)
        file_names += [folder_name + '/' + name, '', '']


    # write to a csv file
    intensities = list(itertools.zip_longest(*intensities, fillvalue=['', '', '']))
    intensities = [list(itertools.chain(*x)) for x in intensities]
    
    intensities = [file_names] + [['x', 'y', 'intensity']  * int(int(len(file_names))/3)] + intensities

    if not os.path.exists("data/intensities"):
        os.makedirs("data/intensities")
    csv_path="data/intensities/" + folder_name + ".csv"
    with open(csv_path, 'w') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.write_rows(intensities)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--action', type=str, default="classify_regions",
                        help='choose one of: find_mser_params, train_region_classifier, hpo_region_classifier, classify_regions, train_code_classifier, test_system, get_intensity')
    parser.add_argument('--dir', type=str,
                        default="/home/cameron/Dropbox (University of Michigan)/DL_training/data/sandbox_CG/raw/Gear_particle/code1/1",
                        help='raw image dir')
    parser.add_argument('--reg_path', type=str, default=None,
                        help='load region detector path')
    parser.add_argument('--code_path', type=str, default="data/best/best_code_classifier.pt",
                        help='load code classifier path')
    parser.add_argument('--pipeline_inputs', type=str, default=None, help="Load JSON input variable dictionary to run various pipeline tasks\n(e.g., optimizing MSER parameters)")

    # print("Fds")
    args = parser.parse_args()
    # action = args.action
    # dir = args.dir
    # reg_path = args.reg_path
    # code_path = args.code_path
    # pipeline_inputs = args.pipeline_inputs

    action = "find_mser_params"
    pipeline_inputs = {
                       "number_iterations":1000,
                       "raw_directory":"C:/Users/jane/Desktop/particle_location_jsons",
                       "mser_save_directory":"C:/Users/jane/Desktop/particle_location_jsons/mser_hyperparameters",
                       "code_list":["1"]
                       }

    # Controls the functionality of the pipeline Jupter notebooks
    # -----------------------------------------------------------------
    # if args.pipeline_inputs is not None:
    #     # Loads the inputs to the pipeline if specified
    #     with open(pipeline_inputs, "r") as inputFile:
    #         pipeline_inputs = json.load(inputFile)
    # ------------------------------------------------------------------
    
    if action == 'find_mser_params':
        find_mser_params(pipeline_inputs=pipeline_inputs)
    
    elif action == 'train_region_classifier':
        train_region_classifier(pipeline_inputs=pipeline_inputs)
    
    elif action == 'hpo_region_classifier':
        grid_search_region_classifier(pipeline_inputs=pipeline_inputs)
    
    elif action == 'classify_regions':
        classify_regions(pipeline_inputs=pipeline_inputs)
    
    elif action == 'train_code_classifier':
        train_code_classifier(pipeline_inputs=pipeline_inputs)

    elif action == "bayesian_optimize":
        # Currently only implemented for the Jupyter notebook pipeline,
        if pipeline_inputs is not None:
            # Create an OpTuna study, maximize the accuracy
            study = optuna.create_study(direction="maximize")
            # By default, OpTuna objective functions for objective minimization/maximization does not accept custom input variables
            # However, we can easily accomodate custom input variables in this way with some lambda operations,
            objective_with_custom_input = lambda trial: objective(trial, pipeline_inputs)
            
            study = checkpoint_study(study,
                     objective_with_custom_input,
                     num_trials=pipeline_inputs["num_hpo"],
                     checkpoint_every=pipeline_inputs["save_every"],
                     checkpoint_path=os.path.join(pipeline_inputs["checkpoint_path"], pipeline_inputs["timestamp"]))
            
            '''# Optimize the study
            study.optimize(objective_with_custom_input, 
                           n_trials=pipeline_inputs["num_hpo"], 
                           timeout=pipeline_inputs["timeout"],
                           callbacks=[pruning_callback])'''

            # Get the pruned trials (trials pruned prematurely)
            pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
            # Get the completed trials
            complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

            # Summarize
            print("\n\nStudy statistics: ")
            print("  Number of finished trials: ", len(study.trials))
            print("  Number of pruned trials: ", len(pruned_trials))
            print("  Number of complete trials: ", len(complete_trials))

            print("Best trial:")
            trial = study.best_trial

            print("  Value: ", trial.value)

            print("  Params: ")
            for key, value in trial.params.items():
                print("    {}: {}".format(key, value))
    else:
      print("invalid action")