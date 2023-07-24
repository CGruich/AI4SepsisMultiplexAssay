from model_training import (
    RegionClassifierTrainerGPU,
)
from object_detection import RegionDetector
from code_classification import CodeClassifier
from sklearn.model_selection import StratifiedKFold, train_test_split
from tqdm import tqdm
import cv2
import numpy as np
import pandas as pd
import os
from datetime import datetime

from helper_functions import load_data


def train_region_classifier(
    pipeline_inputs: dict = None,
    load_data_path='data/classifier_training_samples',
    model_save_path='data/models/region',
    cross_validate=False,
    k=5,
    hpo_trial=None,
    random_state=100,
    verbose=True,
    log=True,
    timestamp: str = None,
):
    if timestamp is None:
        timestamp = datetime.now().strftime('%m_%d_%y_%H:%M')

    # Train a new classifier with the data located under
    # data/classifier_training_samples/positive and
    # data/classifier_training_samples/negative

    # Returns list of lists
    data_list = load_data(load_data_path, verbose=verbose)
    # Based on the format of the return result of .load_data(),
    # Extract all the targets of the training samples
    targets = np.array(list(zip(*data_list))[-1])
    # All the samples
    dataset = np.asarray(
        load_data(load_data_path, verbose=verbose), dtype=object)

    # Do a stratified train/test split of all samples into training and test datasets
    # Returns the actual samples, not the indices of the samples.
    training_data, validation_data = train_test_split(
        dataset, test_size=0.20, stratify=targets
    )
    training_targets = np.asarray(list(zip(*training_data))[-1])

    # CG: Stratified k-Fold cross-validation
    if cross_validate:
        # Object for stratified k-fold cross-validation splitting of training dataset into a new training dataset and validation dataset
        splits = StratifiedKFold(
            n_splits=k, shuffle=True, random_state=random_state)

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
                hpo_trial=hpo_trial,
                verbose=verbose,
                log=log,
                timestamp=timestamp,
                k=fold_index,
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
                cross_validate=cross_validate, cross_val_scores=cross_val_scores
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
    # If not cross-validating,
    else:
        trainer = RegionClassifierTrainerGPU(
            model_save_path=model_save_path, hpo_trial=hpo_trial, verbose=verbose, log=log,
        )
        trainer.load_data(load_data_path, dataset,
                          train_idx=None, val_idx=None)
        trainer.train(cross_validate=False, cross_val_scores=None)


def grid_search_region_classifier(
    pipeline_inputs: dict = None,
    load_hpo_path='/home/cameron/Dropbox (University of Michigan)/DL_training/hpo/region_classifier_grid_search',
    load_data_path='data/classifier_training_samples',
    model_save_path='data/models/region',
    cross_validation=True,
    k=5,
    random_state=100,
    verbose=False,
    save_every=50,
):
    if pipeline_inputs is not None:
        load_hpo_path = pipeline_inputs.get(
            'grid_search_hpo', None).get('hpo_file', None)
        load_data_path = pipeline_inputs.get('sample_parent_directory', None)
        model_save_path = pipeline_inputs.get(
            'model_save_parent_directory', None)
        cross_validation = pipeline_inputs.get(
            'strat_kfold', None).get('activate', None)
        k = pipeline_inputs.get('strat_kfold', None).get('num_folds', None)
        random_state = pipeline_inputs.get(
            'strat_kfold', None).get('random_state', None)
        save_every = pipeline_inputs.get(
            'grid_search_hpo', None).get('save_every', None)
        log = pipeline_inputs.get('grid_search_hpo', None).get('log', None)
        verbose = pipeline_inputs.get(
            'grid_search_hpo', None).get('verbose', None)
        timestamp = pipeline_inputs.get('grid_search_hpo', None).get(
            'hpo_timestamp', datetime.now().strftime('%m_%d_%y_%H:%M')
        )

        # Ensuring proper pipeline inputs to minimize headaches
        assert (
            load_hpo_path is not None
            and type(load_hpo_path) is str
            and '.csv' in load_hpo_path
        )
        assert load_data_path is not None and type(load_data_path) is str
        assert model_save_path is not None and type(model_save_path) is str
        assert cross_validation is not None and type(cross_validation) is bool
        assert k is not None and type(k) is int and k >= 1
        assert random_state is not None and type(random_state) is int
        assert save_every is not None and type(
            save_every) is int and save_every >= 1
        assert log is not None and type(log) is bool
        assert verbose is not None and type(verbose) is bool
        assert timestamp is not None

        # If cross-validating for each hyperparameter trial,
        if cross_validation:
            # Define dataframes to store cross_validation results
            cv_loss_columns = ['Loss_cv' + str(fold)
                               for fold in range(1, k + 1)]
            cv_loss_columns.append('Loss_cv_Avg')
            cv_loss_columns.insert(0, 'hpo_id')
            cv_acc_columns = ['Acc_cv' + str(fold) for fold in range(1, k + 1)]
            cv_acc_columns.append('Acc_cv_Avg')
            cv_acc_columns.insert(0, 'hpo_id')

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
        err_write_row = ['ERR' for fold in range(k + 1)]
        err_write_row = dict(
            zip([fold for fold in range(k + 1)], err_write_row))

        # For each trial,
        for row in tqdm(hpo_df.to_dict(orient='records')):
            try:
                # Try to train
                scores = train_region_classifier(
                    load_data_path=load_data_path,
                    model_save_path=model_save_path,
                    cross_validation=cross_validation,
                    k=k,
                    hpo_trial=row,
                    random_state=random_state,
                    verbose=verbose,
                    log=log,
                    timestamp=timestamp,
                )
                hpo_id = counter
                counter = counter + 1
            except:
                # If training failed, write the error code to the accumulated results
                scores = {'ERR': 'ERR'}
                write_row = err_write_row
                hpo_id = counter
                counter = counter + 1
            if cross_validation:
                dir_head, dir_trial = os.path.split(load_hpo_path)
                cross_validation_loss_path = os.path.join(
                    dir_head, 'hpo_CVLoss_region_classifier.csv'
                )
                cross_validation_acc_path = os.path.join(
                    dir_head, 'hpo_CVAcc_region_classifier.csv'
                )
                testLossPath = os.path.join(
                    dir_head, 'hpo_TestLoss_region_classifier.csv')
                testAccPath = os.path.join(
                    dir_head, 'hpo_TestAcc_region_classifier.csv')
                for key, value in scores.items():
                    if value == 'ERR':
                        cross_validation_loss_df.loc[len(
                            cross_validation_loss_df)] = write_row
                        cross_validation_acc_df.loc[len(
                            cross_validation_acc_df)] = write_row
                        test_loss_df.loc[len(test_loss_df)] = write_row
                        test_acc_df.loc[len(test_acc_df)] = write_row
                        break
                    average_val = np.array(value).mean()
                    write_row = value
                    write_row.append(average_val)
                    write_row.insert(0, hpo_id)
                    if key == 'Val_Loss':
                        cross_validation_loss_df.loc[len(
                            cross_validation_loss_df)] = write_row
                    elif key == 'Val_Acc':
                        cross_validation_acc_df.loc[len(
                            cross_validation_acc_df)] = write_row
                    elif key == 'Test_Loss':
                        test_loss_df.loc[len(test_loss_df)] = write_row
                    elif key == 'Test_Acc':
                        test_acc_df.loc[len(test_acc_df)] = write_row
                if counter % save_every == 0:
                    cross_validation_loss_df.to_csv(cross_validation_loss_path)
                    cross_validation_acc_df.to_csv(cross_validation_acc_path)
                    test_loss_df.to_csv(testLossPath)
                    test_acc_df.to_csv(testAccPath)
                if verbose:
                    print(row)
                    print(scores)
                    print('\n')
    # Legacy code
    else:
        # If cross-validating for each hyperparameter trial,
        if cross_validation:
            # Define dataframes to store cross_validation results
            cv_loss_columns = ['Loss_cv' + str(fold)
                               for fold in range(1, k + 1)]
            cv_loss_columns.append('Loss_cv_Avg')
            cv_loss_columns.insert(0, 'hpo_id')
            cv_acc_columns = ['Acc_cv' + str(fold) for fold in range(1, k + 1)]
            cv_acc_columns.append('Acc_cv_Avg')
            cv_acc_columns.insert(0, 'hpo_id')
            cross_validation_loss_df = pd.DataFrame(columns=cv_loss_columns)
            cross_validation_acc_df = pd.DataFrame(columns=cv_acc_columns)

        # Load hyperparameter trials from "./hpo" folder
        hpo_file_path = os.path.join(
            load_hpo_path, 'hpo_trials_region_classifier.csv')
        hpo_df = pd.read_csv(hpo_file_path)
        print(hpo_df)

        # Iterating through the dataframe as a dictionary is fast.

        # Keep a counter of completed trials so we can save the results in periodic intervals
        counter = 0
        # Define an error code in-case the optimization fails for a particular trial.
        # This allows the grid search to continue.
        err_write_row = ['ERR' for fold in range(k + 1)]
        err_write_row = dict(
            zip([fold for fold in range(k + 1)], err_write_row))

        # For each trial,
        for row in tqdm(hpo_df.to_dict(orient='records')):
            try:
                # Try to train
                scores = train_region_classifier(
                    load_data_path=load_data_path,
                    model_save_path=model_save_path,
                    cross_validation=cross_validation,
                    k=k,
                    hpo_trial=row,
                    random_state=random_state,
                    verbose=False,
                    log=False,
                )
                hpo_id = counter
                counter = counter + 1
            except:
                # If training failed, write the error code to the accumulated results
                write_row = err_write_row
                hpo_id = counter
                counter = counter + 1
            if cross_validation:
                cross_validation_loss_path = os.path.join(
                    load_hpo_path, 'hpo_Loss_region_classifier.csv'
                )
                cross_validation_acc_path = os.path.join(
                    load_hpo_path, 'hpo_Acc_region_classifier.csv'
                )
                for key, value in scores.items():
                    if value == 'ERR':
                        cross_validation_loss_df.loc[len(
                            cross_validation_loss_df)] = write_row
                        cross_validation_acc_df.loc[len(
                            cross_validation_acc_df)] = write_row
                        break
                    average_val = np.array(value).mean()
                    write_row = value
                    write_row.append(average_val)
                    write_row.insert(0, hpo_id)
                    if key == 'Val_Loss':
                        cross_validation_loss_df.loc[len(
                            cross_validation_loss_df)] = write_row
                    elif key == 'Val_Acc':
                        cross_validation_acc_df.loc[len(
                            cross_validation_acc_df)] = write_row
                if counter % save_every == 0:
                    cross_validation_loss_df.to_csv(cross_validation_loss_path)
                    cross_validation_acc_df.to_csv(cross_validation_acc_path)
                if verbose:
                    print(row)
                    print(scores)
                    print('\n')


def test_system(
    img_folder='/Users/apple/Dropbox (University of Michigan)/iMAPS_coding/Selected images for DL/1-1/1-10 (1)',
    region_detector_path='data/best/best_region_classifier.pt',
    code_classifier_path='data/best/best_code_classifier.pt',
):
    codes = ['1', '2', '3']
    code_map = {idx: code for idx, code in enumerate(codes)}

    img_folder = img_folder

    region_detector = RegionDetector(model_load_path=region_detector_path)
    code_classifier = CodeClassifier(
        len(codes), model_load_path=code_classifier_path)

    reference = None
    for file_name in os.listdir(img_folder):
        if 'ref' in file_name:
            print('referencing: ', file_name)
            reference = cv2.imread(
                '{}/{}'.format(img_folder, file_name), cv2.IMREAD_ANYDEPTH)
            break

    for file_name in os.listdir(img_folder):
        if 'ref' in file_name or '.tiff' not in file_name:
            continue

        img = cv2.imread('{}/{}'.format(img_folder, file_name),
                         cv2.IMREAD_ANYDEPTH)
        regions, _ = region_detector.detect_regions(img, reference)
        classes = code_classifier.classify_regions(regions)
        counts = {key: 0 for key in codes}

        for cls in classes:
            counts[code_map[cls.item()]] += 1

        print('/n{}'.format(file_name))
        for key, value in counts.items():
            print('{}: {}'.format(key, value))
