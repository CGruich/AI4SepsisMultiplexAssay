import os

# Hyperparameter optimization
import optuna
import numpy as np
import pickle
from . import action_functions

# Objective function for Bayesian optimization with OpTuna


def objective_code_classifier(trial, pipeline_inputs: dict = None):
    # Learning rate
    lr = trial.suggest_float('learning_rate', 1e-8, 1e-2)
    # Batch size
    bs = trial.suggest_int('batch_size', 128, 1024, 64)
    # Fully connected layer size
    fc_size = trial.suggest_int('fully_connected_size', 128, 1024, 64)
    # Number of fully connected layers
    fc_num = trial.suggest_int('fully_connected_layers', 1, 5, 1)
    # Dropout rate
    dr = trial.suggest_float('dropout_rate', 0.0, 0.8)

    # Dictionary of hyperparameters
    hyper_dict = {'lr': lr, 'bs': bs, 'fc_size': fc_size, 'fc_num': fc_num, 'dr': dr}

    # Run stratified k-fold cross-validation with the hyperparameters
    # Via the pipeline functionality of the workflow,
    cross_val_scores = action_functions.train_code_classifier(
        pipeline_inputs=pipeline_inputs, timestamp=None, hyper_dict=hyper_dict
    )

    # Average stratified k-fold cross-validation accuracy
    avg_val_accuracy = np.array(cross_val_scores['Val_Acc']).mean()

    # Return this accuracy, which we rely on for the Bayesian loop
    return avg_val_accuracy


def objective_region_classifier(trial, pipeline_inputs: dict = None):
    # Learning rate
    lr = trial.suggest_float('learning_rate', 1e-8, 1e-2)
    # Batch size
    bs = trial.suggest_int('batch_size', 128, 1024, 64)
    # Fully connected layer size
    fc_size = trial.suggest_int('fully_connected_size', 128, 1024, 64)
    # Number of fully connected layers
    fc_num = trial.suggest_int('fully_connected_layers', 1, 5, 1)
    # Dropout rate
    dr = trial.suggest_float('dropout_rate', 0.0, 0.8)

    # Dictionary of hyperparameters
    hyper_dict = {'lr': lr, 'bs': bs, 'fc_size': fc_size, 'fc_num': fc_num, 'dr': dr}

    # Run stratified k-fold cross-validation with the hyperparameters
    # Via the pipeline functionality of the workflow,
    cross_val_scores = action_functions.train_code_classifier(
        pipeline_inputs=pipeline_inputs, timestamp=None, hyper_dict=hyper_dict
    )

    # Average stratified k-fold cross-validation accuracy
    avg_val_accuracy = np.array(cross_val_scores['Val_Acc']).mean()

    # Return this accuracy, which we rely on for the Bayesian loop
    return avg_val_accuracy


# Define a function that we can use to restart the optimization from the last trial.
# This is useful if we try a high-throughput amount of trials and don't want to start over after a crash, for example


def checkpoint_study(
    study: optuna.study.Study,
    objective_function=None,
    num_trials: int = None,
    checkpoint_every: int = 100,
    checkpoint_path: str = None,
):
    for trial in range(num_trials):
        # Optimize a single trial
        study.optimize(objective_function, n_trials=1)

        # Checkpoint every checkpoint_every trials
        if (trial + 1) % checkpoint_every == 0:
            # Make a directory to save checkpoints if not exist
            if not os.path.exists(checkpoint_path):
                os.makedirs(checkpoint_path)
            # Pickle the study to a file
            ckptFile = f'ckpt_{trial + 1}.pkl'
            with open(os.path.join(checkpoint_path, ckptFile), 'wb') as fileObj:
                pickle.dump(study, fileObj)
    return study
