import os
# Hyperparameter optimization
import optuna
from optuna.trial import TrialState
import numpy as np
import pickle


def bayesian_optimize_code_classifer(pipeline_inputs: dict = None):
    # Currently only implemented for the Jupyter notebook pipeline,
    assert pipeline_inputs is not None
    # Create an OpTuna study, maximize the accuracy
    study = optuna.create_study(direction='maximize')
    # By default, OpTuna objective functions for objective minimization/maximization does not accept custom input variables
    # However, we can easily accomodate custom input variables in this way with some lambda operations,

    def objective_with_custom_input(trial):
        return objective_code_classifier(trial, pipeline_inputs)

    study = checkpoint_study(
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
    pruned_trials = study.get_trials(
        deepcopy=False, states=[TrialState.PRUNED])
    # Get the completed trials
    complete_trials = study.get_trials(
        deepcopy=False, states=[TrialState.COMPLETE])

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

# Objective function for Bayesian optimization with OpTuna


def objective_code_classifier(trial, pipeline_inputs: dict = None):
    # Learning rate
    lr = trial.suggest_float('learning_rate', 1e-8, 1e-2)
    # Batch size
    bs = trial.suggest_int('batch_size', 128, 1024, 64)
    # Fully connected layer size
    fcSize = trial.suggest_int('fully_connected_size', 128, 1024, 64)
    # Number of fully connected layers
    fcNum = trial.suggest_int('fully_connected_layers', 1, 5, 1)
    # Dropout rate
    dr = trial.suggest_float('dropout_rate', 0.0, 0.8)

    # Dictionary of hyperparameters
    hyper_dict = {'lr': lr, 'bs': bs,
                  'fcSize': fcSize, 'fcNum': fcNum, 'dr': dr}

    # Run stratified k-fold cross-validation with the hyperparameters
    # Via the pipeline functionality of the workflow,
    cross_val_scores = train_code_classifier(
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
