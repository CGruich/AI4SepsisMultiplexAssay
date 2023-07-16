# For getting directory names
import os
import os.path as osp

# For executing bash statements
import subprocess

# For keeping timestamps of the archive
import datetime

# For generating combinations of hyperparameters
from itertools import product

# Saving files
import pandas as pd


def generate_grid_search_file(
    choice_list, choice_labels, choice_type, hyperparameter_lists, save_path=os.getcwd()
):
    """Generate all possible combinations of hyperparameters given.
    choice_list: A list of string values. Holds the user inputted hyperparameters temporarily.
    choice_labels: A list of each hyperparameter name. Parallel with choice_list
    choice_type: A list of each datatype intended for each hyperparameter. Parallel with choice_list.
    hyperparameter_lists: A list of lists where each sub-list are the hyperparameter values to try.
    save_path: Where to save the hyperparameter trials

    Returns: hpo_trials_df, a dataframe of all hyperparameter combinations to try.
    """

    # Assert parallel lists are same length
    assert len(choice_list) == len(choice_labels)
    assert len(choice_list) == len(hyperparameter_lists)

    # For each hyperparameter
    for choice_index in range(len(choice_list)):
        # If we are not done selecting values for a particular hyperparameter,
        while choice_list[choice_index] != "DONE":
            # Enter a value
            choice_list[choice_index] = input(
                choice_labels[choice_index] + "\nEnter 'DONE' to exit.\n"
            )
            assert type(choice_list[choice_index]) is str

            # If we are not done selecting values for a particular hyperparameter,
            if choice_list[choice_index] != "DONE":
                # If the value is intended to be a float,
                if choice_type[choice_index] == float:
                    converted = float(choice_list[choice_index])
                # If the value is intended to be an int,
                elif choice_type[choice_index] == int:
                    converted = int(choice_list[choice_index])
                # Otherwise, raise an exception because of unintended behavior
                else:
                    raise Exception(
                        "An unprogrammed datatype for hyperparameter "
                        + choice_labels[choice_index]
                        + " was specified. See choice_type variable."
                    )
                assert converted is float or int

                # Add the inputted hyperparameter value to the list of values to try for
                # the given hyperparameter.
                hyperparameter_lists[choice_index].append(converted)

            print("\n")

    # Ensure that we have suggested values for all the hyperparameters asked
    empty_counter = 0
    for choice_index in range(len(choice_list)):
        if len(hyperparameter_lists[choice_index]) == 0:
            empty_counter += 1

    assert empty_counter == 0

    # Get all combinations of hyperparameters asked
    hpo_trials = list(product(*hyperparameter_lists))
    print("\nHyperparameter Trials Calculated.\nSaving to: " + save_path)

    print(len(hpo_trials))

    # Convert to PANDAS dataframe and save
    hpo_trials_df = (
        pd.DataFrame(hpo_trials, columns=choice_labels)
        .sample(frac=1)
        .reset_index(drop=True)
    )
    hpo_trials_id = [i for i in range(hpo_trials_df.shape[0])]
    hpo_trials_df.insert(0, "hpoID", hpo_trials_id)
    hpo_trials_df.to_csv(
        osp.join(save_path, "hpo_trials_region_classifier.csv"), index=False
    )
    print(hpo_trials_df)
    return hpo_trials_df


# Current directory
current_dir = os.getcwd()

batch_size_list = [64, 128, 192, 256]
lr_list = [3e-4, 3e-5, 3e-6]
dropout_list = [0.3, 0.5, 0.7]
weight_decay_beta1_list = [0.8, 0.9, 0.95]
epsilon_list = [1e-8, 0.1, 1.0]
fc_size = [64, 128, 192, 256]

batch_size_choice = "initialized"
lr_choice = "initialized"
dropout_choice = "initialized"
weight_decay_beta1_choice = "initialized"
epsilon_choice = "initialized"
fc_choice = "initialized"

choice_list = [
    batch_size_choice,
    lr_choice,
    dropout_choice,
    weight_decay_beta1_choice,
    epsilon_choice,
    fc_choice,
]
choice_labels = [
    "Batch_Size",
    "lr",
    "Dropout_Rate",
    "Weight_Decay_Beta1",
    "Epsilon",
    "FC_Size",
]
choice_type = [int, float, float, float, float, int]
hyperparameter_lists = [
    batch_size_list,
    lr_list,
    dropout_list,
    weight_decay_beta1_list,
    epsilon_list,
    fc_size,
]

hpo_trials_df = generate_grid_search_file(
    choice_list, choice_labels, choice_type, hyperparameter_lists
)
print(hpo_trials_df)
