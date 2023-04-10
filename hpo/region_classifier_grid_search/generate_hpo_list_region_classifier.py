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

def generate_grid_search_file(choiceList, choiceLabels, choiceType, hyperparameterLists, savePath = os.getcwd()):
    '''Generate all possible combinations of hyperparameters given.
    choiceList: A list of string values. Holds the user inputted hyperparameters temporarily.
    choiceLabels: A list of each hyperparameter name. Parallel with choiceList
    choiceType: A list of each datatype intended for each hyperparameter. Parallel with choiceList.
    hyperparameterLists: A list of lists where each sub-list are the hyperparameter values to try.
    savePath: Where to save the hyperparameter trials
    
    Returns: hpoTrialsDF, a dataframe of all hyperparameter combinations to try.
    '''

    # Assert parallel lists are same length
    assert len(choiceList) == len(choiceLabels)
    assert len(choiceList) == len(hyperparameterLists)
    
    # For each hyperparameter
    for choiceInd in range(len(choiceList)):
        # If we are not done selecting values for a particular hyperparameter,
        while choiceList[choiceInd] != "DONE":
            # Enter a value
            choiceList[choiceInd] = input(choiceLabels[choiceInd] + "\nEnter 'DONE' to exit.\n")
            assert type(choiceList[choiceInd]) is str
            
            # If we are not done selecting values for a particular hyperparameter,
            if choiceList[choiceInd] != "DONE":
                # If the value is intended to be a float,
                if choiceType[choiceInd] == float:
                    converted = float(choiceList[choiceInd])
                # If the value is intended to be an int,
                elif choiceType[choiceInd] == int:
                    converted = int(choiceList[choiceInd])
                # Otherwise, raise an exception because of unintended behavior
                else:
                    raise Exception("An unprogrammed datatype for hyperparameter " + choiceLabels[choiceInd] + " was specified. See choiceType variable.")
                assert converted is float or int
                
                # Add the inputted hyperparameter value to the list of values to try for
                # the given hyperparameter.
                hyperparameterLists[choiceInd].append(converted)

            print("\n")
    
    # Ensure that we have suggested values for all the hyperparameters asked
    emptyCounter = 0
    for choiceInd in range(len(choiceList)):
        if len(hyperparameterLists[choiceInd]) == 0:
            emptyCounter+=1

    assert emptyCounter == 0

    # Get all combinations of hyperparameters asked
    hpoTrials = list(product(*hyperparameterLists))
    print("\nHyperparameter Trials Calculated.\nSaving to: " + savePath)
    
    print(len(hpoTrials))

    # Convert to PANDAS dataframe and save
    hpoTrialsDF = pd.DataFrame(hpoTrials, columns=choiceLabels).sample(frac=1).reset_index(drop=True)
    hpoTrialsID = [i for i in range(hpoTrialsDF.shape[0])]
    hpoTrialsDF.insert(0, "hpoID", hpoTrialsID)
    hpoTrialsDF.to_csv(osp.join(savePath, "hpo_trials_region_classifier.csv"), index=False)
    print(hpoTrialsDF)
    return hpoTrialsDF

# Current directory
curDir = os.getcwd()

batchSizeList = [64, 128, 192, 256]
lrList = [3e-4, 3e-5, 3e-6]
dropoutList = [0.3, 0.5, 0.7]
weightDecayBeta1List = [0.8, 0.9, 0.95]
epsilonList = [1e-8, 0.1, 1.0]
fcSize = [64, 128, 192, 256]

batchSizeChoice = "initialized"
lrChoice = "initialized"
dropoutChoice = "initialized"
weightDecayBeta1Choice = "initialized"
epsilonChoice = "initialized"
fcChoice = "initialized"

choiceList = [batchSizeChoice, lrChoice, dropoutChoice, weightDecayBeta1Choice, epsilonChoice, fcChoice]
choiceLabels = ["Batch_Size", "lr", "Dropout_Rate", "Weight_Decay_Beta1", "Epsilon", "FC_Size"]
choiceType = [int, float, float, float, float, int]
hyperparameterLists = [batchSizeList, lrList, dropoutList, weightDecayBeta1List, epsilonList, fcSize]

hpoTrialsDF = generate_grid_search_file(choiceList, choiceLabels, choiceType, hyperparameterLists)
print(hpoTrialsDF)
