import argparse
import json
from utils import (
    action_functions,
)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--action',
        type=str,
        default='classify_regions',
        help='choose one of: find_mser_params, train_region_classifier, hpo_region_classifier, classify_regions, train_code_classifier, test_system, get_intensity',
    )
    parser.add_argument(
        '--pipeline_inputs',
        type=str,
        default=None,
        help='Load JSON input variable dictionary to run various pipeline tasks\n(e.g., optimizing MSER parameters)',
    )

    # print("Fds")
    args = parser.parse_args()
    action = args.action
    pipeline_inputs = args.pipeline_inputs

    # action = 'find_mser_params'
    # pipeline_inputs = {
    #    'number_iterations': 1000,
    #    'raw_directory': 'C:/Users/jane/Desktop/particle_location_jsons',
    #    'mser_save_directory': 'C:/Users/jane/Desktop/particle_location_jsons/mser_hyperparameters',
    #    'code_list': ['1'],
    # }

    # Controls the functionality of the pipeline Jupter notebooks
    # -----------------------------------------------------------------
    if args.pipeline_inputs is not None:
        # Loads the inputs to the pipeline if specified
        with open(pipeline_inputs, 'r') as inputFile:
            pipeline_inputs = json.load(inputFile)
    # ------------------------------------------------------------------

    if action == 'bayesian_optimize_code_classifier':
        action_functions.bayesian_optimize_code_classifer(
            pipeline_inputs=pipeline_inputs)

    elif action == 'classify_regions':
        action_functions.classify_regions(pipeline_inputs=pipeline_inputs)

    elif action == 'find_mser_params':
        action_functions.find_mser_params(pipeline_inputs=pipeline_inputs)

    elif action == 'train_region_classifier':
        action_functions.train_region_classifier(
            pipeline_inputs=pipeline_inputs)

    elif action == 'train_code_classifier':
        action_functions.train_code_classifier(pipeline_inputs=pipeline_inputs)

    else:
        print('invalid action')
