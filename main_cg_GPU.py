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
import re
import os.path as osp
from pathlib import Path
import json
from datetime import datetime

def load_data(folder_path, verbose=True):
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

# Tests if an arbitrary string can be converted into an integer...
# Helper function for explode
def test_if_int(inputStr: str):
    try:
        return int(inputStr)
    except:
        return inputStr

# Defines a key for sorting a list of filenames.
# e.g., ["File_1_Example.ext", "2_File_example.ext"] can be problematic to sort based on simple numerical sorting
# Instead, we can use the built-in natural sorting library (like a human would sort) to sort the filenames
# Example: "testFileName1.ext" gets converted to ["test", "FileName", "1", ".ext"]
def natural_sort_key(inputStr):
    # Handles integers and negative characters
    return [test_if_int(character) for character in re.split('(-*[0-9]+)', inputStr)]

def find_mser_params(pipeline_inputs: dict = None):
    if pipeline_inputs is not None:
        raw_directory = pipeline_inputs["raw_directory"]
        mser_save_directory = pipeline_inputs["mser_save_directory"]
        code_list = pipeline_inputs["code_list"]
        for code in code_list:
            # The directory of all the raw images for a particular code (e.g., (1))
            code_raw_directory = osp.join(raw_directory, "code " + code)
            print(f"Examining Code {code}\n{code_raw_directory}")
            
            # Load
            # Image naming convention: 1.tiff or (for a reference image) 1_ref.tiff
            raw_imgs = [img for img in os.listdir(code_raw_directory) if osp.isfile(osp.join(code_raw_directory, img)) and "ref" not in img and "amp" not in img and "phase" not in img and "MSER" not in img]
            reference_imgs = [img for img in os.listdir(code_raw_directory) if osp.isfile(osp.join(code_raw_directory, img)) and "ref" in img and "amp" not in img and "phase" not in img and "MSER" not in img]
            # Image filenames will be loaded in parallel. e.g., "1.tiff" will be loaded with its' own reference image "1_ref.tiff"
            # To do this, we need to guarantee that the filenames are sorted
            raw_imgs.sort(key=natural_sort_key)
            reference_imgs.sort(key=natural_sort_key)

            print(f"Loading Raw Images:\n{raw_imgs}")
            print(f"Loading Reference Images:\n{reference_imgs}")

            # Ensure we have as many reference images as we do raw images
            assert len(raw_imgs) == len(reference_imgs)

            for imgInd in range(len(raw_imgs)):
                raw_img = osp.join(code_raw_directory, raw_imgs[imgInd])
                reference_img = osp.join(code_raw_directory, reference_imgs[imgInd])
                saveDir = osp.join(code_raw_directory, raw_imgs[imgInd].replace(".tiff", "_MSER.json"))
                print(f"Raw Image: {raw_img}")
                print(f"Reference Image: {reference_img}\n")

                assert Path(raw_img).is_file() and Path(reference_img).is_file()

                holograms = [cv2.imread(raw_img, cv2.IMREAD_ANYDEPTH)]
                reference = cv2.imread(reference_img, cv2.IMREAD_ANYDEPTH)
                grayscales = []

                for hologram_image in holograms:
                    hologram_image = hologram_image.astype(np.float32)

                    # Normalize hologram by reference image.
                    normalized_hologram = hologram_image

                    # Now we change its data-type to a matrix of 16-bit integers, which results in a standard grayscale image.
                    grayscale_hologram = normalized_hologram.astype('uint16')
                    grayscales.append(grayscale_hologram)

                opt = MSEROptimizer(grayscales)
                opt.num_iterations = pipeline_inputs["number_iterations"]
                print(opt.num_iterations)
                opt.train(saveDir=saveDir)
                print(f"\nMSER Parameters Saved To:\n{saveDir}\n")
    else:
        srcImg = "/home/cameron/Dropbox (University of Michigan)/DL_training/data/sandbox_CG/raw/Gear_particle/code1/1.tiff"
        referenceImg = "/home/cameron/Dropbox (University of Michigan)/DL_training/data/sandbox_CG/raw/Gear_particle/code1/1_ref.tiff"
        assert Path(srcImg).is_file() and Path(referenceImg).is_file()

        holograms = [cv2.imread(srcImg, cv2.IMREAD_ANYDEPTH)]
        reference = cv2.imread(referenceImg, cv2.IMREAD_ANYDEPTH)
        grayscales = []

        for hologram_image in holograms:
            hologram_image = hologram_image.astype(np.float32)

            # Normalize hologram by reference image.
            normalized_hologram = hologram_image

            # Now we change its data-type to a matrix of 16-bit integers, which results in a standard grayscale image.
            grayscale_hologram = normalized_hologram.astype('uint16')
            grayscales.append(grayscale_hologram)

        opt = MSEROptimizer(grayscales)
        opt.train()


def train_region_classifier(pipeline_inputs: dict = None, load_data_path="data/classifier_training_samples", model_save_path="data/models/region", crossVal=False, k=5, hpoTrial=None, randomState=100, verbose=True, log=True, timestamp: str = None):
    if timestamp is None:
        timestamp = datetime.now().strftime("%m_%d_%y_%H:%M")
    
    # Train a new classifier with the data located under
    # data/classifier_training_samples/positive and
    # data/classifier_training_samples/negative
    
    # Returns list of lists
    dataList = load_data(load_data_path, verbose=verbose)
    # Based on the format of the return result of .load_data(),
    # Extract all the targets of the training samples
    targets = np.array(list(zip(*dataList))[-1])
    # All the samples
    dataset = np.asarray(load_data(load_data_path, verbose=verbose), dtype=object)

    # Do a stratified train/test split of all samples into training and test datasets
    # Returns the actual samples, not the indices of the samples.
    trainDataset, testDataset = train_test_split(dataset, test_size=0.20, stratify=targets)
    trainTargets = np.asarray(list(zip(*trainDataset))[-1])
    
    # CG: Stratified k-Fold cross-validation
    if crossVal:
        cvColumns = ["cv" + str(fold) for fold in range(1, k + 1)]
        crossValDF = pd.DataFrame(columns=cvColumns)
        splits = StratifiedKFold(n_splits=k, shuffle=True, random_state=randomState)
        
        dataset_idx = np.arange(len(trainDataset))
        crossValScores = {"Val_Loss": [], "Val_Acc": [], "Test_Loss": [], "Test_Acc": []}
        foldInd = 1
        for fold, (train_idx, val_idx) in enumerate(splits.split(dataset_idx, y=trainTargets)):
            if verbose:
                print('\n\nFold {}'.format(fold + 1))
            trainer = RegionClassifierTrainerGPU(model_save_path=model_save_path, hpoTrial=hpoTrial, verbose=verbose, log=log, timestamp=timestamp, k=foldInd)
            trainer.load_data(load_data_path, dataset, train_idx, val_idx, test_dataset=testDataset)
            crossValScores = trainer.train(crossVal=crossVal, crossValScores=crossValScores)
            # Keep track of what k-fold we are on for book-keeping
            foldInd = foldInd + 1
        
        if verbose:
            print("\nTRAINING COMPLETE.\nCross-Validation Dictionary:")
            print(crossValScores)
            for key, value in crossValScores.items():
                print("Avg. " + str(key) + ": " + str(np.array(value).mean()))
        return crossValScores 
    # If not cross-validating,
    else:
        trainer = RegionClassifierTrainerGPU(model_save_path=model_save_path, hpoTrial=hpoTrial, verbose=verbose, log=log)
        trainer.load_data(load_data_path, dataset, train_idx=None, val_idx=None)
        trainer.train(crossVal=crossVal, crossValScores=None)

def grid_search_region_classifier(pipeline_inputs: dict = None, load_hpo_path="/home/cameron/Dropbox (University of Michigan)/DL_training/hpo/region_classifier_grid_search", load_data_path="data/classifier_training_samples", model_save_path="data/models/region", crossVal=True, k=5, randomState=100, verbose=False, save_every=50):
    if pipeline_inputs is not None:
        load_hpo_path = pipeline_inputs.get("grid_search_hpo", None).get("hpo_file", None)
        load_data_path = pipeline_inputs.get("sample_parent_directory", None)
        model_save_path = pipeline_inputs.get("model_save_parent_directory", None)
        crossVal = pipeline_inputs.get("strat_kfold", None).get("activate", None)
        k = pipeline_inputs.get("strat_kfold", None).get("num_folds", None)
        randomState = pipeline_inputs.get("strat_kfold", None).get("random_state", None)
        save_every = pipeline_inputs.get("grid_search_hpo", None).get("save_every", None)
        log = pipeline_inputs.get("grid_search_hpo", None).get("log", None)
        verbose = pipeline_inputs.get("grid_search_hpo", None).get("verbose", None)
        timestamp = pipeline_inputs.get("grid_search_hpo", None).get("hpo_timestamp", datetime.now().strftime("%m_%d_%y_%H:%M"))

        # Ensuring proper pipeline inputs to minimize headaches
        assert load_hpo_path is not None and type(load_hpo_path) is str and ".csv" in load_hpo_path
        assert load_data_path is not None and type(load_data_path) is str
        assert model_save_path is not None and type(model_save_path) is str
        assert crossVal is not None and type(crossVal) is bool
        assert k is not None and type(k) is int and k >= 1
        assert randomState is not None and type(randomState) is int
        assert save_every is not None and type(save_every) is int and save_every >= 1
        assert log is not None and type(log) is bool
        assert verbose is not None and type(verbose) is bool
        assert timestamp is not None

        # If cross-validating for each hyperparameter trial,
        if crossVal:
            # Define dataframes to store crossval results
            cvLossColumns = ["Loss_cv" + str(fold) for fold in range(1, k + 1)]
            cvLossColumns.append("Loss_cv_Avg")
            cvLossColumns.insert(0, "hpoID")
            cvAccColumns = ["Acc_cv" + str(fold) for fold in range(1, k + 1)]
            cvAccColumns.append("Acc_cv_Avg")
            cvAccColumns.insert(0, "hpoID")
            
            crossValLossDF = pd.DataFrame(columns=cvLossColumns)
            crossValAccDF = pd.DataFrame(columns=cvAccColumns)
            testLossDF = pd.DataFrame(columns=cvLossColumns)
            testAccDF = pd.DataFrame(columns=cvAccColumns)
        
        # Load hyperparameter trials from "./hpo" folder
        hpoFilePath = load_hpo_path
        hpoDF = pd.read_csv(hpoFilePath)
        print(hpoDF)

        # Iterating through the dataframe as a dictionary is fast.
        
        # Keep a counter of completed trials so we can save the results in periodic intervals
        counter = 0
        # Define an error code in-case the optimization fails for a particular trial.
        # This allows the grid search to continue.
        errWriteRow = ["ERR" for fold in range(k + 1)]
        errWriteRow = dict(zip([fold for fold in range(k + 1)], errWriteRow))
        
        # For each trial,
        for row in tqdm(hpoDF.to_dict(orient="records")):
            try:
                # Try to train
                scores = train_region_classifier(load_data_path=load_data_path, 
                                                 model_save_path=model_save_path, 
                                                 crossVal=crossVal, 
                                                 k=k, 
                                                 hpoTrial=row, 
                                                 randomState=randomState, 
                                                 verbose=verbose, 
                                                 log=log,
                                                 timestamp=timestamp)
                hpoID = counter
                counter = counter + 1
            except:
                # If training failed, write the error code to the accumulated results
                scores = {"ERR": "ERR"}
                writeRow = errWriteRow
                hpoID = counter
                counter = counter + 1
            if crossVal:
                dirHead, dirTail = osp.split(load_hpo_path)
                crossValLossPath = osp.join(dirHead, "hpo_CVLoss_region_classifier.csv")
                crossValAccPath = osp.join(dirHead, "hpo_CVAcc_region_classifier.csv")
                testLossPath = osp.join(dirHead, "hpo_TestLoss_region_classifier.csv")
                testAccPath = osp.join(dirHead, "hpo_TestAcc_region_classifier.csv")
                for key, value in scores.items():
                    if value == "ERR":
                        crossValLossDF.loc[len(crossValLossDF)] = writeRow
                        crossValAccDF.loc[len(crossValAccDF)] = writeRow
                        testLossDF.loc[len(testLossDF)] = writeRow
                        testAccDF.loc[len(testAccDF)] = writeRow
                        break
                    averageVal = np.array(value).mean()
                    writeRow = value
                    writeRow.append(averageVal)
                    writeRow.insert(0, hpoID)
                    if key == "Val_Loss":
                        crossValLossDF.loc[len(crossValLossDF)] = writeRow
                    elif key == "Val_Acc":
                        crossValAccDF.loc[len(crossValAccDF)] = writeRow
                    elif key == "Test_Loss":
                        testLossDF.loc[len(testLossDF)] = writeRow
                    elif key == "Test_Acc":
                        testAccDF.loc[len(testAccDF)] = writeRow
                if counter % save_every == 0:
                    crossValLossDF.to_csv(crossValLossPath)
                    crossValAccDF.to_csv(crossValAccPath)
                    testLossDF.to_csv(testLossPath)
                    testAccDF.to_csv(testAccPath)
                if verbose:
                    print(row)
                    print(scores)
                    print("\n")
    else:
        # If cross-validating for each hyperparameter trial,
        if crossVal:
            # Define dataframes to store crossval results
            cvLossColumns = ["Loss_cv" + str(fold) for fold in range(1, k + 1)]
            cvLossColumns.append("Loss_cv_Avg")
            cvLossColumns.insert(0, "hpoID")
            cvAccColumns = ["Acc_cv" + str(fold) for fold in range(1, k + 1)]
            cvAccColumns.append("Acc_cv_Avg")
            cvAccColumns.insert(0, "hpoID")
            crossValLossDF = pd.DataFrame(columns=cvLossColumns)
            crossValAccDF = pd.DataFrame(columns=cvAccColumns)
        
        # Load hyperparameter trials from "./hpo" folder
        hpoFilePath = osp.join(load_hpo_path, "hpo_trials_region_classifier.csv")
        hpoDF = pd.read_csv(hpoFilePath)
        print(hpoDF)

        # Iterating through the dataframe as a dictionary is fast.
        
        # Keep a counter of completed trials so we can save the results in periodic intervals
        counter = 0
        # Define an error code in-case the optimization fails for a particular trial.
        # This allows the grid search to continue.
        errWriteRow = ["ERR" for fold in range(k + 1)]
        errWriteRow = dict(zip([fold for fold in range(k + 1)], errWriteRow))
        
        # For each trial,
        for row in tqdm(hpoDF.to_dict(orient="records")):
            try:
                # Try to train
                scores = train_region_classifier(load_data_path=load_data_path, model_save_path=model_save_path, crossVal=crossVal, k=k, hpoTrial=row, randomState=randomState, verbose=False, log=False)
                hpoID = counter
                counter = counter + 1
            except:
                # If training failed, write the error code to the accumulated results
                writeRow = errWriteRow
                hpoID = counter
                counter = counter + 1
            if crossVal:
                crossValLossPath = osp.join(load_hpo_path, "hpo_Loss_region_classifier.csv")
                crossValAccPath = osp.join(load_hpo_path, "hpo_Acc_region_classifier.csv")
                for key, value in scores.items():
                    if value == "ERR":
                        crossValLossDF.loc[len(crossValLossDF)] = writeRow
                        crossValAccDF.loc[len(crossValAccDF)] = writeRow
                        break
                    averageVal = np.array(value).mean()
                    writeRow = value
                    writeRow.append(averageVal)
                    writeRow.insert(0, hpoID)
                    if key == "Val_Loss":
                        crossValLossDF.loc[len(crossValLossDF)] = writeRow
                    elif key == "Val_Acc":
                        crossValAccDF.loc[len(crossValAccDF)] = writeRow
                if counter % save_every == 0:
                    crossValLossDF.to_csv(crossValLossPath)
                    crossValAccDF.to_csv(crossValAccPath)
                if verbose:
                    print(row)
                    print(scores)
                    print("\n")

def classify_regions(pipeline_inputs: dict = None, load_path=None, img_folder=None):
    # If we are using the control panel .ipynb pipeline,
    if pipeline_inputs is not None:
        # Get the parent directory of raw images for all codes
        raw_directory = pipeline_inputs["raw_directory"]
        # Get the list of codes to process from the control panel pipeline inputs 
        code_list = pipeline_inputs["code_list"]
        # For each code to process
        for codeNum in tqdm(code_list):
            # Get the raw image directory for that code
            raw_code_dir = osp.join(raw_directory, "code " + codeNum)
            # Define positive/negative sample save paths
            posSaveDir = osp.join("data/classifier_training_samples/", "code " + codeNum, "positive")
            negSaveDir = osp.join("data/classifier_training_samples/", "code " + codeNum, "negative")
            if not osp.exists(posSaveDir):
                os.makedirs(posSaveDir)
                os.makedirs(negSaveDir)
            print(f"Processing and saving data to:\n{posSaveDir}")
            print(f"Processing and saving data to:\n{negSaveDir}")
            
            filenames = sorted(os.listdir(raw_code_dir))
            
            # Load MSER parameters for an initial division of positive/negative samples
            MSER_params = pipeline_inputs["MSER_params_per_code"][codeNum]
            # For each set of MSER parameters, which corresponds to each reference image of each code,
            for MSERFile in tqdm(MSER_params):
                # These counters for positive/negative samples are counted per-reference image, not per-code
                sum_pos = 0
                sum_neg = 0
                raw_image_id = MSERFile.rstrip("_MSER.json")
                with open(osp.join(raw_code_dir, MSERFile), "r") as MSERObj:
                    MSERDict = json.load(MSERObj)["optimizer.max"]["params"]
                # Define a region detector for positive/negative division of samples
                # This region detector is not optimized to be the most accurate
                # Rather, we use the region detector as a means of initializing positive/negative samples.
                # Later, we will actually hyperparameter optimize the region detectors for accurate positive/negative sample separation
                region_detector = RegionDetector(MSER_parameters=MSERDict)
                # For each reference image in the raw image directory for a particular code,
                refname = raw_image_id + "_ref.tiff"
                
                holograms=[]
                # Ignore the file if it is not an image file
                if ".tiff" not in refname:
                    continue
                # If the file is a reference image,
                elif "ref" in refname:
                    # Load reference
                    reference = cv2.imread(osp.join(raw_code_dir, refname), cv2.IMREAD_ANYDEPTH)
                    # For each image in the raw image directory of a particular code
                    for imname in filenames:
                        # The raw image file is just the name of the reference file without the reference designation
                        code = refname.replace("_ref.tiff", "")
                        # If we found the code image that corresponds with its' own refernece image,
                        if code == imname.replace(".tiff", "").split("_")[0] and ".tiff" in imname and "ref" not in imname:
                            # Read the raw code image
                            hologram = cv2.imread("{}/{}".format(raw_code_dir, imname), cv2.IMREAD_ANYDEPTH)
                            hologram = hologram.astype(np.float32)
                            # Append the raw image data with the reference image data.
                            holograms.append((hologram, imname.replace(".tiff", ""), reference))

                    for hologram in holograms:
                        holo, name, reference = hologram
                        save_img_name = "data/test/{}_{}_regions.png".format(codeNum, name)
                        positive_regions, negative_regions = region_detector.detect_regions(holo,
                                                                                            reference,
                                                                                            save_img_name=save_img_name)
                        for i in range(len(positive_regions)):
                            file_path = "{}/{}_{}_positive_{}.png".format(posSaveDir, codeNum, raw_image_id, i+sum_pos)
                            region = positive_regions[i]
                            shape = region.shape
                            new_shape = (*shape[1:], shape[0])
                            region = region.reshape(new_shape)
                            cv2.imwrite(file_path, region)
                        sum_pos = sum_pos + len(positive_regions)

                        for i in range(len(negative_regions)):
                            file_path = "{}/{}_{}_negative_{}.png".format(negSaveDir, codeNum, raw_image_id, i+sum_neg)
                            region = negative_regions[i]
                            shape = region.shape
                            new_shape = (*shape[1:], shape[0])
                            region = region.reshape(new_shape)
                            cv2.imwrite(file_path, region)
                        sum_neg = sum_neg + len(negative_regions)
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

        for refname in os.listdir(img_folder):
            if ".tiff" not in refname:
                continue
            if "ref" in refname:
                print("referencing: ", refname)
                reference = cv2.imread("{}/{}".format(img_folder, refname), cv2.IMREAD_ANYDEPTH)

                for imname in os.listdir(img_folder):
                    code = refname.replace("_ref.tiff", "")
                    if code == imname.replace(".tiff", "").split("_")[0] and ".tiff" in imname and "ref" not in imname:
                        hologram = cv2.imread("{}/{}".format(img_folder, imname), cv2.IMREAD_ANYDEPTH)
                        hologram = hologram.astype(np.float32)
                        holograms.append((hologram, imname.replace(".tiff", ""), reference))

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


def train_code_classifier():
    codes = ["1", "2", "3"]
    trainer = CodeClassifierTrainerGPU(codes, model_save_path="data/models/code")
    trainer.load_data("data/classifier_training_samples")
    trainer.train()


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

    for refname in os.listdir(img_folder):
        if ".tiff" not in refname:
            continue
        if "ref" in refname:
            print("referencing: ", refname)
            reference = cv2.imread("{}/{}".format(img_folder, refname), cv2.IMREAD_ANYDEPTH)

            for imname in os.listdir(img_folder):
                code = refname.replace("_ref.tiff", "")
                if code == imname.replace(".tiff", "").split("_")[0] and ".tiff" in imname and "ref" not in imname:
                    hologram = cv2.imread("{}/{}".format(img_folder, imname), cv2.IMREAD_ANYDEPTH)
                    hologram = hologram.astype(np.float32)
                    holograms.append((hologram, imname.replace(".tiff", ""), reference))

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
        csvwriter.writerows(intensities)


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
    action = args.action
    dir = args.dir
    reg_path = args.reg_path
    code_path = args.code_path
    pipeline_inputs = args.pipeline_inputs

    # Controls the functionality of the pipeline Jupter notebooks
    # -----------------------------------------------------------------
    if args.pipeline_inputs is not None:
        # Loads the inputs to the pipeline if specified
        with open(pipeline_inputs, "r") as inputFile:
            pipeline_inputs = json.load(inputFile)
    else:
        # If not specified, the code runs in an off-rails fashion without the jupyter notebooks
        # It can run, but a lot of manual intervention is required
        pipeline_inputs = None
    # ------------------------------------------------------------------
    
    if action == 'find_mser_params':
        if pipeline_inputs is None:
            find_mser_params()
        else:
            find_mser_params(pipeline_inputs=pipeline_inputs)
    elif action == 'train_region_classifier':
        if pipeline_inputs is None:
            train_region_classifier()
        else:
            train_region_classifier(pipeline_inputs=pipeline_inputs)
    elif action == 'hpo_region_classifier':
        if pipeline_inputs is None:
            grid_search_region_classifier()
        else:
            grid_search_region_classifier(pipeline_inputs=pipeline_inputs)
    elif action == 'classify_regions':
        if pipeline_inputs is None:
            classify_regions(load_path=reg_path, img_folder=dir)
        else:
            classify_regions(pipeline_inputs=pipeline_inputs)
    elif action == 'train_code_classifier':
        train_code_classifier()
    elif action == 'test_system':
        assert(reg_path is not None and code_path is not None)
        test_system(img_folder=dir, region_detector_path=reg_path,
                    code_classifier_path=code_path)
    elif action == 'get_intensity':
        get_intensity(img_folder=dir,
                      region_detector_path=reg_path)
    else:
      print("invalid action")