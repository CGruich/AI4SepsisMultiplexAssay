import numpy as np
import torch.nn as nn
import torch
from scipy.signal import convolve2d
from pathlib import Path
import json

# from object_detection import RegionDetector
import cv2
import numpy as np
import pandas as pd
import os
import csv
import itertools
import re as regex
from PIL import Image
from tqdm import tqdm

def normalize_by_reference(
    hologram,
    reference,
    conv_window_size=10,
    bit_depth=16,
    scale_to_bit_depth=True,
    ref_already_convolved=False,
):
    conv_window_size = conv_window_size
    convolution_kernel = np.ones((conv_window_size, conv_window_size)) / (
        conv_window_size * conv_window_size
    )
    bit_depth = bit_depth

    hologram_image = hologram.astype(np.float32)
    reference_image = reference.astype(np.float32)

    # For each pixel in the reference, we will construct a square with side lengths `conv_window_size` centered
    # at the current pixel. Then, we will compute the average value of every pixel in side this square, and set
    # the pixel at the current coordinates inside a new image to that value. This gives us a significantly better
    # image to use for normalization.
    if not ref_already_convolved:
        averaged_reference_image = convolve2d(reference_image, convolution_kernel, mode='same')
    else:
        averaged_reference_image = reference_image

    # Normalize hologram by reference image.
    normalized_hologram = hologram_image / averaged_reference_image

    if scale_to_bit_depth:
        # Transform the normalized image into the appropriate bit-depth.
        grayscale_hologram = normalized_hologram * 2 ** 16
        grayscale_hologram = grayscale_hologram.clip(0, 2 ** 16 - 1).astype(
            'uint{}'.format(bit_depth)
        )
    else:
        grayscale_hologram = normalized_hologram
    return grayscale_hologram


def detect_conv_features(input_shape, conv_layers):
    """
    Quick and dirty method to detect how many elements will be at the output of some number of convolutional layers.
    :param input_shape: Shape of the input to be passed to the convolutional layers. Do not include batch size.
    :param conv_layers: List of convolutional layers to test.
    :return: Number of features at the final convolutional layer.
    """

    conv = nn.Sequential(*conv_layers)
    inp = torch.ones((1, *input_shape))
    out = conv(inp)
    return int(np.prod(out.shape))


def non_max_suppression_fast(boxes, maximum_acceptable_overlap, return_picks=False):
    """
    Fast non-maximum suppression algorithm by Malisiewicz et al.
    :param boxes: Array of boundary boxes to suppress.
    :param maximum_acceptable_overlap: Maximum acceptable overlap threshold. Must be on the interval [0, 1].
    :return: Array of suppressed boundary boxes, none of whom will overlap by more than `maximum_acceptable_overlap`
    """

    # if there are no boxes, return an empty list
    if len(boxes) == 0:
        if return_picks:
            return [], []
        return []

    # if the bounding boxes are integers, convert them to floats --
    # this is important since we'll be doing a bunch of divisions
    if boxes.dtype.kind == 'i':
        boxes = boxes.astype('float')

    # initialize the list of picked indexes
    pick = []

    # grab the coordinates of the bounding boxes
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    # compute the area of the bounding boxes and sort the bounding
    # boxes by the bottom-right y-coordinate of the bounding box
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(y2)

    # keep looping while some indexes still remain in the indexes
    # list
    while len(idxs) > 0:
        # grab the last index in the indexes list and add the
        # index value to the list of picked indexes
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        # find the largest (x, y) coordinates for the start of
        # the bounding box and the smallest (x, y) coordinates
        # for the end of the bounding box
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])

        # compute the width and height of the bounding box
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        # compute the ratio of overlap
        overlap = (w * h) / area[idxs[:last]]

        # delete all indexes from the index list that exceed the maximum overlap threshold
        idxs = np.delete(
            idxs, np.concatenate(([last], np.where(overlap > maximum_acceptable_overlap)[0])),
        )

    if return_picks:
        return boxes[pick].astype('int'), pick

    # return only the bounding boxes that were picked using the
    # integer data type
    return boxes[pick].astype('int')


def expand_bbox(bbox, desired_dims, image_boundary):
    """
    Function to expand a bounding box around its center to some desired new shape. This function will force the final
    bounding boxes to be within the dimensions (0, 0, image_boundary[0], image_boundary[1]) by truncating any boxes that
    extend beyond the limiting boundary. This is to prevent bounding boxes from expanding beyond the edges of the image.

    :param bbox: Bounding box to expand.
    :param desired_dims: Desired width & height.
    :param image_boundary: Edges of the image in which boxes must be kept.
    :return: Expanded bounding box.
    """

    # Unpack the box.
    x1, y1, x2, y2 = bbox
    # h & w are the maximum x,y coordinates that the box is allowed to attain.
    h, w = image_boundary
    # dw & dh are the desired width and height of the box.
    dw, dh = desired_dims

    # Calculated the amount that the dimensions of the box will need to be changed
    height_expansion = dh - abs(y1 - y2)
    width_expansion = dw - abs(x1 - x2)

    # Force y1 to be the smallest of the y values. Shift y1 by half of the necessary expansion.
    # Bound y1 to a minimum of 0.
    y1 = max(int(round(min(y1, y2) - height_expansion / 2.0)), 0)

    # Force y2 to be the largest of the y values. Shift y2 by half the necessary expansion. Bound y2 to a maximum of h.
    y2 = min(int(round(max(y1, y2) + height_expansion / 2.0)), h)

    # These two lines are a repeat of the above y1,y2 calculations but with x and w instead of y and h.
    x1 = max(int(round(min(x1, x2) - width_expansion / 2.0)), 0)
    x2 = min(int(round(max(x1, x2) + width_expansion / 2.0)), w)

    # Calculate the center point of the newly reshaped box, truncated.
    cx = x1 + abs(x1 - x2) // 2
    cy = y1 + abs(y1 - y2) // 2

    return x1, y1, x2, y2, cx, cy


def load_data(folder_path: str, verbose: bool = True, stratify_by_stain: bool = False):
    positive_sample_folder = os.path.join(folder_path, 'positive')
    negative_sample_folder = os.path.join(folder_path, 'negative')

    # Filter incorrect image sizes, if any happen to make it in the dataset
    filter_invalid_image_sizes(image_folder_path=positive_sample_folder, correct_image_size=(128, 128), img_ext='.png')
    filter_invalid_image_sizes(image_folder_path=negative_sample_folder, correct_image_size=(128, 128), img_ext='.png')

    data = []

    # For each image in the positive samples folder.
    for file_name in os.listdir(positive_sample_folder):
        if not file_name.endswith('.png'):
            continue

        # Load region.
        region = cv2.imread(
            os.path.join(positive_sample_folder, file_name), cv2.IMREAD_ANYDEPTH
        )

        if stratify_by_stain:
            # Follows filename convention:
            # code 3_100_2_2170_1686_128x128.png
            # code {code_num}_{stain_level}_{hologram_num}_{pixelX_position}_{pixelY_position}_{img_length}_{img_width}
            stain_level = file_name.split('_')[1]
            label = '1' + '_' + stain_level
        else:
            label = 1

        # Append region and positive label to dataset.
        data.append([region.reshape(1, *region.shape), label])

    n_positive = len(data)

    if verbose:
        print('Loaded {} positive training samples.'.format(n_positive))

    # For each image in the negative samples folder.
    for file_name in os.listdir(negative_sample_folder):
        if not file_name.endswith('.png'):
            continue

        # Load region.
        region = cv2.imread(
            os.path.join(negative_sample_folder, file_name), cv2.IMREAD_ANYDEPTH
        )
        
        if stratify_by_stain:
            # Follows filename convention:
            # code 3_100_2_2170_1686_128x128.png
            # code {code_num}_{stain_level}_{hologram_num}_{pixelX_position}_{pixelY_position}_{img_length}_{img_width}
            stain_level = file_name.split('_')[1]
            label = '0' + '_' + stain_level
        else:
            label = 0
        # Append region and negative label to dataset.
        data.append([region.reshape(1, *region.shape), label])

    if verbose:
        print('Loaded {} negative training samples.'.format(len(data) - n_positive))

    # Return dataset
    return data

def stain_labels_to_training_labels(data: np.ndarray, substr: str = '_'):
    mask = np.char.find(data.astype(str), substr) != -1
    data[mask] = [int(s.split('_')[0]) for s in data[mask]]

    return data

def sort_alphanumeric(string_list):
    assert hasattr(string_list, 'sort'), 'ERROR! TYPE {} DOES NOT HAVE A SORT FUNCTION'.format(
        type(string_list)
    )
    """
    Function to sort a list of strings in alphanumeric order.
    Example: the list ['b1','a1','b2','a3','b3','a2'] will be sorted as ['a1', 'a2', 'a3', 'b1', 'b2', 'b3']

    :param string_list: list of strings to sort.
    """

    def sorting_key(x):
        return [int(c) if type(c) == int else c for c in regex.split('(-*[0-9]+)', x)]

    string_list.sort(key=sorting_key)


def load_code(code_folder_path, verbose=True, stratify_by_stain: bool = False):
    code_sample_folder = os.path.join(code_folder_path, 'positive')

    # Filter incorrect image sizes, if any happen to make it in the dataset
    filter_invalid_image_sizes(image_folder_path=code_sample_folder, correct_image_size=(128, 128), img_ext='.png')

    data = []
    try:
        code_designation = int(code_folder_path[-3:].strip('()'))
    except:
        print(
            '\n\nFAILURE OBTAINING DESIGNATED CODE LABEL FROM THE PARENT FOLDER PATH.\nThe parent folder name may not have been correctly labelled.'
        )

    # For each image in the code's positive samples folder,
    for file_name in os.listdir(code_sample_folder):
        if not file_name.endswith('.png'):
            continue

        # Load region.
        # Normalize by max possible pixel value
        region = cv2.imread(os.path.join(code_sample_folder, file_name), cv2.IMREAD_ANYDEPTH) / 65535
        try:
            label = int(file_name.split('_')[0].replace('code ', ''))

            assert label == code_designation
            
            if stratify_by_stain:
                # Follows filename convention:
                # code 3_100_2_2170_1686_128x128.png
                # code {code_num}_{stain_level}_{hologram_num}_{pixelX_position}_{pixelY_position}_{img_length}_{img_width}
                stain_level = file_name.split('_')[1]
                label = str(label) + '_' + stain_level
        except:
            print(
                '\n\nFAILURE OBTAINING TARGET LABEL FROM SAMPLE FILENAMES.\nThe sample filenames may not have been correctly labelled.'
            )
        # Append region and negative label to dataset.
        data.append([region.reshape(1, *region.shape), label])

    n_positive = len(data)

    if verbose:
        print('Loaded {} positive training samples'.format(n_positive))

    # Return dataset for one code
    return data


def find_intensity(normalized_hologram, particle_locations):
    window_x = 20
    window_y = 20
    h, w = normalized_hologram.shape

    intensities = []

    for coord in particle_locations['particle_locations']:
        x, y = coord
        left = max(x - window_x, 0)
        right = min(x + window_x, w)
        top = max(y - window_y, 0)
        bottom = min(y + window_y, h)
        region = normalized_hologram[top:bottom, left:right]
        region_intensity = region.mean()
        intensities.append(region_intensity)

    return intensities

def get_particle_intensity(particle_image, zoom_dims=(32,32)):
    h, w = particle_image.shape
    x1 = w // 2 - zoom_dims[0]
    x2 = w // 2 + zoom_dims[0]
    y1 = h // 2 - zoom_dims[1]
    y2 = h // 2 + zoom_dims[1]
    region = particle_image[y1:y2, x1:x2]
    intensity_estimate = region.mean()
    return intensity_estimate

def get_particle_intensities(particle_image_list, zoom_dims=(32, 32)):
    intensity_list = []
    for particle_image in tqdm(particle_image_list):
        intensity_estimate = get_particle_intensity(particle_image=particle_image, zoom_dims=zoom_dims)
        intensity_list.append(intensity_estimate)
    return intensity_list

# Takes some data within a dataframe, based on some column name (col_name),
# and puts that data within labelled bins.
# This is for generating discretized labels of supervised training data
# Specifically, this function forces all the bins to contain a minimum number of datapoints (min_bin_count)
def guarantee_bin_width_for_bin_count(df, col_name, min_bin_count=2):
    # Get the number of bins expected via simple relationship
    num_bins = len(df) // min_bin_count

    # While the minimum number of datapoints in any given bin is greater than our expected minimum
    while True:
        # Keep trying to bin fairly
        bins = pd.qcut(df[col_name], q=num_bins, duplicates='drop')
        if bins.value_counts().min() >= min_bin_count:
            break
        # Keep reducing the number of bins for a fixed dataset to try and meet the min_bin_count condition
        num_bins -= 1

    # Assign a label to each bin for each datapoint in the original dataframe
    df[col_name + '_bin'] = bins.cat.codes.apply(lambda x: x)
    
    # Return dataframe with col_name and a newly added label column (col_name_bin)
    return df, bins

def load_and_normalize(raw_directory, code_list, color=False, one_ref_per_img=True):
    for code in code_list:
        # If we are processing colored images of the barcoded particles,
        if color:
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

            if 'ref' in file_name:
                reference_img_names.append(file_name)
            elif 'particle_locations' in file_name:
                particle_location_names.append(file_name)
            else:
                raw_img_names.append(file_name)

        # Image filenames will be loaded in parallel. e.g., "1.tiff" will be loaded with its own reference image "1_ref.tiff"
        # Here we ensure the file names are sorted alphanumerically so each file name is paired with the appropriate reference
        # and particle position list.
        sort_alphanumeric(raw_img_names)
        sort_alphanumeric(particle_location_names)
        sort_alphanumeric(reference_img_names)

        print(f'Loading Raw Images:\n{raw_img_names}')
        print(f'Loading Reference Images:\n{reference_img_names}')
        print(f'Loading Particle Locations:\n{particle_location_names}')

        # Ensure we have as many reference images as we do raw images
        assert len(raw_img_names) == len(reference_img_names)

        holograms = []
        references = []
        grayscales = []
        particle_locations = []

        if not one_ref_per_img:
            reference_img_path = os.path.join(code_raw_directory, 'ref.tiff')
            references.append(cv2.imread(reference_img_path, cv2.IMREAD_ANYDEPTH))

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
            if one_ref_per_img:
                references.append(cv2.imread(reference_img_path, cv2.IMREAD_ANYDEPTH))

            with open(particle_location_path, 'r') as particle_file:
                particle_locations_json = dict(json.load(particle_file))

            particle_locations_list = list(particle_locations_json['particle_locations'])
            particle_locations.append(particle_locations_list)

        for i in range(len(holograms)):
            hologram_image = holograms[i]
            if one_ref_per_img:
                reference_image = references[i]
            else:
                reference_image = references[0]

            grayscale_hologram = normalize_by_reference(hologram_image, reference_image)
            grayscales.append(grayscale_hologram)

        return grayscales, particle_locations

def visualize_image(image_array: np.ndarray, scale: tuple = None):

    cv2.namedWindow('Image Preview Debug (Press Enter to exit...)')
    cv2.imshow('Image Preview Debug (Press Enter to exit...)', cv2.resize(image_array, scale))
    cv2.moveWindow('Image Preview Debug (Press Enter to exit...)', 0, 0)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imshow('Image', image_array)

def filter_invalid_image_sizes(image_folder_path: str, correct_image_size: tuple = (128, 128), img_ext: str = '.png'):
    
    for filename in os.listdir(image_folder_path):
        if filename.endswith(img_ext):
            img_path = os.path.join(image_folder_path, filename)
            img = Image.open(img_path)
            img_length, img_width = img.size
            if img_length != correct_image_size[0] or img_width != correct_image_size[1]:
                os.remove(img_path)