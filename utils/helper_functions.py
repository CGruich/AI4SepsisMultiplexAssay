import numpy as np
import torch.nn as nn
import torch


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
    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")

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
        idxs = np.delete(idxs, np.concatenate(([last], np.where(overlap > maximum_acceptable_overlap)[0])))

    if return_picks:
        return boxes[pick].astype("int"), pick

    # return only the bounding boxes that were picked using the
    # integer data type
    return boxes[pick].astype("int")


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
    y1 = max(int(round(min(y1, y2) - height_expansion / 2.)), 0)

    # Force y2 to be the largest of the y values. Shift y2 by half the necessary expansion. Bound y2 to a maximum of h.
    y2 = min(int(round(max(y1, y2) + height_expansion / 2.)), h)

    # These two lines are a repeat of the above y1,y2 calculations but with x and w instead of y and h.
    x1 = max(int(round(min(x1, x2) - width_expansion / 2.)), 0)
    x2 = min(int(round(max(x1, x2) + width_expansion / 2.)), w)

    # Calculate the center point of the newly reshaped box, truncated.
    cx = x1 + abs(x1 - x2) // 2
    cy = y1 + abs(y1 - y2) // 2

    return x1, y1, x2, y2, cx, cy