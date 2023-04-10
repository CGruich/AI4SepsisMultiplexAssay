import numpy as np
import torch.nn as nn
import torch
from utils import helper_functions


class RegionClassifier(nn.Module):
    def __init__(self, region_shape=(1, 128, 128), h1=128, dropout=0.5, verbose=True):
        super().__init__()
        
        # Printing verbosity
        self.verbose = verbose

        ch1 = 32
        ch2 = 16
        # Controls size of fully connected layer
        self.h1 = h1
        # Controls dropout rate before fully connected layer
        self.dropout = dropout
        
        self.conv_layers = [
            nn.BatchNorm2d(1),
            nn.Conv2d(in_channels=1, out_channels=ch1, kernel_size=(6, 6), stride=(3, 3)),
            nn.PReLU(),

            nn.MaxPool2d(kernel_size=(2, 2)),

            nn.BatchNorm2d(ch1),
            nn.Conv2d(in_channels=ch1, out_channels=ch2, kernel_size=(4, 4), stride=(2, 2)),
            nn.PReLU(),

            nn.MaxPool2d(kernel_size=(2, 2))
        ]

        # Convolutional layers reduce the size of their input by some amount. Because of this, we need to find out how
        # many features will remain when we finish passing the input through the convolutional layers. Once we know,
        # we can concatenate the output of the final convolutional layer into one long vector so we can pass it through
        # the feedforward layers for classification.
        detected_conv_features = helper_functions.detect_conv_features(region_shape, self.conv_layers)
        if self.verbose:
            print("FINAL VECTOR LENGTH:", detected_conv_features)
        self.ff_layers = [
            nn.Flatten(),
            nn.Dropout(p=self.dropout),
            nn.Linear(detected_conv_features, self.h1),
            nn.PReLU(),
            nn.Linear(self.h1, 2),
            nn.Softmax(dim=-1)
        ]

        layers = self.conv_layers + self.ff_layers
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        if type(x) != torch.Tensor:
            if type(x) != np.ndarray:
                x = np.asarray(x).astype(np.float32)  # apparently faster to do this conversion first

            x = torch.as_tensor(x, dtype=torch.float32)
        return self.model(x)

    def classify_regions(self, regions, return_picks=False):
        """
        Function to classify an array of 64x64 grayscale images.
        :param regions: Regions to classify. Shape should be (batch_size, 1, 64, 64)
        :return: Array of positive classified regions & array of negative classified regions
        """
        if len(regions) == 0:
            return [], []

        positive_regions = []
        negative_regions = []

        positive_idx = []

        # Predict labels.
        predicted_labels = self.forward(regions)

        # Assign each label to the corresponding region.
        for i in range(len(regions)):
            region = regions[i]
            label = predicted_labels[i]

            # The model will always output a 2-element array for each region. We're checking the index of the largest
            # element in this array, and using that to determine the classification result. 1 = positive, 0 = negative.
            if label[1] > label[0]:
                positive_regions.append(region)
                positive_idx.append(i)
            else:
                negative_regions.append(region)
        
        if return_picks:
            return positive_regions, negative_regions, positive_idx

        return positive_regions, negative_regions

    def load_weights(self, filepath):
        self.load_state_dict(torch.load(filepath))
        self.eval()
