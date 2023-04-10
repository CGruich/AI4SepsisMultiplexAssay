import numpy as np
import torch.nn as nn
import torch
from utils import helper_functions


class CodeClassifier(nn.Module):
    def __init__(self, n_codes, region_shape=(1, 128, 128), model_load_path=None):
        super().__init__()
        ch1 = 64
        ch2 = 32
        ch3 = 16
        h1 = 64
        self.conv_layers = [
            nn.BatchNorm2d(1),
            nn.Conv2d(in_channels=1, out_channels=ch1, kernel_size=(6,6), stride=(3,3)),
            nn.PReLU(),

            nn.Dropout(p=0.1),
            nn.MaxPool2d(kernel_size=(2,2)),

            nn.BatchNorm2d(ch1),
            nn.Conv2d(in_channels=ch1, out_channels=ch2, kernel_size=(4, 4), stride=(2, 2)),
            nn.PReLU(),

            nn.Dropout(p=0.1),
            nn.MaxPool2d(kernel_size=(2,2)),

            nn.BatchNorm2d(ch2),
            nn.Conv2d(in_channels=ch2, out_channels=ch3, kernel_size=(3, 3), stride=(1, 1)),
            nn.PReLU(),

            nn.Dropout(p=0.1),
        ]

        # Convolutional layers reduce the size of their input by some amount. Because of this, we need to find out how
        # many features will remain when we finish passing the input through the convolutional layers. Once we know,
        # we can concatenate the output of the final convolutional layer into one long vector so we can pass it through
        # the feedforward layers for classification.
        detected_conv_features = helper_functions.detect_conv_features(region_shape, self.conv_layers)
        print("FINAL VECTOR LENGTH:",detected_conv_features)
        self.ff_layers = [
            nn.Flatten(),
            nn.Dropout(p=0.1),
            nn.BatchNorm1d(detected_conv_features),
            nn.Linear(detected_conv_features, h1),
            nn.PReLU(),

            nn.BatchNorm1d(h1),
            nn.Linear(h1, n_codes),
            nn.Softmax(dim=-1)
        ]

        layers = self.conv_layers + self.ff_layers
        self.model = nn.Sequential(*layers)

        if model_load_path is not None:
            self.load_weights(model_load_path)

    def forward(self, x):
        if type(x) != torch.Tensor:
            if type(x) != np.ndarray:
                x = np.asarray(x).astype(np.float32)  # apparently faster to do this conversion first

            x = torch.as_tensor(x, dtype=torch.float32)
        return self.model(x)

    def classify_regions(self, regions):
        """
        Function to classify an array of 64x64 grayscale images.
        :param regions: Regions to classify. Shape should be (batch_size, 1, 64, 64)
        :return: Array of positive classified regions & array of negative classified regions
        """
        if len(regions) == 0:
            return [], []

        predicted_region_codes = []

        # Predict labels.
        predicted_labels = self.forward(regions)

        # Assign each label to the corresponding region.
        for i in range(len(regions)):
            region = regions[i]
            label = predicted_labels[i]
            predicted_region_codes.append(label.argmax())

        return predicted_region_codes

    def load_weights(self, filepath):
        self.load_state_dict(torch.load(filepath, map_location=torch.device('cpu')))
        self.eval()
