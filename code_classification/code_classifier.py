import numpy as np
import torch.nn as nn
import torch
from utils import helper_functions


class CodeClassifier(nn.Module):
    def __init__(
        self,
        n_codes,
        region_shape=(1, 128, 128),
        fc_size: int = 256,
        fc_num: int = 2,
        dropout_rate: float = 0.1,
        model_load_path: str = None,
    ):
        super().__init__()

        # CG: CPU or GPU, prioritizes GPU if available.
        self.device = torch.device(
            'cuda:0' if torch.cuda.is_available() else 'cpu')

        ch1 = 64
        ch2 = 32
        ch3 = 16
        h1 = fc_size
        self.conv_layers = [
            nn.BatchNorm2d(1),
            nn.Conv2d(in_channels=1, out_channels=ch1,
                      kernel_size=(6, 6), stride=(3, 3)),
            nn.PReLU(),
            nn.Dropout(p=dropout_rate),
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.BatchNorm2d(ch1),
            nn.Conv2d(in_channels=ch1, out_channels=ch2,
                      kernel_size=(4, 4), stride=(2, 2)),
            nn.PReLU(),
            nn.Dropout(p=dropout_rate),
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.BatchNorm2d(ch2),
            nn.Conv2d(in_channels=ch2, out_channels=ch3,
                      kernel_size=(3, 3), stride=(1, 1)),
            nn.PReLU(),
            nn.Dropout(p=dropout_rate),
        ]

        # Convolutional layers reduce the size of their input by some amount. Because of this, we need to find out how
        # many features will remain when we finish passing the input through the convolutional layers. Once we know,
        # we can concatenate the output of the final convolutional layer into one long vector so we can pass it through
        # the feedforward layers for classification.
        detected_conv_features = helper_functions.detect_conv_features(
            region_shape, self.conv_layers
        )
        # print("FINAL VECTOR LENGTH:",detected_conv_features)
        self.ff_layers = [
            nn.Flatten(),
            nn.BatchNorm1d(detected_conv_features),
            nn.Linear(detected_conv_features, h1),
            nn.PReLU(),
            nn.Dropout(p=dropout_rate),
        ]

        self.fc_layers = []
        # Ensure at least one fully-connected layer
        assert fc_num >= 1
        # If more than 1 fully connected layer to add,
        if fc_num > 1:
            # For each layer, not counting the very last layer
            for _ in range(fc_num - 1):
                # Append batch normalization
                self.fc_layers.append(nn.BatchNorm1d(h1))
                # Append a linear layer
                self.fc_layers.append(nn.Linear(h1, h1))
                # Append PRELU() activation function
                self.fc_layers.append(nn.PReLU())
                # Append dropout layer to prevent overfitting
                self.fc_layers.append(nn.Dropout(p=dropout_rate))
            # Add the final layer that outputs the code label predictions
            self.fc_layers.append(nn.BatchNorm1d(h1))
            self.fc_layers.append(nn.Linear(h1, n_codes))
            # Dropout layer to prevent overfitting
            self.fc_layers.append(nn.Dropout(p=dropout_rate))
            # Because we use cross-entropy loss to train the model,
            # log(softmax(x)) is applied internally within the loss function.
            # So, we cannot apply softmax redundantly here.
            # Because this is a multi-class classification problem where we want to select the class with the highest probability,
            # it behooves us to use an activation function that can segregate values into a high-value regime and a low-value regime,
            # such as Softplus()
            # If we wanted the true class probabilities, we can apply softmax() after-the-fact during inference.
            self.fc_layers.append(nn.Softplus())

            # Not compatible with cross-entropy loss function, as cross-entropy loss applies softmax internally
            # For now, this is commented out. In production outside of model training, this can be uncommented and used.
            # nn.Softmax(dim=-1)
        # Else if only one fully connected layer to add
        else:
            self.fc_layers.append(nn.BatchNorm1d(h1))
            self.fc_layers.append(nn.Linear(h1, n_codes))
            # Dropout layer to prevent overfitting
            self.fc_layers.append(nn.Dropout(p=dropout_rate))
            # Because we use cross-entropy loss to train the model,
            # log(softmax(x)) is applied internally within the loss function.
            # So, we cannot apply softmax redundantly here.
            # Because this is a multi-class classification problem where we want to select the class with the highest probability,
            # it behooves us to use an activation function that can segregate values into a high-value regime and a low-value regime,
            # such as Softplus()
            # If we wanted the true class probabilities, we can apply softmax() after-the-fact during inference.
            self.fc_layers.append(nn.Softplus())

        # Combine all the layers
        layers = self.conv_layers + self.ff_layers + self.fc_layers
        self.model = nn.Sequential(*layers)
        self.model.to(self.device)

        if model_load_path is not None:
            self.load_weights(model_load_path)

    def forward(self, x):
        if type(x) != torch.Tensor:
            if type(x) != np.ndarray:
                # apparently faster to do this conversion first
                x = np.asarray(x).astype(np.float32)
            x = torch.as_tensor(x, dtype=torch.float32)
        x = x.to(self.device)
        return self.model(x).to(self.device)

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
            label = predicted_labels[i]
            predicted_region_codes.append(label.argmax())

        return predicted_region_codes

    def load_weights(self, filepath):
        self.load_state_dict(torch.load(filepath, map_location=self.device))
        self.eval()
