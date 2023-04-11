from code_classification import CodeClassifier
import numpy as np
import cv2
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

class CodeClassifierTrainerGPU(object):
    def __init__(self, codes=["1-1", "1-2", "1-3", "1-4", "1-5", "1-6", "1-7", "1-8", "1-9", "1-10"], 
                 model_save_path="data/models/code_classifier",
                 hpoTrial=None,
                 save_every_n=10,
                 batch_size=256,
                 verbose=True,
                 log=True
                 ):
        
        # CG: CPU or GPU, prioritizes GPU if available.
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        self.verbose = verbose
        if self.verbose:
            print("CUDA Availability: " + str(torch.cuda.is_available()))
        
        n_codes = len(codes)
        self.model = CodeClassifier(n_codes)
        if self.verbose:
            print("Model Loaded to GPU: " + str(next(self.model.parameters()).is_cuda))
        
        self.train_data = None
        self.val_data = None

        self.num_codes = n_codes
        # codes = ["1-1", "1-3", "1-12", "1-18", "2-10", "2-17"] # + ["invalid"]

        self.code_map = {code: idx for idx, code in enumerate(codes)}
        print(f"Code Map Between Sample Filenames and Internal Code Integer Designation:\n{self.code_map}")

        self.model_save_path = model_save_path
        self.save_every_n = save_every_n

        # Hyper-parameters.
        self.batch_size = batch_size
        self.n_epochs = 20000
        self.learning_rate = 1e-4
        self.val_split = 0.2
        self.max_transform_sequence = 10
        self.best_val_acc = 0
        self.losses = {"epoch": [], "ta": [], "va": [], "tl": [], "vl": []}
        self.patience = 5

        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.loss_fn = nn.BCELoss()

    def train(self):
        """
        Function to train a classifier on hologram regions.
        :return: None.
        """

        # Set the PyTorch model to training mode.
        self.model.train()

        # Doing this lets us access these four variables much faster than if we accessed them through `self` every time.
        loss_fn = self.loss_fn
        optimizer = self.optimizer
        model = self.model
        train_data = self.train_data
        patience = self.patience

        best_val_loss = np.inf

        # For each epoch
        for epoch in range(self.n_epochs + 1):
            train_acc = 0
            train_loss = 0

            # Generate batches of augmented training samples.
            batches = self.generate_batches(train_data)
            for batch in batches:
                samples, labels = batch
                # Use the model to predict the labels for each sample.
                predictions = model.forward(samples)

                # Compute the loss and take one step along the gradient.
                optimizer.zero_grad()
                loss = loss_fn(predictions, labels)
                loss.backward()
                optimizer.step()

                train_acc += self.compute_accuracy(labels, predictions)
                train_loss += loss.detach().item()

            # Report training loss, training accuracy, validation loss, and validation accuracy.
            train_loss /= len(batches)
            train_acc /= len(batches)
            val_loss, val_acc = self.validate()

            self.losses["ta"].append(train_acc)
            self.losses["va"].append(val_acc)
            self.losses["tl"].append(train_loss)
            self.losses["vl"].append(val_loss)
            self.losses["epoch"].append(epoch)

            print("EPOCH {}\nTRAIN_LOSS: {:7.4f}\nTRAIN_ACC: {:7.4f}\nVAL_LOSS: {:7.4f}\nVAL_ACC: {:7.4f}\n".format(
                epoch, train_loss, train_acc, val_loss, val_acc))

            # If enough epochs have passed that we need to save the model, do so.
            if val_acc > self.best_val_acc:
                print("NEW BEST ACCURACY", val_acc, epoch)
                self.best_val_acc = val_acc
                self.save_model(epoch)
            
            if val_loss > best_val_loss:
                patience -= 1
            else:
                best_val_loss = val_loss
                patience = self.patience
            if patience == 0 and epoch > 200:
                break

            if epoch % self.save_every_n == 0:
                self.save_model(epoch)

    def generate_batches(self, data):
        """
        Function to split a dataset into random augmented batches.

        :param data: Array of samples to choose from. Either `self.train_data` or `self.val_data`
        :return: Batches of augmented samples and the appropriate labels.
        """

        batches = []
        transform_prob = 0.2

        # Shuffle the indices at which we will access the dataset.
        indices = [i for i in range(len(data))]
        np.random.shuffle(indices)

        bs = self.batch_size
        for i in range(len(data) // bs):
            # Choose our random batch.
            idxs = indices[i * bs:i * bs + bs]
            batch = data[idxs]

            samples = []
            labels = []

            # For each sample in the selected batch.
            for training_sample in batch:
                # Append the augmented sample to the batch.
                samples.append(training_sample[0])

                # Append the appropriate binary label for this sample to the batch.
                label = training_sample[1]
                labels.append(label)

            # Cast batch to tensor for PyTorch.
            samples = torch.as_tensor(samples, dtype=torch.float32)
            if np.random.uniform(0, 1) < transform_prob:
                tf = transforms.RandomRotation(degrees=np.random.randint(0, 365))
                samples = tf(samples)

            if np.random.uniform(0, 1) < transform_prob:
                tf = transforms.RandomHorizontalFlip()
                samples = tf(samples)

            if np.random.uniform(0, 1) < transform_prob:
                tf = transforms.RandomVerticalFlip()
                samples = tf(samples)

            # if np.random.uniform(0, 1) < transform_prob:
            #     tf = transforms.RandomAutocontrast()
            #     samples = tf(samples)
            #
            # if np.random.uniform(0, 1) < transform_prob:
            #     tf = transforms.RandomAdjustSharpness(sharpness_factor=np.random.uniform(0, 10))
            #     samples = tf(samples)

            factors = []
            for j in range(len(samples)):
                max_factor = 65535./torch.max(samples[j])
                min_factor = 0.5
                factors.append(np.random.uniform(min_factor, max_factor))

            samples *= np.asarray(factors).astype(np.float32).reshape(-1, 1, 1, 1)
            labels = torch.as_tensor(labels, dtype=torch.float32)

            batches.append((samples, labels))

        # Return augmented batch.
        return batches

    @torch.no_grad()
    def validate(self):
        """
        Function to compute the validation loss and accuracy on a random batch of validation data.
        :return: Computed loss and accuracy.
        """

        # Set the model to evaluation mode.
        self.model.eval()

        # Generate a random augmented batch of validation data.
        # samples, labels = self.generate_batch(self.val_data)

        samples, labels = self.val_data

        # Compute loss and accuracy of model on the generated batch.
        predictions = self.model.forward(samples)
        loss = self.loss_fn(predictions, labels).item()
        acc = self.compute_accuracy(labels, predictions)

        # Set the model back to training mode.
        self.model.train()

        # Return the computed loss and accuracy values.
        return loss, acc

    @torch.no_grad()
    def compute_accuracy(self, labels, predictions):
        """
        Function to compute the accuracy of a batch of predictions given a batch of labels.
        :param labels: Ground-truth labels to compare to.
        :param predictions: Predicted labels from the model.
        :return: Computed accuracy.
        """

        predicted_labels = predictions.argmax(dim=-1)
        known_labels = labels.argmax(dim=-1)
        n_samples = labels.shape[0]

        n_correct = torch.where(predicted_labels == known_labels, 1, 0).sum()

        acc = 100*n_correct / n_samples
        return acc.item()

    def load_data(self, folder_path):
        """
        Function to load all positive and negative samples given a folder. This assumes there are two folders inside the
        specified folder, such that the file paths `folder_path/positive` and `folder_path/negative` exist. Positive
        samples will be loaded from `folder_path/positive` and negative samples from `folder_path/negative`. The
        resulting data will be split into a training set and validation set for training.

        :param folder_path: Folder to load from.
        :return: None
        """

        positive_sample_folder = os.path.join(folder_path, "positive")
        negative_sample_folder = os.path.join(folder_path, "negative")
        n_negative = 0
        data = []
        label_counts = {key:0 for key in self.code_map.keys()}

        # For each image in the positive samples folder.
        for file_name in os.listdir(positive_sample_folder):
            if not file_name.endswith(".png"):
              continue       
            if "set" in file_name:
                end = file_name.find("_")
                code = file_name[:end].strip()
            else:
                # CG: Legacy code for square particles
                #code = file_name[:file_name.find("(")].strip()
                # CG: For gear particles,
                code = file_name[file_name.find("("):file_name.find("_")].strip()
            # Load region.
            region = cv2.imread(os.path.join(positive_sample_folder, file_name), cv2.IMREAD_ANYDEPTH)
            label = self.one_hot(self.code_map[code])
            label_counts[code] += 1

            # Append region and positive label to dataset.
            data.append([region.reshape(1, *region.shape), label])

        n_positive = len(data)
        print("Loaded {} positive training samples.".format(n_positive))
        print("Label counts:")
        for key, value in label_counts.items():
            print("{} | {}".format(key, value))

        # Randomly shuffle all loaded samples.
        np.random.shuffle(data)

        # Split resulting dataset into training and validation sets.
        split = int(round(len(data)*self.val_split))
        self.train_data = np.asarray(data[split:])
        val_data = np.asarray(data[:split])

        v_labels = []
        v_regions = []
        for region, label in val_data:
            v_labels.append(label)
            v_regions.append(region)
        self.val_data = (torch.as_tensor(v_regions, dtype=torch.float32), torch.as_tensor(v_labels, dtype=torch.float32))

    def save_model(self, epoch):
        """
        Function to save the parameters of a model during training. Models will be named `model_{epoch}.pt` and saved
        in the folder `self._model_save_path`
        :param epoch: Current training epoch.
        :return: None.
        """

        path = self.model_save_path
        model_save_file = os.path.join(path, "model_{}.pt".format(epoch))
        train_csv_path = "data/code_training_losses.csv"

        print(path)
        if not os.path.exists(path):
            os.makedirs(path)

        torch.save(self.model.state_dict(), model_save_file)
        with open(train_csv_path, 'w') as f:
            ls = self.losses
            f.write("Epoch,Training Accuracy,Validation Accuracy,Training Loss,Validation Loss\n")
            for i in range(epoch):
                f.write("{},{},{},{},{}\n".format(ls["epoch"][i], ls["ta"][i], ls["va"][i], ls["tl"][i], ls["vl"][i]))

    def one_hot(self, value):
        arr = [0 for _ in range(self.num_codes)]
        arr[value] = 1
        return arr
