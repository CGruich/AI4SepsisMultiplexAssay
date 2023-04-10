from object_detection import RegionClassifier
import numpy as np
import cv2
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from datetime import datetime

class RegionClassifierTrainerGPU(object):
    def __init__(self, model_save_path="data/models/region", hpoTrial = None, verbose: bool = True, log: bool = True, timestamp: str = datetime.now().strftime("%m_%d_%y_%H:%M"), k: int = None):
        # Printing verbosity
        self.verbose = verbose

        # CG: CPU or GPU, prioritizes GPU if available.
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if self.verbose:
            print("CUDA Availability: " + str(torch.cuda.is_available()))
        
        self.dropout = 0.3
        self.h1 = 64

        self.train_data = None
        self.val_data = None

        self.model_save_path = model_save_path
        self.save_every_n = 10
        
        self.log_timestamp = timestamp

        # Hyper-parameters.
        self.batch_size = 192
        self.n_epochs = 500
        self.learning_rate = 3e-4
        self.beta1 = 0.95
        self.epsilon=1e-08
        self.val_split = 0.2
        self.max_transform_sequence = 10
        self.losses = {"epoch": [], "ta": [], "va": [], "test_acc": [], "tl": [], "vl": [], "test_loss": []}
        self.best_val_acc = 0
        self.patience = 10

        # Loss function
        self.loss_fn = nn.BCELoss()
        
        # Tensorboard
        self.log = True
        self.writer = None
        
        # Hyperparameter Optimization
        self.hpo = False

        # Specific hyperparameter trial to test if hyperparameter optimizing
        self.hpoTrial = hpoTrial

        # If running grid search hyperparameter optimization,
        if self.hpoTrial is not None:
            for key, value in self.hpoTrial.items():
                if key == "Batch_Size":
                    self.batch_size = value
                elif key == "lr":
                    self.learning_rate = value
                elif key == "Dropout_Rate":
                    self.dropout = value
                elif key == "Weight_Decay_Beta1":
                    self.beta1 = value
                elif key == "Epsilon":
                    self.epsilon = value
                elif key == "FC_Size":
                    self.h1 = value
                elif key == "hpoID":
                    self.hpoID = "hpoID_" + str(value)
                    # If cross-validating, then save each hyperparameter trial with a folder for each k-fold.
                    if k is not None:
                        self.hpoLogPath = os.path.join(model_save_path, self.log_timestamp, self.hpoID, self.hpoID + "_" + str(k))
                    # Otherwise if not cross-validating but still hyperparameter optimizing
                    else:
                        self.hpoLogPath = os.path.join(model_save_path, self.log_timestamp, self.hpoID)
        
        self.model = RegionClassifier(h1=self.h1, dropout=self.dropout, verbose=self.verbose)
        self.model.dropout = self.dropout
        self.model.h1 = self.h1

        self.model.to(self.device)
        if self.verbose:
            print("Model Loaded to GPU: " + str(next(self.model.parameters()).is_cuda))

        # How many epochs to wait before early-stopping is allowed.
        self.warmup = 10

        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate, betas=(self.beta1, 0.999), eps=self.epsilon)

        if self.log:
            if self.hpoTrial is not None:
                self.writer = SummaryWriter(self.hpoLogPath)
            else:
                self.writer = SummaryWriter(os.path.join(self.model_save_path, self.log_timestamp, "logs"))

    def train(self, crossVal=False, crossValScores = {"Val_Loss": [], "Val_Acc": [], "Test_Loss": [], "Test_Acc": []}):
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
        warmup = self.warmup
        log = self.log

        writer = self.writer
        hpoTrial = self.hpoTrial
        verbose = self.verbose
        best_val_loss = np.inf
        self.test_loss_for_best_val = np.inf
        self.test_acc_for_best_val = np.inf

        # For each epoch
        for epoch in range(self.n_epochs+1):
            train_acc = 0
            train_loss = 0

            # Generate batches of augmented training samples.
            batches = self.generate_batches(train_data)
            for batch in tqdm(batches, desc="Epoch " + str(epoch) + ":", disable=not verbose):

                samples, labels = batch
                # Moving the model to GPU is in-place, but moving the data is not.
                samples = samples.to(self.device)
                labels = labels.to(self.device)
                
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
            test_loss, test_acc = self.test()

            writer.add_scalars("Loss", {"Train_Loss":train_loss, "Val_Loss":val_loss, "Test_Loss":test_loss}, epoch)
            writer.add_scalars("Accuracy", {"Train_Acc":train_acc, "Val_Acc":val_acc, "Test_Acc":test_acc}, epoch)
            writer.add_scalar("Train_Loss", train_loss, epoch)
            writer.add_scalar("Train_Acc", train_acc, epoch)
            writer.add_scalar("Val_Loss", val_loss, epoch)
            writer.add_scalar("Val_Acc", val_acc, epoch)
            writer.add_scalar("Test_Loss", test_loss, epoch)
            writer.add_scalar("Test_Acc", test_acc, epoch)
            writer.add_scalar("Patience (Early Stopping)", patience, epoch)

            self.losses["ta"].append(train_acc)
            self.losses["va"].append(val_acc)
            self.losses["test_acc"].append(test_acc)
            self.losses["tl"].append(train_loss)
            self.losses["vl"].append(val_loss)
            self.losses["test_loss"].append(test_loss)
            self.losses["epoch"].append(epoch)

            #print("EPOCH {}\nTRAIN_LOSS: {:7.4f}\nTRAIN_ACC: {:7.4f}\nVAL_LOSS: {:7.4f}\nVAL_ACC: {:7.4f}\n".format(
            #      epoch, train_loss, train_acc, val_loss, val_acc))

            # If enough epochs have passed that we need to save the model, do so.
            if val_acc > self.best_val_acc:
                if verbose:
                    print("NEW BEST VAL. ACCURACY", val_acc, epoch)
                self.best_val_acc = val_acc
                self.test_acc_for_best_val = test_acc
                if hpoTrial is None:
                    self.save_model(epoch)

            if val_loss > best_val_loss:
                if epoch > warmup:
                    patience -= 1
            else:
                best_val_loss = val_loss
                self.test_loss_for_best_val = test_loss
                patience = self.patience
            if patience == 0 and epoch > warmup:
                break

            if epoch % self.save_every_n == 0 and hpoTrial is None:
                self.save_model(epoch)
        # Save changes to hard drive and close tensorboard writer in memory.
        writer.flush()
        writer.close()
        
        if crossVal:
            crossValScores["Val_Loss"].append(best_val_loss)
            crossValScores["Val_Acc"].append(self.best_val_acc)
            crossValScores["Test_Loss"].append(self.test_loss_for_best_val)
            crossValScores["Test_Acc"].append(self.test_acc_for_best_val)
            return crossValScores

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
        for i in range(len(data)//bs):
            # Choose our random batch.
            idxs = indices[i*bs:i*bs+bs]
            batch = data[idxs]

            samples = []
            labels = []
            
            # For each sample in the selected batch.
            for training_sample in batch:
                # Append the augmented sample to the batch.
                samples.append(training_sample[0])
    
                # Append the appropriate binary label for this sample to the batch.
                label = (1, 0) if training_sample[1] == 0 else (0, 1)
                labels.append(label)
    
            # Cast batch to tensor for PyTorch.
            samples = torch.as_tensor(np.array(samples, dtype=np.int32), dtype=torch.float32)
            if np.random.uniform(0, 1) < transform_prob:
                tf = transforms.GaussianBlur(kernel_size=np.random.choice([2*i+1 for i in range(10)]))
                samples = tf(samples)
    
            if np.random.uniform(0, 1) < transform_prob:
                tf = transforms.RandomRotation(degrees=np.random.randint(0, 365))
                samples = tf(samples)
    
            if np.random.uniform(0, 1) < transform_prob:
                tf = transforms.RandomHorizontalFlip()
                samples = tf(samples)
    
            if np.random.uniform(0, 1) < transform_prob:
                tf = transforms.RandomVerticalFlip()
                samples = tf(samples)
    
            if np.random.uniform(0, 1) < transform_prob:
                tf = transforms.RandomAutocontrast()
                samples = tf(samples)
    
            if np.random.uniform(0, 1) < transform_prob:
                tf = transforms.RandomAdjustSharpness(sharpness_factor=np.random.uniform(0, 10))
                samples = tf(samples)
    
            if np.random.uniform(0, 1) < transform_prob:
                tf = transforms.RandomInvert()
                samples = tf(samples)
    
            labels = torch.as_tensor(np.array(labels, dtype=np.int32), dtype=torch.float32)
            
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
        
        # Moving the model to GPU is in-place, but moving data is not.
        samples, labels = self.val_data
        samples = samples.to(self.device)
        labels = labels.to(self.device)

        # Compute loss and accuracy of model on the generated batch.
        predictions = self.model.forward(samples)
        loss = self.loss_fn(predictions, labels).item()
        acc = self.compute_accuracy(labels, predictions)

        # Set the model back to training mode.
        self.model.train()

        # Return the computed loss and accuracy values.
        return loss, acc

    @torch.no_grad()
    def test(self):
        """
        Function to compute the test loss and accuracy on a pre-defined test dataset data.
        :return: Computed loss and accuracy.
        """

        # Set the model to evaluation mode.
        self.model.eval()

        # Generate a random augmented batch of validation data.
        # samples, labels = self.generate_batch(self.val_data)
        
        # Moving the model to GPU is in-place, but moving data is not.
        samples, labels = self.test_data
        samples = samples.to(self.device)
        labels = labels.to(self.device)

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

        diff = (predicted_labels - known_labels).abs().sum()
        acc = 100*(n_samples - diff) / n_samples
        return acc.item()

    def load_data(self, folder_path: str, datasetNP, train_idx = None, val_idx = None, test_dataset: np.ndarray = None):
        if self.hpoTrial is not None:
            # Ensuring a train/val/test split during hyperparameter optimization
            assert train_idx is not None
            assert val_idx is not None
            assert test_dataset is not None

            self.train_data = np.take(datasetNP, train_idx, axis=0)
            val_data = np.take(datasetNP, val_idx, axis=0)

            # Setting up validation dataset
            v_labels = []
            v_regions = []
            for region, label in val_data:
                v_labels.append((1, 0) if label == 0 else (0, 1))
                v_regions.append(region)
            v_labels = np.array(v_labels, dtype=np.int32)
            v_regions = np.array(v_regions, dtype=np.int32)

            self.val_data = (torch.as_tensor(v_regions, dtype=torch.float32), torch.as_tensor(v_labels, dtype=torch.float32))

            # Setting up training dataset
            t_labels = []
            t_regions = []
            for region, label in test_dataset:
                t_labels.append((1, 0) if label == 0 else (0, 1))
                t_regions.append(region)
            t_labels = np.array(t_labels, dtype=np.int32)
            t_regions = np.array(t_regions, dtype=np.int32)

            self.test_data = (torch.as_tensor(t_regions, dtype=torch.float32), torch.as_tensor(t_labels, dtype=torch.float32))
        
        else:
            '''
            Function to load all positive and negative samples given a folder. This assumes there are two folders inside the
            specified folder, such that the file paths `folder_path/positive` and `folder_path/negative` exist. Positive
            samples will be loaded from `folder_path/positive` and negative samples from `folder_path/negative`. The
            resulting data will be split into a training set and validation set for training.

            :param folder_path: Folder to load from.
            :return: None'''

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

            print("Loaded {} negative training samples.".format(len(data) - n_positive))

            # CG: Deprecated for cross-validation scheme.
            # This deprecated scheme uses one validation dataset, which may be hard or easy
            # Opting for cross-validation, more robust.

            # Randomly shuffle all loaded samples.
            np.random.shuffle(data)

            # Split resulting dataset into training and validation sets.
            split = int(round(len(data)*self.val_split))
            self.train_data = np.asarray(data[split:])
            val_data = np.asarray(data[:split])

            v_labels = []
            v_regions = []
            for region, label in val_data:
                v_labels.append((1, 0) if label == 0 else (0, 1))
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
        train_csv_path = "data/region_training_losses.csv"

        if not os.path.exists(path):
            os.makedirs(path)

        torch.save(self.model.state_dict(), model_save_file)
        with open(train_csv_path, 'w') as f:
            ls = self.losses
            f.write("Epoch,Training Accuracy,Validation Accuracy,Test Accuracy,Training Loss,Validation Loss,Test Loss\n")
            for i in range(epoch):
                f.write("{},{},{},{},{},{},{}\n".format(ls["epoch"][i], ls["ta"][i], ls["va"][i], ls["test_acc"][i], ls["tl"][i], ls["vl"][i], ls["test_loss"][i]))