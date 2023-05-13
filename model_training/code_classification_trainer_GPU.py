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
from datetime import datetime

# Prints augmented images out for debugging
def print_augmented_image(sampleBatchTensor, 
                          path: str = None,
                          batchID: str = None,
                          # Activate image saving
                          activate: bool = False):
    if activate:
        if not os.path.exists(path):
            os.makedirs(path)
        imagePILTF = transforms.ToPILImage()
        for imageInd in range(len(sampleBatchTensor)):
            imageTensor = sampleBatchTensor[imageInd]
            imagePIL = imagePILTF(imageTensor)
            filename = batchID + "_" + str(imageInd) + ".png"
            imagePIL.save(os.path.join(path, filename))

class CodeClassifierTrainerGPU(object):
    def __init__(self, codes=None, 
                 model_save_path="data/models/code_classifier",
                 save_every_n: int = 10,
                 batch_size: int = 256,
                 verbose: bool = True,
                 log: bool = True,
                 timestamp: str = datetime.now().strftime("%m_%d_%y_%H:%M"),
                 k: int = None):
        
        # CG: CPU or GPU, prioritizes GPU if available.
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        self.verbose = verbose
        # Print availability to GPU
        if self.verbose:
            print("CUDA Availability: " + str(torch.cuda.is_available()))

        # Number of codes for the multi-class model
        n_codes = len(codes)
        self.model = CodeClassifier(n_codes)
        # Move to GPU if available
        self.model.to(self.device)
        # Check if model is on GPU
        if self.verbose:
            print("Model Loaded to GPU: " + str(next(self.model.parameters()).is_cuda))
        
        self.train_data = None
        self.val_data = None

        self.num_codes = n_codes

        # Python list indexing starts at 0, but our class labels start at 1
        # Let's just confirm the mapping between our class labels and internal label indexing,
        self.code_map = {code: idx for idx, code in enumerate(codes)}
        print(f"Code Map Between Sample Filenames and Internal Code Integer Designation:\n{self.code_map}")

        # Save the model at this path
        self.model_save_path = model_save_path
        # Save the model every n epochs
        self.save_every_n = save_every_n

        # Save the model results as a timestamp
        self.log_timestamp = timestamp
        
        # Hyper-parameters.
        # Batch Size
        self.batch_size = batch_size
        # Num Epochs
        self.n_epochs = 20000
        # Learning rate for optimizer
        self.learning_rate = 1e-5
        # Validation split variable (DEPRECATED)
        #self.val_split = 0.2
        # Max Transform Sequence (Deprecated)
        #self.max_transform_sequence = 10
        # Keep track of the best validation accuracy
        self.best_val_acc = 0
        # Store the training, validation, test accuracy and training, validation, test loss
        self.losses = {"epoch": [], 
                       "ta": [], 
                       "va": [], 
                       "test_acc": [], 
                       "tl": [], 
                       "vl": [], 
                       "test_loss": []}
        # How many epochs to wait before stopping training if the model does not improve
        # This is an early-stopping hyperparameter
        self.patience = 10

        # Tensorboard
        self.log = log
        self.writer = None

        # How many epochs to wait before early-stopping is allowed.
        self.warmup = 20

        # Using the Adam optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        # Cross Entropy Loss for multi-class problems
        self.loss_fn = nn.CrossEntropyLoss()

        # If logging via tensorboard, define a dedicated writer to log the results
        if self.log:
            self.writer = SummaryWriter(os.path.join(self.model_save_path, self.log_timestamp, "logs"))

    def train(self, 
              crossVal=False, 
              crossValScores = {"Val_Loss": [], 
                                "Val_Acc": [], 
                                "Test_Loss": [], 
                                "Test_Acc": []}):
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
        
        writer = self.writer
        verbose = self.verbose
        # Default loss should be infinite, as this corresponds to the worst possible value
        best_val_loss = np.inf
        self.test_loss_for_best_val = np.inf
        self.test_acc_for_best_val = np.inf

        # For each epoch
        for epoch in range(self.n_epochs + 1):
            # Reset training accuracy and loss to zero
            train_acc = 0
            train_loss = 0

            # Generate batches of augmented training samples.
            # CG: These augmented samples are confirmed to be reasonable with commit #36a7ce4
            batches = self.generate_batches(train_data)
            # For each generated batch,
            for batch in tqdm(batches, desc="Epoch " + str(epoch) + ":", disable=not verbose):
                # Clear gradients
                optimizer.zero_grad()
                
                #print("TRAIN BATCH")
                #print(batch)
                # Get the samples and labels
                samples, labels = batch
                # Moving the model to GPU is in-place, but moving the data is not.
                samples = samples.to(self.device)
                labels = labels.to(self.device)

                # Use the model to predict the labels for each sample.
                predictions = model.forward(samples)
                predictedLabels = ((torch.argmax(predictions, dim=1) + 1).float()).clone().detach().requires_grad_(True)

                # Compute the loss and take one step along the gradient.
                loss = loss_fn(predictedLabels, labels)

                loss.backward()
                optimizer.step()

                # Compute the accuracy
                train_acc += self.compute_accuracy(labels, predictedLabels)
                train_loss += loss.detach().item()

            # Report training loss, training accuracy, validation loss, validation accuracy, and test loss/accuracy.
            # This is a per-batch average of the loss and accuracy, making the training robust to different batch sizes/gradient estimates
            train_loss /= len(batches)
            train_acc /= len(batches)
            # Validation loss and accuracy
            val_loss, val_acc = self.validate()
            # Test loss and accuracy
            test_loss, test_acc = self.test()

            # Write the loss and accuracies to tensorboard
            writer.add_scalars("Loss", {"Train_Loss":train_loss, 
                                        "Val_Loss":val_loss, 
                                        "Test_Loss":test_loss}, epoch)
            writer.add_scalars("Accuracy", {"Train_Acc":train_acc, 
                                            "Val_Acc":val_acc, 
                                            "Test_Acc":test_acc}, epoch)
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

            # CG: Legacy Code
            #print("EPOCH {}\nTRAIN_LOSS: {:7.4f}\nTRAIN_ACC: {:7.4f}\nVAL_LOSS: {:7.4f}\nVAL_ACC: {:7.4f}\n".format(
            #    epoch, train_loss, train_acc, val_loss, val_acc))

            # If enough epochs have passed that we need to save the model, do so.
            if val_acc > self.best_val_acc:
                if verbose:
                    print("NEW BEST VAL. ACCURACY", val_acc, epoch)
                self.best_val_acc = val_acc
                self.test_acc_for_best_val = test_acc
                self.save_model(epoch)

            # If the loss is greater than the best loss (i.e., the current minimum loss)
            if val_loss > best_val_loss:
                # If we are past the warmup stage,
                if epoch > warmup:
                    # Lower the patience of how long to wait for the model accuracy to improve
                    patience -= 1
            # If our new loss is best,
            else:
                # Update the best loss with the new loss
                best_val_loss = val_loss
                # Update the test loss corresponding to the best validation loss
                self.test_loss_for_best_val = test_loss
                # Reset the patience because the model improved
                patience = self.patience
            # If we are out of training patience and past the warmup stage,
            if patience == 0 and epoch > warmup:
                break

            if epoch % self.save_every_n == 0:
                self.save_model(epoch)
        # Save changes to hard drive and close tensorboard writer in memory.
        writer.flush()
        writer.close()
        
        # If cross-validating, then add the current fold scores to the running cross-validation counts of accuracy and loss
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

        # For saving augmented images for debugging,
        debug = False

        batches = []
        transform_prob = 0.2

        # Shuffle the indices at which we will access the dataset.
        indices = [i for i in range(len(data))]
        np.random.shuffle(indices)

        bs = self.batch_size
        for i in range(len(data) // bs):
            # Choose our random batch.
            idxs = indices[i*bs:i*bs+bs]
            batch = data[idxs]

            # CG: Enable if not work
            #print("BATCH TO AUGMENT")
            #print(batch)

            samples = []
            labels = []

            # For each sample in the selected batch.
            for training_sample in batch:
                # Append the augmented sample to the batch.
                samples.append(training_sample[0])

                # Append the appropriate label for this sample to the batch.
                label = training_sample[1]
                labels.append(label)

            # Cast batch to tensor for PyTorch.
            samples = torch.as_tensor(np.array(samples, dtype=np.int32), dtype=torch.float32)
            #print("Samples before Augmentation")
            #print(samples)
            print_augmented_image(samples,
                                  path="data/classifier_training_samples/Data_Augmentation_Inspection/NoAugment",
                                  batchID=str(i),
                                  activate=debug)

            #if np.random.uniform(0, 1) < transform_prob:
            #    tf = transforms.GaussianBlur(kernel_size=np.random.choice([2*i+1 for i in range(10)]))
            #    samples = tf(samples)
            #    print("Samples after Gaussian blur")
            #    print(samples)
            #    print_augmented_image(samples,
            #                        path="data/classifier_training_samples/Data_Augmentation_Inspection/GaussBlur",
            #                        batchID=str(i))

            # Need to loop through and rotate all image samples in the batch and readd them to the list
            samplesTemp = []
            if np.random.uniform(0, 1) < transform_prob:
                for image in samples:
                    # Reshape to from (batch_size, channels, height, width) to (channels, height, width)
                    # Need to convert to Python Image Library (PIL) image representation for rotations to be proper
                    singleImage = image.permute(0, 1, 2)
                    #print("SINGLEIMAGE")
                    #print(singleImage.shape)
                    singleImagePIL = transforms.ToPILImage()(singleImage)
                    tf = transforms.RandomRotation(degrees=np.random.randint(0, 365))
                    singleImagePIL = tf(singleImagePIL)
                    # Convert back to PyTorch tensor when done.
                    sample = transforms.ToTensor()(singleImagePIL)
                    samplesTemp.append(sample)
                # Cast batch to tensor for PyTorch.
                samples = torch.as_tensor(np.array(samples, dtype=np.int32), dtype=torch.float32)
                #print("Samples after random rotation")
                #print(samples)
                print_augmented_image(samples,
                                    path="data/classifier_training_samples/Data_Augmentation_Inspection/Rotations",
                                    batchID=str(i),
                                    activate=debug)

            if np.random.uniform(0, 1) < transform_prob:
                tf = transforms.RandomHorizontalFlip()
                samples = tf(samples)
                #print("Samples after random horizontal flip")
                #print(samples)
                print_augmented_image(samples,
                                    path="data/classifier_training_samples/Data_Augmentation_Inspection/HorizontalFlip",
                                    batchID=str(i),
                                    activate=debug)

            if np.random.uniform(0, 1) < transform_prob:
                tf = transforms.RandomVerticalFlip()
                samples = tf(samples)
                #print("Samples after random vertical flip")
                #print(samples)
                print_augmented_image(samples,
                                    path="data/classifier_training_samples/Data_Augmentation_Inspection/VerticalFlip",
                                    batchID=str(i),
                                    activate=debug)


            #if np.random.uniform(0, 1) < transform_prob:
            #    tf = transforms.RandomAutocontrast()
            #    samples = tf(samples)
            #    print("Samples after random auto contrast")
            #    print(samples)

            #if np.random.uniform(0, 1) < transform_prob:
            #    tf = transforms.RandomAdjustSharpness(sharpness_factor=np.random.uniform(0.5, 10))
            #    samples = tf(samples)
            #    print("Samples after random adjust sharpness")
            #    print(samples)

            #if np.random.uniform(0, 1) < transform_prob:
            #    tf = transforms.RandomInvert()
            #    samples = tf(samples)
            #    print("Samples after random invert")
            #    print(samples)

            # Legacy
            #factors = []
            
            #for j in range(len(samples)):
            #    max_factor = 65535./torch.max(samples[j])
            #    min_factor = 0.5
            #    factors.append(np.random.uniform(min_factor, max_factor))

            # Inconsistent torch and numpy calls
            #samples *= np.asarray(factors).astype(np.float32).reshape(-1, 1, 1, 1)

            #samples *= torch.tensor(factors).reshape(-1, 1, 1, 1)
            
            #labels = torch.as_tensor(labels, dtype=torch.float32)
            labels = torch.as_tensor(np.array(labels, dtype=np.int32), dtype=torch.float32)

            # CG: Enable if not work
            #print("Augmented samples")
            #print(samples)
            #print("Augmented Labels")
            #print(labels)
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
        samples = torch.unsqueeze(samples, dim=1)
        labels = labels.to(self.device)

        # Compute loss and accuracy of model on the generated batch.
        predictions = self.model.forward(samples)
        predictedLabels = (torch.argmax(predictions, dim=1) + 1).float()
        loss = self.loss_fn(predictedLabels, labels).item()
        acc = self.compute_accuracy(labels, predictedLabels)

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
        samples = torch.unsqueeze(samples, dim=1)
        labels = labels.to(self.device)

        # Compute loss and accuracy of model on the generated batch.
        predictions = self.model.forward(samples)
        predictedLabels = (torch.argmax(predictions, dim=1) + 1).float()
        loss = self.loss_fn(predictedLabels, labels).item()
        acc = self.compute_accuracy(labels, predictedLabels)

        # Set the model back to training mode.
        self.model.train()

        # Return the computed loss and accuracy values.
        return loss, acc
    
    @torch.no_grad()
    def compute_accuracy(self, labels, predictedLabels):
        """
        Function to compute the accuracy of a batch of predictions given a batch of labels.
        :param labels: Ground-truth labels to compare to.
        :param predictedLabels: Predicted labels from the model.
        :return: Computed accuracy.
        """

        predicted_labels = predictedLabels.argmax(dim=-1) + 1
        #print("predicted_labels")
        #print(predicted_labels)
        #print("labels")
        #print(labels)

        n_samples = labels.shape[0]

        n_correct = torch.where(predicted_labels == labels, 1, 0).sum()

        acc = 100*n_correct / n_samples
        #print("acc")
        #print(acc)

        return acc.item()

    def load_data(self, folder_path: str, 
                  trainDatasetNP = None,
                  trainTargetsNP = None, 
                  train_idx = None,
                  val_idx = None, 
                  test_dataset: np.ndarray = None,
                  test_targets: np.ndarray = None):
        """
        Function to load all positive and negative samples given a folder. This assumes there are two folders inside the
        specified folder, such that the file paths `folder_path/positive` and `folder_path/negative` exist. Positive
        samples will be loaded from `folder_path/positive` and negative samples from `folder_path/negative`. The
        resulting data will be split into a training set and validation set for training.

        :param folder_path: Folder to load from.
        :return: None
        """
        # Ensuring a train/val/test split
        assert train_idx is not None
        assert val_idx is not None
        assert test_dataset is not None

        self.train_data = np.take(trainDatasetNP, train_idx, axis=0)
        train_targets = np.take(trainTargetsNP, train_idx, axis=0)
        val_data = np.take(trainDatasetNP, val_idx, axis=0)
        val_targets = np.take(trainTargetsNP, val_idx, axis=0)
        
        # Setting up validation dataset
        v_labels = []
        v_regions = []
        for region, label in zip(val_data, val_targets):
            v_labels.append(label)
            v_regions.append(np.array(region[0][0], dtype=np.float32))

        v_labels = np.array(v_labels, dtype=np.int32)
        v_regions = np.array(v_regions, dtype=np.int32)

        self.val_data = (torch.as_tensor(v_regions, dtype=torch.float32), torch.as_tensor(v_labels, dtype=torch.float32))

        # Setting up test dataset
        t_labels = []
        t_regions = []
        for region, label in zip(test_dataset, train_targets):
            t_labels.append(label)
            t_regions.append(np.array(region[0][0], dtype=np.float32))
        t_labels = np.array(t_labels, dtype=np.int32)
        t_regions = np.array(t_regions, dtype=np.int32)

        self.test_data = (torch.as_tensor(t_regions, dtype=torch.float32), torch.as_tensor(t_labels, dtype=torch.float32))

        print("self.train_data")
        print(self.train_data)
        print("self.val_data")
        print(self.val_data)
        print("self.test_data")
        print(self.test_data)

        # Legacy Code
        '''positive_sample_folder = os.path.join(folder_path, "positive")
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
                # CG: For gear particles, this is just a new filename search criteria for how the samples are labeled,
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
        self.val_data = (torch.as_tensor(v_regions, dtype=torch.float32), torch.as_tensor(v_labels, dtype=torch.float32))'''

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
            f.write("Epoch,Training Accuracy,Validation Accuracy,Test Accuracy,Training Loss,Validation Loss,Test Loss\n")
            for i in range(epoch):
                f.write("{},{},{},{},{},{},{}\n".format(ls["epoch"][i], ls["ta"][i], ls["va"][i], ls["test_acc"][i], ls["tl"][i], ls["vl"][i], ls["test_loss"][i]))

    def one_hot(self, value):
        arr = [0 for _ in range(self.num_codes)]
        arr[value] = 1
        return arr
