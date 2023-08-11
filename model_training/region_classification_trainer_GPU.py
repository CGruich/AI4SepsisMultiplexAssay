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

# Prints augmented images out for debugging


def print_images(
    sample_batch_tensor,
    path: str = None,
    batch_id: str = None,
    # Activate image saving
    activate: bool = False,
):
    if activate:
        if not os.path.exists(path):
            os.makedirs(path)
        image_pil_tf = transforms.ToPILImage()
        for image_index in range(len(sample_batch_tensor)):
            image_tensor = sample_batch_tensor[image_index]
            image_pil = image_pil_tf(image_tensor)
            filename = batch_id + '_' + str(image_index) + '.png'
            image_pil.save(os.path.join(path, filename))


class RegionClassifierTrainerGPU(object):
    def __init__(
        self,
        model_save_path: str = 'data/models/region',
        save_every_n: int = 10,
        batch_size: int = 192,
        lr: float = 3e-4,
        fc_size: int = 64,
        fc_num: int = 2,
        dropout_rate: float = 0.3,
        verbose: bool = True,
        log: bool = True,
        timestamp: str = datetime.now().strftime('%m_%d_%y_%H:%M'),
    ):
        # Prints out augmented images if set to true
        self.debug = False

        # Printing verbosity
        self.verbose = verbose

        # CG: CPU or GPU, prioritizes GPU if available.
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        # Print availability to GPU
        if self.verbose:
            print('CUDA Availability: ' + str(torch.cuda.is_available()))

        self.model = RegionClassifier(
            fc_size=fc_size, 
            fc_num=fc_num, 
            dropout_rate=dropout_rate,
        )

        # Print the model architecture as a sanity check
        if self.verbose:
            print('\nRegion Classifier Model Architecture:')
            print(self.model)
            print('\n')

        # Move to GPU if available
        self.model.to(self.device)
        # Check if model is on GPU
        if self.verbose:
            print('Model Loaded to GPU: ' + str(next(self.model.parameters()).is_cuda))

        self.train_data = None
        self.val_data = None
        self.test_data = None

        # Save the model at this path
        self.model_save_path = model_save_path
        # Save the model every n epochs
        self.save_every_n = save_every_n

        # Save the model results as a timestamp
        self.log_timestamp = timestamp

        # Hyper-parameters.
        # Dropout Rate
        self.dropout = dropout_rate
        # Fully-connected layer size for all fully-connected layers
        self.h1 = fc_size
        # Batch size
        self.batch_size = batch_size
        # Num Epochs
        self.n_epochs = 20000
        # Learning rate for optimizer
        self.learning_rate = lr
        # Keep track of the best validation accuracy
        self.best_val_acc = 0

        self.test_loss_for_best_val = None
        self.test_acc_for_best_val = 0

        # Adam optimizer hyperparam
        # self.beta1 = 0.95
        # Adam optimizer hyperparam
        # self.epsilon = 1e-08
        self.val_split = 0.2
        self.max_transform_sequence = 10
        self.losses = {
            'epoch': [],
            'ta': [],
            'va': [],
            'test_acc': [],
            'tl': [],
            'vl': [],
            'test_loss': [],
        }
        # How many epochs to wait before stopping training if the model does not improve
        # This is an early-stopping hyperparameter
        self.patience = 10

        # Tensorboard
        self.log = log
        self.writer = None

        self.model.dropout = dropout_rate
        self.model.h1 = fc_size

        # How many epochs to wait before early-stopping is allowed.
        self.warmup = 10

        # Using the Adam optimizer
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.learning_rate,
            # betas=(self.beta1, 0.999),
            # eps=self.epsilon,
        )
        # Loss function for binary classification problems, cross-entropy
        self.loss_fn = nn.BCELoss()

        # If logging via tensorboard, define a dedicated writer to log the results
        if self.log:
            self.writer = SummaryWriter(
                os.path.join(self.model_save_path, self.log_timestamp, 'logs')
            )

    def train(self, cross_validate=False, cross_validation_scores=None):
        """
        Function to train a classifier on hologram regions.
        :return: None.
        """

        # Mutable default values should not be defined in the function parameter list.
        if cross_validation_scores is None:
            cross_validation_scores = {
                'Val_Loss': [],
                'Val_Acc': [],
                'Test_Loss': [],
                'Test_Acc': [],
            }

        # Set the PyTorch model to training mode.
        self.model.train()

        # Doing this lets us access these four variables much faster than if we accessed them through `self` every time.
        loss_fn = self.loss_fn
        optimizer = self.optimizer
        model = self.model
        train_data = self.train_data
        patience = self.patience
        warmup = self.warmup

        if self.log:
            writer = self.writer
        # Default loss should be infinite, as this corresponds to the worst possible value
        best_val_loss = np.inf
        self.test_loss_for_best_val = np.inf
        self.test_acc_for_best_val = np.inf

        # For each epoch
        for epoch in range(self.n_epochs + 1):
            train_acc = 0
            train_loss = 0

            # Generate batches of augmented training samples.
            batches = self.generate_batches(train_data)
            for batch in tqdm(
                batches, desc='Epoch ' + str(epoch) + ':', disable=not self.verbose
            ):

                # Compute the loss and take one step along the gradient.
                optimizer.zero_grad()

                samples, labels = batch
                # Moving the model to GPU is in-place, but moving the data is not.
                samples = samples.to(self.device)
                labels = labels.to(self.device)

                # Use the model to predict the labels for each sample.
                predictions = model.forward(samples)

                loss = loss_fn(predictions.to(torch.float64), labels.to(torch.int64))

                loss.backward()
                optimizer.step()

                train_acc += self.compute_accuracy(labels.clone().detach(), predictions.clone().detach())
                train_loss += loss.detach().item()

            # Report training loss, training accuracy, validation loss, validation accuracy, and test loss/accuracy.
            train_loss /= len(batches)
            train_acc /= len(batches)
            val_loss, val_acc = self.validate()
            test_loss, test_acc = self.test()

            if self.log:
                writer.add_scalars(
                    'Loss',
                    {'Train_Loss': train_loss, 'Val_Loss': val_loss, 'Test_Loss': test_loss,},
                    epoch,
                )
                writer.add_scalars(
                    'Accuracy',
                    {'Train_Acc': train_acc, 'Val_Acc': val_acc, 'Test_Acc': test_acc},
                    epoch,
                )
                writer.add_scalar('Train_Loss', train_loss, epoch)
                writer.add_scalar('Train_Acc', train_acc, epoch)
                writer.add_scalar('Val_Loss', val_loss, epoch)
                writer.add_scalar('Val_Acc', val_acc, epoch)
                writer.add_scalar('Test_Loss', test_loss, epoch)
                writer.add_scalar('Test_Acc', test_acc, epoch)
                writer.add_scalar('Patience (Early Stopping)', patience, epoch)

            self.losses['ta'].append(train_acc)
            self.losses['va'].append(val_acc)
            self.losses['test_acc'].append(test_acc)
            self.losses['tl'].append(train_loss)
            self.losses['vl'].append(val_loss)
            self.losses['test_loss'].append(test_loss)
            self.losses['epoch'].append(epoch)

            # If enough epochs have passed that we need to save the model, do so.
            if val_acc > self.best_val_acc:
                if self.verbose:
                    print('NEW BEST VAL. ACCURACY', val_acc, epoch)
                self.best_val_acc = val_acc
                self.test_acc_for_best_val = test_acc
                self.save_model(epoch)

            # If our new loss is best,
            if val_loss <= (best_val_loss - self.early_stop_delta):
                # Update the best loss with the new loss
                best_val_loss = val_loss
                # Update the test loss corresponding to the best validation loss
                self.test_loss_for_best_val = test_loss
                # Reset the patience because the model improved
                patience = self.patience
            # If the loss is greater than the best loss (i.e., the current minimum loss)
            else:
                # If we are out of training patience and past the warmup stage,
                # If we are past the warmup stage,
                if epoch > warmup:
                    # Lower the patience of how long to wait for the model accuracy to improve
                    patience -= 1
            if patience == 0 and epoch > warmup:
                break

            if epoch % self.save_every_n == 0:
                self.save_model(epoch)

        if self.log:
            # Save changes to hard drive and close tensorboard writer in memory.
            writer.flush()
            writer.close()

        # If cross-validating, then add the current fold scores to the running cross-validation counts of accuracy and loss
        if cross_validate:
            cross_validation_scores['Val_Loss'].append(best_val_loss)
            cross_validation_scores['Val_Acc'].append(self.best_val_acc)
            cross_validation_scores['Test_Loss'].append(self.test_loss_for_best_val)
            cross_validation_scores['Test_Acc'].append(self.test_acc_for_best_val)
            return cross_validation_scores

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
            idxs = indices[i * bs : i * bs + bs]
            batch = data[idxs]

            # CG: Enable if not work
            # print("BATCH TO AUGMENT")
            # print(batch)

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
            # print("Samples before Augmentation")
            # print(samples)
            print_images(
                samples,
                path='data/classifier_training_samples/Data_Augmentation_Inspection/NoAugment',
                batch_id=str(i),
                activate=self.debug,
            )

            # Need to loop through and rotate all image samples in the batch and readd them to the list
            temp_samples = []
            if np.random.uniform(0, 1) < transform_prob:
                for image in samples:
                    # Reshape to from (batch_size, channels, height, width) to (channels, height, width)
                    # Need to convert to Python Image Library (PIL) image representation for rotations to be proper
                    single_image = image.permute(0, 1, 2)
                    # print("single_image")
                    # print(single_image.shape)
                    single_image_pil = transforms.ToPILImage()(single_image)
                    tf = transforms.RandomRotation(degrees=np.random.randint(0, 365))
                    single_image_pil = tf(single_image_pil)
                    # Convert back to PyTorch tensor when done.
                    sample = transforms.ToTensor()(single_image_pil)
                    temp_samples.append(sample)

                # Cast batch to tensor for PyTorch.
                samples = torch.as_tensor(
                    np.array(samples, dtype=np.int32), dtype=torch.float32
                )
                # print("Samples after random rotation")
                # print(samples)
                print_images(
                    samples,
                    path='data/classifier_training_samples/Data_Augmentation_Inspection/Rotations',
                    batch_id=str(i),
                    activate=self.debug,
                )

            if np.random.uniform(0, 1) < transform_prob:
                tf = transforms.RandomHorizontalFlip()
                samples = tf(samples)
                # print("Samples after random horizontal flip")
                # print(samples)
                print_images(
                    samples,
                    path='data/classifier_training_samples/Data_Augmentation_Inspection/HorizontalFlip',
                    batch_id=str(i),
                    activate=self.debug,
                )

            if np.random.uniform(0, 1) < transform_prob:
                tf = transforms.RandomVerticalFlip()
                samples = tf(samples)
                # print("Samples after random vertical flip")
                # print(samples)
                print_images(
                    samples,
                    path='data/classifier_training_samples/Data_Augmentation_Inspection/VerticalFlip',
                    batch_id=str(i),
                    activate=self.debug,
                )

            labels = torch.as_tensor(np.array(labels, dtype=np.int32), dtype=torch.float32)

            # CG: Enable if not work
            # print("Augmented samples")
            # print(samples)
            # print("Augmented Labels")
            # print(labels)
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
        samples, labels = self.val_data
        samples = samples.to(self.device)
        samples = torch.unsqueeze(samples, dim=1)
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
        acc = 100 * (n_samples - diff) / n_samples
        return acc.item()

    def load_data(
        self,
        folder_path: str,
        train_dataset_np=None,
        train_targets_np=None,
        train_idx=None,
        val_idx=None,
        test_dataset: np.ndarray = None,
    ):
        """
        Function to load all positive and negative samples given a folder. This assumes there are two folders inside the
        specified folder, such that the file paths `folder_path/positive` and `folder_path/negative` exist. Positive
        samples will be loaded from `folder_path/positive` and negative samples from `folder_path/negative`. The
        resulting data will be split into a training set and validation set for training.

        :param folder_path: Folder to load from.
        :return: None"""

        # Ensuring a train/val/test split
        assert train_idx is not None
        assert val_idx is not None
        assert test_dataset is not None

        self.train_data = np.take(train_dataset_np, train_idx, axis=0)
        train_targets = np.take(train_targets_np, train_idx, axis=0)
        val_data = np.take(train_dataset_np, val_idx, axis=0)
        val_targets = np.take(train_targets_np, val_idx, axis=0)

        # Setting up validation dataset
        v_labels = []
        v_regions = []
        for region, label in zip(val_data, val_targets):
            v_labels.append(label)
            v_regions.append(np.array(region[0][0], dtype=np.float32))

        v_labels = torch.as_tensor(np.array(v_labels, dtype=np.int32), dtype=torch.float32)
        v_regions = torch.as_tensor(np.array(v_regions, dtype=np.int32), dtype=torch.float32)

        print_images(
            v_regions,
            path='data/classifier_training_samples/Validation_Dataset/',
            batch_id='val',
            activate=self.debug,
        )

        self.val_data = (v_regions, v_labels)

        # Setting up test dataset
        t_labels = []
        t_regions = []
        for region, label in zip(test_dataset, train_targets):
            t_labels.append(label)
            t_regions.append(np.array(region[0][0], dtype=np.float32))
        t_labels = torch.as_tensor(np.array(t_labels, dtype=np.int32), dtype=torch.float32)
        t_regions = torch.as_tensor(np.array(t_regions, dtype=np.int32), dtype=torch.float32)

        print_images(
            t_regions,
            path='data/classifier_training_samples/Test_Dataset/',
            batch_id='test',
            activate=self.debug,
        )

        self.test_data = (t_regions, t_labels)

        # print("self.train_data")
        # print(self.train_data)
        # print("self.val_data")
        # print(self.val_data)
        # print("self.test_data")
        # print(self.test_data)

    def save_model(self, epoch):
        """
        Function to save the parameters of a model during training. Models will be named `model_{epoch}.pt` and saved
        in the folder `self._model_save_path`
        :param epoch: Current training epoch.
        :return: None.
        """

        path = self.model_save_path
        model_save_file = os.path.join(path, 'model_{}.pt'.format(epoch))
        train_csv_path = 'data/region_training_losses.csv'

        if self.verbose:
            print(path)
        if not os.path.exists(path):
            os.makedirs(path)

        torch.save(self.model.state_dict(), model_save_file)
        with open(train_csv_path, 'w') as f:
            ls = self.losses
            f.write(
                'Epoch,Training Accuracy,Validation Accuracy,Test Accuracy,Training Loss,Validation Loss,Test Loss\n'
            )
            for i in range(epoch):
                f.write(
                    '{},{},{},{},{},{},{}\n'.format(
                        ls['epoch'][i],
                        ls['ta'][i],
                        ls['va'][i],
                        ls['test_acc'][i],
                        ls['tl'][i],
                        ls['vl'][i],
                        ls['test_loss'][i],
                    )
                )
