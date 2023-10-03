from code_classification import CodeClassifier
import numpy as np
import cv2
import os
import torch
import torch.nn as nn
import torch.optim as optim
import pathlib
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from datetime import datetime
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

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


class CodeClassifierTrainerGPU(object):
    def __init__(
        self,
        codes=None,
        model_save_path='data/models/code_classifier',
        save_every_n: int = 10,
        batch_size: int = 256,
        lr: float = 1e-5,
        fc_size: int = 256,
        fc_num: int = 2,
        dropout_rate: float = 0.1,
        k: int = None,
        patience: int = 10,
        verbose: bool = True,
        log: bool = True,
        timestamp: str = datetime.now().strftime('%m_%d_%y_%H:%M'),
    ):
        # Prints out augmented images if set to true
        self.debug = False

        # Printing verbosity
        self.verbose = verbose

        # Activates confusion matrices each epoch, which activates calculation of metrics like precision/recall
        self.cm = True

        # Activates printing confusion matrices each epoch
        self.cm_fig = True

        # CG: CPU or GPU, prioritizes GPU if available.
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        # Print availability to GPU
        if self.verbose:
            print('CUDA Availability: ' + str(torch.cuda.is_available()))

        # Number of codes for the multi-class model
        n_codes = len(codes)
        self.model = CodeClassifier(
            n_codes, fc_size=fc_size, fc_num=fc_num, dropout_rate=dropout_rate
        )

        # Print the model architecture as a sanity check
        if self.verbose:
            print('\nCode Classifier Model Architecture:')
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

        self.num_codes = n_codes

        # Python list indexing starts at 0, but our class labels start at 1
        # Let's just confirm the mapping between our class labels and internal label indexing,
        self.code_map = {code: idx for idx, code in enumerate(codes)}
        if self.verbose:
            print(
                f'Code Map Between Sample Filenames and Internal Code Integer Designation:\n{self.code_map}'
            )

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
        self.learning_rate = lr
        # Keep track of the best validation accuracy
        self.best_val_acc = 0
        # When early-stopping based on some loss
        # We often want to early-stop based on some finite precision.
        # In other words, if the loss improves by 10^-6, this is not real improvement for practical purposes
        # Here, we define a minimum amount of improvement for the loss function
        self.early_stop_delta = 0.1

        self.test_acc_for_best_val = 0

        # Store the training, validation, test accuracy and training, validation, test loss
        self.losses = {
            'epoch': [],
            'ta': [],
            'va': [],
            'test_acc': [],
            'tl': [],
            'vl': [],
            'test_loss': [],
            'tp': [],
            'vp': [],
            'test_precision': [],
            'tr': [],
            'vr': [],
            'test_recall': [],
            'tf1': [],
            'vf1': [],
            'test_f1': [],
        }
        # How many epochs to wait before stopping training if the model does not improve
        # This is an early-stopping hyperparameter
        self.patience = patience

        # Tensorboard
        self.log = log
        self.writer = None

        # How many epochs to wait before early-stopping is allowed.
        self.warmup = 20

        # Using the Adam optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        # Cross Entropy Loss for multi-class problems
        self.loss_fn = nn.CrossEntropyLoss()
        # Current cross-validation fold
        self.k = k

        # If logging via tensorboard, define a dedicated writer to log the results
        if self.log:
            # If cross-validating,
            if k is not None:
                self.writer = SummaryWriter(
                    os.path.join(self.model_save_path, self.log_timestamp, "fold_" + str(self.k), 'logs')
                )
            else:
                self.writer = SummaryWriter(
                    os.path.join(self.model_save_path, self.log_timestamp, 'logs')
                )

    def train(self, cross_validation: bool = False, cross_validation_scores=None):
        """
        Function to train a classifier on hologram regions.
        :return: None.
        """
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
        for epoch in range(self.n_epochs):
            # Reset training accuracy and loss to zero
            train_acc = 0
            train_loss = 0

            # Generate batches of augmented training samples.
            # CG: These augmented samples are confirmed to be reasonable with commit #36a7ce4
            batches = self.generate_batches(train_data)
            len_batches = len(batches)
            len_stat_vector = self.num_codes
            # Initialize classifier scores as 0
            train_precision = 0
            train_recall = 0
            train_f1_score = 0
            # For each generated batch,
            for batch in tqdm(
                batches, desc='Epoch ' + str(epoch) + ':', disable=not self.verbose
            ):
                # Clear gradients
                optimizer.zero_grad()

                # Get the samples and labels
                samples, labels = batch
                # Moving the model to GPU is in-place, but moving the data is not.
                samples = samples.to(self.device)
                labels = labels.to(self.device)
                # Use the model to predict the labels for each sample.
                predictions = model.forward(samples)

                # Compute the loss and take one step along the gradient.
                # Our barcodes are labelled from 1 ... N
                # But the cross-entropy loss function accepts class labels from 0 ... N-1
                # Here, we just decrement by 1 to match this convention
                # CrossEntropyLoss() accepts unnormalized prediction logits
                loss = loss_fn(predictions.to(torch.float32), (labels - 1).to(torch.long))

                loss.backward()
                optimizer.step()

                # Compute the accuracy
                train_acc += self.compute_accuracy(
                    labels.clone().detach(), predictions.clone().detach()
                )
                train_loss += loss.detach().item()

                save_path = os.path.join(self.model_save_path, self.log_timestamp, "fold_" + str(self.k), 'confusion', 'train')
                train_precision_batch, train_recall_batch, train_f1_score_batch = self.confusion_matrix_plot(labels, predictions, epoch=epoch, save_root=save_path, activate=self.cm, save_fig=self.cm_fig)
                
                # Correct for undefined results
                train_precision_batch[np.isnan(train_precision_batch)] = 0
                train_recall_batch[np.isnan(train_recall_batch)] = 0
                train_f1_score_batch[np.isnan(train_f1_score_batch)] = 0

                # Based on bugs or label imbalance, it is possible that a label could be excluded from the confusion matrix
                # Only compose a precision, recall, f1score for consistent # of labels,

                # This satisfies an edge case of inconsistent vector lengths resulting from the sklearn confusion matrix function
                # If vector lengths are inconsistent, then the vectors-per-batch cannot be added together.
                # If we happen to find a vector of larger length than the current largest length,
                # then call that the new largest length
                if len(train_precision_batch) > len_stat_vector:
                    len_stat_vector = len(train_precision_batch)

                if len(train_precision_batch) == len_stat_vector:
                    train_precision += train_precision_batch
                    train_recall += train_recall_batch
                    train_f1_score += train_f1_score_batch
                # Otherwise, if we found a batch with inconsistent vector lengths, decrement the # of batches
                # We do this to properly compute an average without considering the problematic batch..
                else:
                    len_batches - 1

            # Report training loss, training accuracy, validation loss, validation accuracy, and test loss/accuracy.
            # This is a per-batch average of the loss and accuracy, making the training robust to different batch sizes/gradient estimates
            train_loss /= len(batches)
            train_acc /= len(batches)

            if train_precision.all() != None and train_recall.all() != None:
                train_precision /= len(batches)
                train_recall /= len(batches)
                train_f1_score /= len(batches)

            val_loss, val_acc, val_precision, val_recall, val_f1_score = self.validate(epoch=epoch)
            # Correct for undefined results
            val_precision[np.isnan(val_precision)] = 0
            val_recall[np.isnan(val_recall)] = 0
            val_f1_score[np.isnan(val_f1_score)] = 0

            test_loss, test_acc, test_precision, test_recall, test_f1_score = self.test(epoch=epoch)
            # Correct for undefined results
            test_precision[np.isnan(test_precision)] = 0
            test_recall[np.isnan(test_recall)] = 0
            test_f1_score[np.isnan(test_f1_score)] = 0

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
                writer.add_scalars(
                    'Mean Precision',
                    {'Train_Mean_Precision': np.mean(train_precision) if train_precision.all() != None else -999, 'Val_Mean_Precision': np.mean(val_precision) if val_precision.all() != None else -999, 'Test_Mean_Precision': np.mean(test_precision) if test_precision.all() != None else -999},
                    epoch,
                )
                writer.add_scalars(
                    'Mean Recall',
                    {'Train_Mean_Recall': np.mean(train_recall) if train_recall.all() != None else -999, 'Val_Mean_Recall': np.mean(val_recall) if val_recall.all() != None else -999, 'Test_Mean_Recall': np.mean(test_recall) if test_recall.all() != None else -999},
                    epoch,
                )
                writer.add_scalars(
                    'Mean F1 Score',
                    {'Train_F1_Score': np.mean(train_f1_score) if train_f1_score.all() != None else -999, 'Val_F1_Score': np.mean(val_f1_score) if val_f1_score.all() != None else -999, 'Test_F1_Score': np.mean(test_f1_score) if test_f1_score.all() != None else -999},
                    epoch,
                )

                writer.add_scalar('Train_Loss', train_loss, epoch)
                writer.add_scalar('Train_Acc', train_acc, epoch)

                writer.add_scalar('Val_Loss', val_loss, epoch)
                writer.add_scalar('Val_Acc', val_acc, epoch)

                writer.add_scalar('Test_Loss', test_loss, epoch)
                writer.add_scalar('Test_Acc', test_acc, epoch)

                writer.add_scalar('Patience (Early Stopping)', patience, epoch)

                writer.add_scalar('Train_Mean_Precision', np.mean(train_precision) if train_precision.all() != None else -999, epoch)
                writer.add_scalar('Train_Mean_Recall', np.mean(train_recall) if train_recall.all() != None else -999, epoch)
                writer.add_scalar('Train_Mean_F1_Score', np.mean(train_f1_score) if train_f1_score.all() != None else -999, epoch)

                writer.add_scalar('Val_Mean_Precision', np.mean(val_precision) if val_precision.all() != None else -999, epoch)
                writer.add_scalar('Val_Mean_Recall', np.mean(val_recall) if val_recall.all() != None else -999, epoch)
                writer.add_scalar('Val_Mean_F1_Score', np.mean(val_f1_score) if val_f1_score.all() != None else -999, epoch)

                writer.add_scalar('Test_Mean_Precision', np.mean(test_precision) if test_precision.all() != None else -999, epoch)
                writer.add_scalar('Test_Mean_Recall', np.mean(test_recall) if test_recall.all() != None else -999, epoch)
                writer.add_scalar('Test_Mean_F1_Score', np.mean(test_f1_score) if test_f1_score.all() != None else -999, epoch)

            self.losses['ta'].append(train_acc)
            self.losses['va'].append(val_acc)
            self.losses['test_acc'].append(test_acc)

            self.losses['tl'].append(train_loss)
            self.losses['vl'].append(val_loss)
            self.losses['test_loss'].append(test_loss)

            self.losses['tp'].append(np.mean(train_precision) if train_precision.all() != None else -999)
            self.losses['vp'].append(np.mean(val_precision) if val_precision.all() != None else -999)
            self.losses['test_precision'].append(np.mean(test_precision) if test_precision.all() != None else -999)

            self.losses['tr'].append(np.mean(train_recall) if train_recall.all() != None else -999)
            self.losses['vr'].append(np.mean(val_recall) if val_recall.all() != None else -999)
            self.losses['test_recall'].append(np.mean(test_recall) if test_recall.all() != None else -999)

            self.losses['tf1'].append(np.mean(train_f1_score) if train_f1_score.all() != None else -999)
            self.losses['vf1'].append(np.mean(val_f1_score) if val_f1_score.all() != None else -999)
            self.losses['test_f1'].append(np.mean(test_f1_score) if test_f1_score.all() != None else -999)

            self.losses['epoch'].append(epoch)

            # If enough epochs have passed that we need to save the model, do so.
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.test_acc_for_best_val = test_acc
                if self.verbose:
                    print(f'(New Best Val. Acc., Correspond. Test Acc., Epoch):\n({self.best_val_acc}, {self.test_acc_for_best_val}, {epoch})\n')
                self.save_model(epoch, save_name='best_model.pth')

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

            if epoch % self.save_every_n == 0 and self.log:
                self.save_model(epoch)

        if self.log:
            # Save changes to hard drive and close tensorboard writer in memory.
            writer.flush()
            writer.close()

        # If cross-validating, then add the current fold scores to the running cross-validation counts of accuracy and loss
        if cross_validation:
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
            samples = torch.as_tensor(np.array(samples, dtype=np.float32), dtype=torch.float32)

            print_images(
                samples/65535,
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
                    single_image_pil = transforms.ToPILImage()(single_image)
                    tf = transforms.RandomRotation(degrees=np.random.randint(0, 365))
                    single_image_pil = tf(single_image_pil)
                    # Convert back to PyTorch tensor when done.
                    sample = transforms.ToTensor()(single_image_pil)
                    temp_samples.append(sample)

                # Cast batch to tensor for PyTorch.
                samples = torch.as_tensor(
                    np.array(samples, dtype=np.float32), dtype=torch.float
                )

                print_images(
                    samples/65535,
                    path='data/classifier_training_samples/Data_Augmentation_Inspection/Rotations',
                    batch_id=str(i),
                    activate=self.debug,
                )

            if np.random.uniform(0, 1) < transform_prob:
                tf = transforms.RandomHorizontalFlip()
                samples = tf(samples)

                print_images(
                    samples/65535,
                    path='data/classifier_training_samples/Data_Augmentation_Inspection/HorizontalFlip',
                    batch_id=str(i),
                    activate=self.debug,
                )

            if np.random.uniform(0, 1) < transform_prob:
                tf = transforms.RandomVerticalFlip()
                samples = tf(samples)

                print_images(
                    samples/65535,
                    path='data/classifier_training_samples/Data_Augmentation_Inspection/VerticalFlip',
                    batch_id=str(i),
                    activate=self.debug,
                )

            labels = torch.as_tensor(np.array(labels, dtype=np.int32), dtype=torch.int32)

            batches.append((samples, labels))

        # Return augmented batch.
        return batches

    @torch.no_grad()
    def confusion_matrix_plot(self, labels, predictions, save_root: str, epoch = None, activate: bool = False, save_fig: bool = False):
        
        if activate:
            pathlib.Path(save_root).mkdir(parents=True, exist_ok=True)

            labels=labels.squeeze(dim=-1).to(torch.int32).cpu().numpy() - 1
            predicted_labels=predictions.argmax(dim=-1).cpu().numpy()

            cm = confusion_matrix(labels, predicted_labels)
            precision = np.diag(cm) / np.sum(cm, axis = 0)
            recall = np.diag(cm) / np.sum(cm, axis = 1)
            f1_score = 2*((precision*recall) / (precision + recall))

            if save_fig:
                confusion_matrix_display = ConfusionMatrixDisplay(confusion_matrix=cm).plot()
                confusion_matrix_display.figure_.savefig(os.path.join(save_root, f'cm_epoch{str(epoch)}'),dpi=72)
            
            plt.close(fig='all')
            
            return (precision, recall, f1_score)

        else:
            return (None, None, None)

    @torch.no_grad()
    def validate(self, epoch=None):
        """
        Function to compute the validation loss and accuracy on a random batch of validation data.
        :return: Computed loss and accuracy.
        """

        # Set the model to evaluation mode.
        self.model.eval()

        # Moving the model to GPU is in-place, but moving data is not.
        samples, labels = self.val_data
        samples = samples.to(self.device)
        samples = torch.unsqueeze(samples, dim=1)
        labels = labels.to(self.device)

        # Compute loss and accuracy of model on the generated batch.
        predictions = self.model.forward(samples)
        # CrossEntropyLoss() accepts unnormalized prediction logits
        loss = self.loss_fn(predictions, (labels - 1).to(torch.long)).item()
        acc = self.compute_accuracy(labels.clone().detach(), predictions.clone().detach())

        if epoch is not None:
            save_path = os.path.join(self.model_save_path, self.log_timestamp, "fold_" + str(self.k), 'confusion', 'validate')
            precision, recall, f1_score = self.confusion_matrix_plot(labels.clone().detach(), predictions.clone().detach(), save_root=save_path, epoch=epoch, activate=self.cm, save_fig=self.cm_fig)

        # Set the model back to training mode.
        self.model.train()

        # Return the computed loss and accuracy values.
        return loss, acc, precision, recall, f1_score

    @torch.no_grad()
    def test(self, epoch=None):
        """
        Function to compute the test loss and accuracy on a pre-defined test dataset data.
        :return: Computed loss and accuracy.
        """

        # Set the model to evaluation mode.
        self.model.eval()

        # Moving the model to GPU is in-place, but moving data is not.
        samples, labels = self.test_data
        samples = samples.to(self.device)
        samples = torch.unsqueeze(samples, dim=1)
        labels = labels.to(self.device)

        # Compute loss and accuracy of model on the generated batch.
        predictions = self.model.forward(samples)
        # CrossEntropyLoss() accepts unnormalized prediction logits
        loss = self.loss_fn(predictions, (labels - 1).to(torch.long)).item()
        acc = self.compute_accuracy(labels.clone().detach(), predictions.clone().detach())

        if epoch is not None:
            save_path = os.path.join(self.model_save_path, self.log_timestamp, "fold_" + str(self.k), 'confusion', 'test')
            precision, recall, f1_score = self.confusion_matrix_plot(labels.clone().detach(), predictions.clone().detach(), save_root=save_path, epoch=epoch, activate=self.cm, save_fig=self.cm_fig)

        # Set the model back to training mode.
        self.model.train()

        # Return the computed loss and accuracy values.
        return loss, acc, precision, recall, f1_score

    @torch.no_grad()
    def compute_accuracy(self, labels, logits, softmax=nn.Softmax(dim=1)):
        """
        Function to compute the accuracy of a batch of predictions given a batch of labels.
        :param labels: Ground-truth labels to compare to.
        :param predicted_labels: Predicted labels from the model.
        :return: Computed accuracy.
        """

        labels = labels.to(torch.int64)
        softmax_logits = softmax(logits)
        predicted_labels = softmax_logits.argmax(dim=1) + 1
        n_samples = labels.shape[0]
        n_correct = torch.where(predicted_labels == labels, 1, 0).sum()
        acc = 100 * n_correct / n_samples

        return acc.item()

    def load_data(
        self,
        train_dataset_np: np.ndarray = None,
        train_targets_np: np.ndarray = None,
        train_idx=None,
        val_idx=None,
        test_dataset_np: np.ndarray = None,
        test_targets_np: np.ndarray = None,
    ):
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
        assert test_dataset_np is not None

        self.train_data = np.take(train_dataset_np, train_idx, axis=0)
        val_data = np.take(train_dataset_np, val_idx, axis=0)
        val_targets = np.take(train_targets_np, val_idx, axis=0)

        # Setting up validation dataset
        v_labels = []
        v_regions = []
        for region, label in zip(val_data, val_targets):
            v_labels.append(label)
            v_regions.append(np.array(region[0][0], dtype=np.float32))
        v_labels = torch.as_tensor(np.array(v_labels, dtype=np.int32), dtype=torch.int32)
        v_regions = torch.as_tensor(np.array(v_regions), dtype=torch.float32)

        print_images(
            v_regions/65535,
            path='data/classifier_training_samples/Validation_Dataset/',
            batch_id='val',
            activate=self.debug,
        )

        self.val_data = (v_regions, v_labels)

        # Setting up test dataset
        t_labels = []
        t_regions = []
        for region, label in zip(test_dataset_np, test_targets_np):
            t_labels.append(label)
            t_regions.append(np.array(region[0][0], dtype=np.float32))
        t_labels = torch.as_tensor(np.array(t_labels, dtype=np.int32), dtype=torch.int32)
        t_regions = torch.as_tensor(np.array(t_regions), dtype=torch.float32)

        print_images(
            t_regions/65535,
            path='data/classifier_training_samples/Test_Dataset/',
            batch_id='test',
            activate=self.debug,
        )

        self.test_data = (t_regions, t_labels)

    def save_model(self, epoch, save_name: str = None):
        """
        Function to save the parameters of a model during training. Models will be named `model_{epoch}.pt` and saved
        in the folder `self._model_save_path`
        :param epoch: Current training epoch.
        :return: None.
        """

        path = os.path.join(self.model_save_path, self.log_timestamp, "fold_" + str(self.k), "checkpoints")
        if save_name is None:
            model_save_file = os.path.join(path, 'fold_{}_{}.pt'.format(self.k, epoch))
        else:
            assert '.pth' in save_name
            model_save_file = os.path.join(path, save_name)
        train_csv_path = os.path.join(self.model_save_path, self.log_timestamp, "fold_" + str(self.k), 'region_classifier_learning_curves.csv')

        if self.verbose:
            print(path)
        if not os.path.exists(path):
            os.makedirs(path)

        torch.save(self.model.state_dict(), model_save_file)
        with open(train_csv_path, 'w') as f:
            ls = self.losses
            f.write(
                'Epoch,Training Accuracy,Validation Accuracy,Test Accuracy,Training Loss,Validation Loss,Test Loss,Training Mean Precision,Validation Mean Precision,Test Mean Precision,Training Mean Recall,Validation Mean Recall,Test Mean Recall,Training Mean F1 Score,Validation Mean F1 Score,Test Mean F1 Score\n'
            )
            for i in range(epoch):
                f.write(
                    '{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}\n'.format(
                        ls['epoch'][i],
                        ls['ta'][i],
                        ls['va'][i],
                        ls['test_acc'][i],
                        ls['tl'][i],
                        ls['vl'][i],
                        ls['test_loss'][i],
                        ls['tp'][i],
                        ls['vp'][i],
                        ls['test_precision'][i],
                        ls['tr'][i],
                        ls['vr'][i],
                        ls['test_recall'][i],
                        ls['tf1'][i],
                        ls['vf1'][i],
                        ls['test_f1'][i],
                    )
                )
