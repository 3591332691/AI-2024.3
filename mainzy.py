import os
import argparse
import numpy as np
import pandas as pd
import time

from sklearn.preprocessing import label_binarize
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from fea import feature_extraction

from Bio.PDB import PDBParser


class SVMModel:
    def __init__(self, kernel='rbf', C=1.0):
        self.model = SVC(kernel=kernel, C=C, probability=True, max_iter=1000)

    def train(self, train_data, train_targets):
        self.model.fit(train_data, train_targets)

    def evaluate(self, data, targets):
        return self.model.score(data, targets)


class LRModel:
    def __init__(self, C=1.0):
        self.model = LogisticRegression(C=C, max_iter=1000)

    def train(self, train_data, train_targets):
        self.model.fit(train_data, train_targets)

    def evaluate(self, data, targets):
        return self.model.score(data, targets)


class LinearSVMModel:
    def __init__(self, C=1.0):
        self.model = LinearSVC(C=C, max_iter=1000)

    def train(self, train_data, train_targets):
        self.model.fit(train_data, train_targets)

    def evaluate(self, data, targets):
        return self.model.score(data, targets)


def data_preprocess(args):
    if args.ent:
        # diagrams = feature_extraction()[0]
        diagrams = np.load('./data/diagrams_pca.npy')
    else:
        diagrams = np.load('./data/diagrams.npy')
    cast = pd.read_table('./data/SCOP40mini_sequence_minidatabase_19.cast')
    cast.columns.values[0] = 'protein'

    data_list = []
    target_list = []
    for task in range(1, 56):
        task_col = cast.iloc[:, task]
        # todo
        train_indices = []
        test_indices = []

        # find indices that satisfy the conditions: 1,2-"train" 3,4-"test"
        train_indices = np.where((task_col == 1) | (task_col == 2))[0]
        test_indices = np.where((task_col == 3) | (task_col == 4))[0]

        # Select training and testing data from 'diagrams' using the indices found above
        train_data = diagrams[train_indices]
        test_data = diagrams[test_indices]

        # Set the targets for training and testing data:
        train_targets = np.where(task_col.iloc[train_indices] == 1, 1, 0)
        test_targets = np.where(task_col.iloc[test_indices] == 3, 1, 0)

        data_list.append((train_data, test_data))
        target_list.append((train_targets, test_targets))

    return data_list, target_list


def main(args):
    data_list, target_list = data_preprocess(args)

    task_acc_train = []
    task_acc_test = []

    # Model Initialization based on input argument
    if args.model_type == 'svm':
        model = SVMModel(kernel=args.kernel, C=args.C)
    else:
        print("Attention: Kernel option is not supported")
        if args.model_type == 'linear_svm':
            model = LinearSVMModel(C=args.C)
        elif args.model_type == 'lr':
            model = LRModel(C=args.C)
        else:
            raise ValueError("Unsupported model type")
    train_times = []
    for i in range(len(data_list)):
        train_data, test_data = data_list[i]
        train_targets, test_targets = target_list[i]

        print(f"Processing dataset {i + 1}/{len(data_list)}")

        # Record the start time for training
        start_train_time = time.time()

        # Train the model
        model.train(train_data, train_targets)

        # Record the end time for training and calculate the duration
        end_train_time = time.time()
        train_duration = end_train_time - start_train_time

        # Append the training time for this dataset
        train_times.append(train_duration)

        # Evaluate the model
        train_accuracy = model.evaluate(train_data, train_targets)
        test_accuracy = model.evaluate(test_data, test_targets)

        print(f"Dataset {i + 1}/{len(data_list)} - Train Accuracy: {train_accuracy}, Test Accuracy: {test_accuracy}")

        task_acc_train.append(train_accuracy)
        task_acc_test.append(test_accuracy)

    # Calculate and print the average training time
    average_train_time = sum(train_times) / len(train_times)
    print(f"Average Training Time: {average_train_time:.2f} seconds")
    print("Training accuracy:", sum(task_acc_train) / len(task_acc_train))
    print("Testing accuracy:", sum(task_acc_test) / len(task_acc_test))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SVM Model Training and Evaluation")
    parser.add_argument('--model_type', type=str, default='svm', choices=['svm', 'linear_svm', 'lr'], help="Model type")
    parser.add_argument('--kernel', type=str, default='linear', choices=['linear', 'poly', 'rbf', 'sigmoid'],
                        help="Kernel type")
    parser.add_argument('--C', type=float, default=20, help="Regularization parameter")
    parser.add_argument('--ent', action='store_true',
                        help="Load data from a file using a feature engineering function feature_extraction() from fea.py")
    args = parser.parse_args()
    main(args)