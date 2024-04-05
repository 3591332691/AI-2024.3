import os
import argparse
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from sklearn.preprocessing import label_binarize
from sklearn.svm import SVC, LinearSVC
from fea import feature_extraction

from Bio.PDB import PDBParser


class SVMModel:
    def __init__(self, kernel='rbf', C=1.0):
        self.model = SVC(kernel=kernel, C=C, probability=True)

    def train(self, train_data, train_targets):
        self.model.fit(train_data, train_targets)

    def evaluate(self, data, targets):
        return self.model.score(data, targets)


class LRModel:

    def __init__(self, C=1.0):
        """
        初始化逻辑回归模型。

        参数:
        - C (float): 正则化强度的倒数；必须为正浮点数。默认值为 1.0。
        """
        self.model = LogisticRegression(C=C)

    def train(self, train_data, train_targets):
        """
        训练逻辑回归模型。

        参数:
        - train_data (array-like): 训练数据。
        - train_targets (array-like): 训练数据的目标值。
        """
        self.model.fit(train_data, train_targets)

    def evaluate(self, data, targets):
        """
        评估逻辑回归模型的性能。

        参数:
        - data (array-like): 待评估的数据。
        - targets (array-like): 数据对应的真实目标值。

        返回值:
        - float: 模型在给定数据上的准确率得分。
        """
        return self.model.score(data, targets)



class LinearSVMModel:
    def __init__(self, C=1.0):
        """
        初始化线性支持向量机模型。

        参数:
        - C (float): 正则化强度的倒数；必须为正浮点数。默认值为 1.0。
        """
        self.model = LinearSVC(C=C)

    def train(self, train_data, train_targets):
        """
        训练线性支持向量机模型。

        参数:
        - train_data (array-like): 训练数据。
        - train_targets (array-like): 训练数据的目标值。
        """
        self.model.fit(train_data, train_targets)

    def evaluate(self, data, targets):
        """
        评估线性支持向量机模型的性能。

        参数:
        - data (array-like): 待评估的数据。
        - targets (array-like): 数据对应的真实目标值。

        返回值:
        - float: 模型在给定数据上的准确率得分。
        """
        return self.model.score(data, targets)


def data_preprocess(args):
    # 如果提供了——ent标志，则使用特征工程函数feature_extraction()从一个文件中加载数据。
    # 否则，它将从先前存在的文件加载。
    if args.ent:
        diagrams = feature_extraction()[0]
    else:
        diagrams = np.load('./data/diagrams.npy')
    # 读取包含蛋白质序列信息的CAST文件和包含图表的Numpy数组。
    cast = pd.read_table('./data/SCOP40mini_sequence_minidatabase_19.cast')
    cast.columns.values[0] = 'protein'

    data_list = []
    target_list = []
    for task in range(1, 56):  # Assuming only one task for now
        task_col = cast.iloc[:, task]
        train_data = []
        train_targets = []
        test_data = []
        test_targets = []

        for index in range(len(task_col)):
            element = task_col[index]
            if element == 1:
                # 正例，属于训练集
                train_data.append(diagrams[index])  # 添加对应的 diagrams 元素
                train_targets.append(1)
            elif element == 2:
                # 负例，属于训练集
                train_data.append(diagrams[index])  # 添加对应的 diagrams 元素
                train_targets.append(0)
            elif element == 3:
                # 正例，属于测试集
                test_data.append(diagrams[index])  # 添加对应的 diagrams 元素
                test_targets.append(1)
            elif element == 4:
                # 负例，属于测试集
                test_data.append(diagrams[index])  # 添加对应的 diagrams 元素
                test_targets.append(0)
            else:
                # 其他情况，可以根据具体需求进行处理
                pass

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

    for i in range(len(data_list)):
        train_data, test_data = data_list[i]
        train_targets, test_targets = target_list[i]

        print(f"Processing dataset {i + 1}/{len(data_list)}")

        # Train the model
        model.train(train_data, train_targets)

        # Evaluate the model
        train_accuracy = model.evaluate(train_data, train_targets)
        test_accuracy = model.evaluate(test_data, test_targets)

        print(f"Dataset {i + 1}/{len(data_list)} - Train Accuracy: {train_accuracy}, Test Accuracy: {test_accuracy}")

        task_acc_train.append(train_accuracy)
        task_acc_test.append(test_accuracy)

    print("Training accuracy:", sum(task_acc_train) / len(task_acc_train))
    print("Testing accuracy:", sum(task_acc_test) / len(task_acc_test))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SVM Model Training and Evaluation")
    parser.add_argument('--model_type', type=str, default='svm', choices=['svm', 'linear_svm', 'lr'], help="Model type")
    parser.add_argument('--kernel', type=str, default='linear', choices=['linear', 'poly', 'rbf', 'sigmoid'],
                        help="Kernel type")
    parser.add_argument('--C', type=float, default=20, help="Regularization parameter")
    parser.add_argument('--ent', action='store_true',
                        help="Load data from a file using a feature engineering function feature_extraction() from "
                             "fea.py")
    args = parser.parse_args()
    main(args)
