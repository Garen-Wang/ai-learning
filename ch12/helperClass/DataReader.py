from ch12.helperClass.EnumDef import NetType, InitMethod
import numpy as np
from pathlib import Path

class DataReader(object):
    def __init__(self, train_file_name, test_file_name):
        self.train_file_name = train_file_name
        self.test_file_name = test_file_name
        self.num_train = 0
        self.num_valid = 0
        self.num_test = 0
        self.num_feature = 0
        self.num_category = 0
        self.XTrain = None
        self.XValid = None
        self.XTest = None
        self.YTrain = None
        self.YValid = None
        self.YTest = None
        self.XTrainRaw = None
        self.XTestRaw = None
        self.YTrainRaw = None
        self.YTestRaw = None

    def read_data(self):
        # reading train data
        train_file = Path(self.train_file_name)
        if train_file.exists():
            data = np.load(self.train_file_name)
            self.XTrainRaw = data['data']
            self.YTrainRaw = data['label']
            self.num_train = self.XTrainRaw.shape[0]
            self.num_feature = self.XTrainRaw.shape[1]
            self.num_category = len(np.unique(self.YTrainRaw))
            # default
            self.XTrain = self.XTrainRaw
            self.YTrain = self.YTrainRaw
        else:
            raise Exception('file not found')

        # reading test data
        test_file = Path(self.test_file_name)
        if test_file.exists():
            data = np.load(self.test_file_name)
            self.XTestRaw = data['data']
            self.YTestRaw = data['label']
            self.num_test = self.XTestRaw.shape[0]
            # default
            self.XTest = self.XTestRaw
            self.YTest = self.YTestRaw
            self.XValid = self.XTest
            self.YValid = self.YTest
        else:
            raise Exception('file not found')

    def __normalize_x(self, XRaw):
        XNorm = np.zeros(XRaw.shape)
        for i in range(self.num_feature):
            x = XRaw[:, i]
            max_value = np.max(x)
            min_value = np.min(x)
            x_new = (x - min_value) / (max_value - min_value)
            XNorm[:, i] = x_new
        return XNorm


    def normalize_x(self):
        XMerge = np.vstack((self.XTrainRaw, self.XTestRaw))
        XMergeNorm = self.__normalize_x(XMerge)
        self.XTrain = XMergeNorm[:self.num_train, :]
        self.XTest = XMergeNorm[self.num_train:, :]

    def __normalize_y(self, YRaw):
        max_value = np.max(YRaw)
        min_value = np.min(YRaw)
        return (YRaw - min_value) / (max_value - min_value)

    def __one_hot(self, Y, base=0):
        YNorm = np.zeros((Y.shape[0], self.num_category))
        for i in range(Y.shape[0]):
            n = int(Y[i, 0])
            YNorm[i, n-base] = 1
        return YNorm

    def __zero_one(self, Y, positive_label=1, negative_label=0, positive_value=1, negative_value=0):
        YNorm = np.zeros(Y.shape)
        for i in range(Y.shape[0]):
            if Y[i, 0] == negative_label:
                YNorm[i, 0] = negative_value
            elif Y[i, 0] == positive_label:
                YNorm[i, 0] = positive_value
        return YNorm

    def normalize_y(self, net_type, base=0):
        if net_type == NetType.Fitting:
            YRaw = np.vstack((self.YTrainRaw, self.YTestRaw))
            YNorm = self.__normalize_y(YRaw)
            self.YTrain = YNorm[:self.num_train, :]
            self.YTest = YNorm[self.num_train:, :]
        elif net_type == NetType.BinaryClassifier:
            self.YTrain = self.__zero_one(self.YTrainRaw, base)
            self.YTest = self.__zero_one(self.YTestRaw, base)
        elif net_type == NetType.MultipleClassifier:
            self.YTrain = self.__one_hot(self.YTrainRaw, base)
            self.YTest = self.__one_hot(self.YTestRaw, base)

    def generate_validation_set(self, k):
        self.num_valid = int(self.num_train / k)
        self.num_train -= self.num_valid
        self.XValid = self.XTrain[:self.num_valid, :]
        self.YValid = self.YTrain[:self.num_valid, :]
        self.XTrain = self.XTrain[self.num_valid:, :]
        self.YTrain = self.YTrain[self.num_valid:, :]

    def get_test_set(self):
        return self.XTest, self.YTest

    def get_validation_set(self):
        return self.XValid, self.YValid

    def get_batch_train_samples(self, batch_size, iteration):
        start = batch_size * iteration
        end = start + batch_size
        return self.XTrain[start:end, :], self.YTrain[start:end, :]

    def shuffle(self):
        seed = np.random.randint(0, 100)
        np.random.seed(seed)
        XP = np.random.permutation(self.XTrain)
        np.random.seed(seed)
        YP = np.random.permutation(self.YTrain)
        self.XTrain = XP
        self.YTrain = YP
        return self.XTrain, self.YTrain
