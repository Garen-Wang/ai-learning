import numpy as np
from pathlib import Path

from ch10.HelperClass2.EnumDef import *


class DataReader_2_0(object):
    def __init__(self, train_file_name, test_file_name):
        self.train_file_name = train_file_name
        self.test_file_name = test_file_name
        self.num_train = 0
        self.num_test = 0
        self.num_validation = 0
        self.num_feature = 0
        self.num_category = 0
        self.XTrain = None
        self.YTrain = None
        self.XTest = None
        self.YTest = None
        self.XTrainRaw = None
        self.YTrainRaw = None
        self.XTestRaw = None
        self.YTestRaw = None
        self.XDev = None  # validation feature set
        self.YDev = None  # validation label set

    def ReadData(self):
        train_file = Path(self.train_file_name)
        if train_file.exists():
            data = np.load(self.train_file_name)
            self.XTrainRaw = data['data']
            self.YTrainRaw = data['label']
            self.num_train = self.XTrainRaw.shape[0]
            self.num_feature = self.XTrainRaw.shape[1]
            self.num_category = len(np.unique(self.YTrainRaw))
            self.XTrain = self.XTrainRaw
            self.YTrain = self.YTrainRaw
        else:
            raise Exception("Cannot find file!")

        test_file = Path(self.test_file_name)
        if test_file.exists():
            data = np.load(self.test_file_name)
            self.XTestRaw, self.YTestRaw = data['data'], data['label']
            self.num_test = self.XTestRaw.shape[0]
            self.XTest, self.YTest = self.XTestRaw, self.YTestRaw
            self.XDev, self.YDev = self.XTest, self.YTest
        else:
            raise Exception("Cannot find file!")

    def __NormalizeX(self, raw):
        X = np.zeros_like(raw)
        self.XNorm = np.zeros((2, self.num_feature))
        for i in range(self.num_feature):
            x = raw[:, i]
            max_value = np.max(x)
            min_value = np.min(x)
            self.XNorm[0, i] = min_value
            self.XNorm[1, i] = max_value - min_value
            x_new = (x - self.XNorm[0, i]) / self.XNorm[1, i]
            X[:, i] = x_new
        return X

    def NormalizeX(self):
        XMerge = np.vstack((self.XTrainRaw, self.XTestRaw))
        XMergeNorm = self.__NormalizeX(XMerge)
        self.XTrain = XMergeNorm[0:self.num_train, :]
        self.XTest = XMergeNorm[self.num_train:, :]

    def __normalizeY(self, raw):
        self.YNorm = np.zeros((2, 1))
        max_value = np.max(raw)
        min_value = np.min(raw)
        self.YNorm[0, 0] = min_value
        self.YNorm[1, 0] = max_value - min_value
        YNew = (raw - self.YNorm[0, 0]) / self.YNorm[1, 0]
        return YNew

    def __ToOneHot(self, Y, base=0):
        YNew = np.zeros((self.num_train, self.num_category))
        for i in range(Y.shape[0]):
            n = int(Y[i, 0])
            YNew[i, n - base] = 1
        return YNew

    def __ToZeroOne(self, Y, positive_label=1, negative_label=0, positive_value=1, negative_value=0):
        YNew = np.zeros_like(Y)
        for i in range(Y.shape[0]):
            if Y[i, 0] == negative_label:
                YNew[i, 0] = negative_value
            if Y[i, 0] == positive_label:
                YNew[i, 0] = positive_value

        return YNew

    def NormalizeY(self, net_type, base=0):
        if net_type == NetType.Fitting:
            YMerge = np.vstack((self.YTrainRaw, self.YTestRaw))
            YMergeNorm = self.__normalizeY(YMerge)
            self.YTrain = YMergeNorm[0:self.num_train, :]
            self.YTest = YMergeNorm[self.num_train:, :]
        elif net_type == NetType.BinaryClassifier:
            self.YTrain = self.__ToZeroOne(self.YTrainRaw, base)
            self.YTest = self.__ToZeroOne(self.YTestRaw, base)
        elif net_type == NetType.MultipleClassifier:
            self.YTrain = self.__ToOneHot(self.YTrainRaw, base)
            self.YTest = self.__ToOneHot(self.YTestRaw, base)

    def DenormalizeY(self, predicate_data):
        real_value = predicate_data * self.YNorm[0, 0] + self.YNorm[1, 0]
        return real_value

    def NormalizePredicateData(self, XPredicate):
        XNew = np.zeros(XPredicate.shape)
        for i in range(self.num_feature):
            x = XPredicate[i, :]
            XNew[i, :] = (x - self.XNorm[0, i]) / self.XNorm[1, i]
        return XNew

    def GenerateValidationSet(self, k=10):
        self.num_validation = int(self.num_train / k)
        self.num_train -= self.num_validation
        self.XDev, self.YDev = self.XTrain[0:self.num_validation], self.YTrain[0:self.num_validation]
        self.XTrain, self.YTrain = self.XTrain[self.num_validation:], self.YTrain[self.num_validation:]

    def GetBatchTrainSamples(self, batch_size, iteration):
        start = batch_size * iteration
        end = start + batch_size
        batchX = self.XTrain[start:end, :]
        batchY = self.YTrain[start:end, :]
        return batchX, batchY

    def GetTestSet(self):
        return self.XTest, self.YTest

    def GetValidationSet(self):
        return self.XDev, self.YDev

    def Shuffle(self):
        seed = np.random.randint(0, 100)
        np.random.seed(seed)
        XP = np.random.permutation(self.XTrain)
        np.random.seed(seed)
        YP = np.random.permutation(self.YTrain)
        self.XTrain = XP
        self.YTrain = YP
