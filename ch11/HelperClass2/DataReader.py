import numpy as np
from pathlib import Path

from ch11.HelperClass2.EnumDef import *


class DataReader(object):
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
        self.XTrainRaw = None
        self.YTrainRaw = None
        self.XTest = None
        self.YTest = None
        self.XTestRaw = None
        self.YTestRaw = None
        self.XVld = None
        self.YVld = None

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
            raise Exception('GG')

        test_file = Path(self.test_file_name)
        if test_file.exists():
            data = np.load(self.test_file_name)
            self.XTestRaw = data['data']
            self.YTestRaw = data['label']
            self.num_test = self.XTestRaw.shape[0]

            self.XTest = self.XTestRaw
            self.YTest = self.YTestRaw

            self.XVld = self.XTest
            self.YVld = self.YTest
        else:
            raise Exception('GG')

    def __NormalizeX(self, XRaw):
        XNew = np.zeros_like(XRaw)
        XNorm = np.zeros((2, self.num_feature))
        for i in range(self.num_feature):
            X = XRaw[:, i]
            max_value = np.max(X)
            min_value = np.min(X)
            XNorm[0, i] = min_value
            XNorm[1, i] = max_value - min_value
            XX = (X - XNorm[0, i]) / XNorm[1, i]
            XNew[:, i] = XX
        return XNew

    def NormalizeX(self):
        XMerge = np.vstack((self.XTrainRaw, self.XTestRaw))
        XMergeNorm = self.__NormalizeX(XMerge)
        self.XTrain = XMergeNorm[:self.num_train, :]
        self.XTest = XMergeNorm[self.num_train:, :]

    def __NormalizeY(self, Y):
        YNorm = np.zeros((2, 1))
        max_value = np.max(Y)
        min_value = np.min(Y)
        YNorm[0, 0] = min_value
        YNorm[1, 0] = max_value - min_value
        YNew = (Y - YNorm[0, 0]) / YNorm[1, 0]
        return YNew

    def __ToZeroOne(self, Y, negative_label=0, positive_label=1, negative_value=0, positive_value=1):
        YNew = np.zeros_like(Y)
        for i in range(Y.shape[0]):
            if Y[i, 0] == negative_label:
                YNew[i, 0] = negative_value
            elif Y[i, 0] == positive_label:
                YNew[i, 0] = positive_value
        return YNew

    def __ToOneHot(self, Y, base=0):
        YNew = np.zeros((Y.shape[0], self.num_category))
        for i in range(Y.shape[0]):
            n = int(Y[i, 0])
            YNew[i, n - base] = 1
        return YNew

    def NormalizeY(self, net_type, base=0):
        if net_type == NetType.Fitting:
            YMerge = np.vstack((self.YTrainRaw, self.YTestRaw))
            YMergeNorm = self.__NormalizeY(YMerge)
            self.YTrain = YMergeNorm[:self.num_train, :]
            self.YTest = YMergeNorm[self.num_train:, :]
        elif net_type == NetType.BinaryClassifier:
            self.YTrain = self.__ToZeroOne(self.YTrainRaw, base)
            self.YTest = self.__ToZeroOne(self.YTestRaw, base)
        elif net_type == NetType.MultipleClassifier:
            self.YTrain = self.__ToOneHot(self.YTrainRaw, base)
            self.YTest = self.__ToOneHot(self.YTestRaw, base)

    def GenerateValidationSet(self, k=10):
        self.num_validation = int(self.num_train / k)
        self.num_train -= self.num_validation
        self.XVld = self.XTrain[:self.num_validation]
        self.YVld = self.YTrain[:self.num_validation]
        self.XTrain = self.XTrain[self.num_validation:]
        self.YTrain = self.YTrain[self.num_validation:]

    def GetBatchTrainSamples(self, batch_size, iteration):
        start = batch_size * iteration
        end = start + batch_size
        batch_x = self.XTrain[start:end, :]
        batch_y = self.YTrain[start:end, :]
        return batch_x, batch_y

    def GetTestSet(self):
        return self.XTest, self.YTest

    def GetValidationSet(self):
        return self.XVld, self.YVld

    def Shuffle(self):
        seed = np.random.randint(0, 100)
        np.random.seed(seed)
        XP = np.random.permutation(self.XTrain)
        np.random.seed(seed)
        YP = np.random.permutation(self.YTrain)
        self.XTrain = XP
        self.YTrain = YP
