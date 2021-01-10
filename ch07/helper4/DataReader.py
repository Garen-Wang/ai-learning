import numpy as np
import pandas as pd
from pathlib import Path


class DataReader(object):
    def __init__(self, data_file):
        self.train_file_name = data_file
        self.num_train = 0
        self.xTrain = None
        self.yTrain = None
        self.xRaw = None
        self.yRaw = None
        self.xNorm = None
        self.yNorm = None

    def readData(self):
        print(self.train_file_name)
        train_file = Path(self.train_file_name)
        if train_file.exists():
            data = np.load(self.train_file_name)
            self.xRaw = data['data']
            self.yRaw = data['label']
            self.num_train = self.xRaw.shape[0]
            self.xTrain = self.xRaw
            self.yTrain = self.yRaw
        else:
            raise Exception('Cannot find train file!')


    def getSingleTrainSample(self, iteration):
        return self.xTrain[iteration], self.yTrain[iteration]

    def getBatchTrainSamples(self, batch_size, iteration):
        start = batch_size * iteration
        end = start + batch_size
        xBatch = self.xTrain[start:end, :]
        yBatch = self.yTrain[start:end, :]
        return xBatch, yBatch

    def getWholeTrainSamples(self):
        return self.xTrain, self.yTrain

    def shuffle(self):
        seed = np.random.randint(0, 100)
        np.random.seed(seed)
        XP = np.random.permutation(self.xTrain)
        np.random.seed(seed)
        YP = np.random.permutation(self.yTrain)
        self.xTrain = XP
        self.yTrain = YP

    # update here

    def normalizeX(self):
        print(self.xRaw)
        xNew = np.zeros(self.xRaw.shape)
        num_feature = self.xRaw.shape[1]
        self.xNorm = np.zeros((num_feature, 2))
        for i in range(num_feature):
            col_i = self.xRaw[:, i]
            maxv = np.max(col_i)
            minv = np.min(col_i)

            self.xNorm[i, 0] = minv
            self.xNorm[i, 1] = maxv - minv
            new_col = (col_i - self.xNorm[i, 0]) / (self.xNorm[i, 1])
            xNew[:, i] = new_col

        self.xTrain = xNew

    def normalizeY(self):
        self.yNorm = np.zeros((1, 2))
        maxv = np.max(self.yRaw)
        minv = np.min(self.yRaw)
        self.yNorm[0, 0] = minv
        self.yNorm[0, 1] = maxv - minv
        yNew = (self.yRaw - self.yNorm[0, 0]) / self.yNorm[0, 1]
        self.yTrain = yNew

    def normalizePredicateData(self, xRaw):
        xNew = np.zeros(xRaw.shape)
        n = xRaw.shape[1]
        for i in range(n):
            col_i = xRaw[:, i]
            xNew[:, i] = (col_i - self.xNorm[i, 0]) / self.xNorm[i, 1]
        return xNew

    def toOneHot(self, num_category, base=0):
        pass
        ans = np.zeros((self.yRaw.shape[0], num_category))
        for i in range(self.yRaw.shape[0]):
            n = int(self.yRaw[i, 0])
            ans[i, n-base] = 1
        self.yTrain = ans
