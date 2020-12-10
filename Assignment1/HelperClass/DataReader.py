import pandas as pd
import numpy as np


class DataReader(object):
    def __init__(self, file_name):
        self.xRaw = None
        self.yRaw = None
        self.xNorm = None
        self.yNorm = None
        self.xTrain = None
        self.yTrain = None
        self.num_train = None
        self.file_name = file_name

    def readData(self):
        try:
            data = pd.read_csv(self.file_name)
            self.num_train = len(list(data['x']))
            x1 = np.array(list(data['x'])).reshape(self.num_train, 1)
            x2 = np.array(list(data['y'])).reshape(self.num_train, 1)
            self.xRaw = np.hstack((x1, x2))
            self.yRaw = np.array(list(data['z'])).reshape(self.num_train, 1)
            self.xTrain = self.xRaw
            self.yTrain = self.xRaw
        except:
            print("FILE NOT FOUND!")

    def normalizeX(self):
        self.xNorm = np.zeros((self.num_train, self.xRaw.shape[1]))
        for i in range(self.xRaw.shape[1]):
            col = self.xRaw[:, i]
            min_value = np.min(col)
            max_value = np.max(col)
            self.xNorm[i, 0] = min_value
            self.xNorm[i, 1] = max_value - min_value
            new_col = (col - self.xNorm[i, 0]) / self.xNorm[i, 1]
            self.xTrain[:, i] = new_col

    def normalizeY(self):
        self.yNorm = np.zeros((1, 2))
        max_value = np.max(self.yRaw)
        min_value = np.min(self.yRaw)
        self.yNorm[0, 0] = min_value
        self.yNorm[0, 1] = max_value - min_value
        self.yTrain = (self.yRaw - self.yNorm[0, 0]) / self.yNorm[0, 1]
        print("Y shape is ", self.yTrain.shape)

    def getSingleTrainSample(self, iteration):
        return self.xTrain[iteration], self.yTrain[iteration]

    def getBatchTrainSamples(self, batch_size, iteration):
        start = batch_size * iteration
        end = start + batch_size
        return self.xTrain[start:end, :], self.yTrain[start:end, :]

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
