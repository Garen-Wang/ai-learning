import numpy as np
from pathlib import Path

class DataReader(object):
    def __init__(self, data_file):
        self.train_file_name = data_file
        self.num_train = 0
        self.Xtrain = None
        self.Ytrain = None
    
    def readData(self):
        train_file = Path(self.train_file_name)
        if train_file.exists():
            data = np.load(self.train_file_name)
            self.Xtrain = data['data']
            self.Ytrain = data['label']
            self.num_train = self.Xtrain.shape[0]
        else:
            raise Exception('cannot open train file!')

    def getSingleTrainSample(self, iteration):
        x = self.Xtrain[iteration]
        y = self.Ytrain[iteration]
        return x, y
    
    def getBatchTrainSamples(self, batch_size, iteration):
        start = iteration * batch_size
        end = start + batch_size
        batch_X = self.Xtrain[start:end,:]
        batch_Y = self.Ytrain[start:end,:]
        return batch_X, batch_Y
    
    def getWholeTrainSamples(self):
        return self.Xtrain, self.Ytrain
    
    def shuffle(self):
        seed = np.random.randint(0, 100)
        np.random.seed(seed)
        XP = np.random.permutation(self.Xtrain)
        np.random.seed(seed)
        YP = np.random.permutation(self.Ytrain)
        self.Xtrain = XP
        self.Ytrain = YP
    
