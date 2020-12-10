import numpy as np
import matplotlib.pyplot as plt

from step6.HelperClass2.NeuralNet import *
from step6.HelperClass2.DataReader import *
from step6.HelperClass2.HyperParameters import *
from step6.HelperClass2.Visualizer import *

train_file_name = '../ai-data/Data/ch11.train.npz'
test_file_name = '../ai-data/Data/ch11.test.npz'

if __name__ == '__main__':
    reader = DataReader(train_file_name, test_file_name)
    reader.ReadData()
    reader.NormalizeY(NetType.MultipleClassifier, base=1)

    plt.figure(figsize=(6, 6))
    # DrawThreeCategoryPoints(reader.XTrainRaw[:, 0], reader.XTrainRaw[:, 1], reader.YTrain, show=True, title="Source Data")

    reader.NormalizeX()
    reader.GenerateValidationSet()

    num_input = reader.num_feature
    num_hidden = 3
    num_output = reader.num_category
    eta = 0.1
    batch_size = 10
    max_epoch = 5000
    eps = 1e-1

    params = HyperParameters(num_input, num_hidden, num_output, eta, max_epoch, batch_size, eps, NetType.MultipleClassifier, InitialMethod.Xavier)
    net = NeuralNet(params, "Bank_233")

    # net.LoadResult()

    net.train(reader, 100, True)
    net.ShowTrainingHistory()

    DrawThreeCategoryPoints(reader.XTrain[:, 0], reader.XTrain[:, 1], reader.YTrain, show=True, title=params.toString())
