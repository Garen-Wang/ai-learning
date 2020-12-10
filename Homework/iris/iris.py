import numpy as np
import pandas as pd

from Homework.iris.HelperClass.NeuralNet import *
from Homework.iris.HelperClass.DataReader import *

train_file_name = 'iris.csv'

# def display(net, reader):
    # XTrain = reader.__NormalizeX(reader.XTrainRaw)
    # print(XTrain)
    # max_iteration = int(XTrain.shape[0] / net.params.batch_size)
    # for iteration in range(max_iteration):
    #     start = net.params.batch_size * iteration
    #     end = start + net.params.batch_size
    #     X = XTrain[start:end, :]
    #     Z = net.inference(X)
    #     Z = np.max(Z, axis=1, keepdims=True)
    #     print(Z)

if __name__ == '__main__':
    reader = DataReader(train_file_name)
    reader.ReadData()
    # print(reader.XTrainRaw)
    # print(reader.YTrainRaw)
    reader.NormalizeY(NetType.MultipleClassifier, base=1)
    reader.NormalizeX()
    reader.GenerateValidationSet()
    # print(reader.XTrain)
    # print(reader.YTrain)

    num_input = reader.num_feature
    num_hidden = 4
    num_output = reader.num_category
    eta = 0.1
    max_epoch = 10000
    batch_size = 5
    eps = 1e-2
    params = HyperParameters(num_input, num_hidden, num_output,
                             eta, max_epoch, batch_size, eps,
                             NetType.MultipleClassifier, InitialMethod.Xavier)

    net = NeuralNet(params, "iris_443")
    net.train(reader, 10)
    net.ShowTrainingHistory()
    # print("wb1.W = ", net.wb1.W)
    # print("wb1.B = ", net.wb1.B)
    # print("wb2.W = ", net.wb2.W)
    # print("wb2.B = ", net.wb2.B)

