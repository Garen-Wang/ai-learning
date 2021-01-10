import numpy as np
import matplotlib.pyplot as plt

from ch11.HelperClass2.NeuralNet import *
from ch11.HelperClass2.Visualizer import *
from ch11.HelperClass2.DataReader import *

train_file_name = '../ai-data/Data/ch11.train.npz'
test_file_name = '../ai-data/Data/ch11.test.npz'



if __name__ == '__main__':
    reader = DataReader(train_file_name, test_file_name)
    reader.ReadData()
    reader.NormalizeY(NetType.MultipleClassifier, base=1)
    reader.NormalizeX()
    reader.GenerateValidationSet()

    num_input = reader.num_feature
    num_hidden = 64  # change here
    num_output = reader.num_category
    eta = 0.1
    batch_size = 10
    max_epoch = 10000
    eps = 1e-3

    params = HyperParameters(num_input, num_hidden, num_output, eta, max_epoch, batch_size, eps, NetType.MultipleClassifier, InitialMethod.Xavier)
    net = NeuralNet(params, "Bank_2N3")

    net.train(reader, 100, True)
    net.ShowTrainingHistory()
    loss = net.GetLatestAverageLoss()

    fig = plt.figure(figsize=(6, 6))
    DrawThreeCategoryPoints(reader.XTrain[:, 0], reader.XTrain[:, 1], reader.YTrain, title=params.toString())
    ShowClassificationResult25D(net, 50, "{}, loss={}".format(params.toString(), loss))
    plt.show()

