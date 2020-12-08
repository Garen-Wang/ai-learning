import numpy as np
import matplotlib.pyplot as plt

from step5.HelperClass2.NeuralNet import *
from step5.HelperClass2.DataReader_2_0 import *

train_file_name = '../ai-data/Data/ch10.train.npz'
test_file_name = '../ai-data/Data/ch10.test.npz'


if __name__ == '__main__':
    reader = DataReader_2_0(train_file_name, test_file_name)
    reader.ReadData()
    reader.NormalizeX()
    # reader.NormalizeY(NetType.BinaryClassifier, 1)
    reader.GetValidationSet()
    n_input, n_hidden, n_output = reader.num_feature, 2, 1
    eta, batch_size, max_epoch = 0.1, 5, 10000
    eps = 0.08

    params = HyperParameters(n_input, n_hidden, n_output, eta, max_epoch, batch_size, eps, NetType.BinaryClassifier, InitialMethod.Xavier)
    net = NeuralNet(params, "Arc_221")
    net.train(reader, 5, True)
    net.ShowTrainingHistory()
