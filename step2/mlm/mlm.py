import numpy as np
import matplotlib.pyplot as plt

from helper4.NeuralNet import *

file_name = './Assignment1/Dataset/mlm.csv'

if __name__ == '__main__':
    reader = DataReader(file_name)
    reader.readData()
    reader.normalizeX()
    params = HyperParameters(2, 1, eta=0.1, max_epoch=100, batch_size=5, eps=1e-4)
    neural = NeuralNet(params)
    neural.train(reader, checkpoint=0.1)

