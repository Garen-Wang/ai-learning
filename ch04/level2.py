import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from helper1.DataReader import *

file_name = 'ai-data/Data/ch04.npz'
if __name__ == '__main__':
    reader = DataReader(file_name)
    reader.readData()
    X, Y = reader.getWholeTrainSamples()

    eta = 0.1
    w, b = 0, 0
    for i in range(reader.num_train):
        x_i = X[i]
        y_i = Y[i]
        z_i = x_i * w + b
        dz = z_i - y_i
        dw = dz * x_i
        db = dz
        w = w - eta * dw
        b = b - eta * db

        print('w = {}'.format(w))
        print('b = {}'.format(b))