import numpy as np
import matplotlib.pyplot as plt
from helper1.DataReader import *

file_name = '../ai-data/Data/ch04.npz'

class NeuralNet(object):
    def __init__(self, eta):
        self.eta = eta
        self.w = 0 
        self.b = 0
    
    def forward(self, x):
        z = x * self.w + self.b
        return z
    def backward(self, x, y, z):
        dz = z - y
        dw = dz * x
        db = dz
        return dw, db
    def update(self, dw, db):
        self.w = self.w - self.eta * dw
        self.b = self.b - self.eta * db
    
    def train(self, reader):
        for i in range(reader.num_train):
            x, y = reader.getSingleTrainSample(i)
            z = self.forward(x)
            dw, db = self.backward(x, y, z)
            self.update(dw, db)

def show_graph(neural, reader):
    X, Y = reader.getWholeTrainSamples()
    plt.plot(X, Y, 'b.')
    PX = np.linspace(0, 1)
    PZ = neural.forward(PX)
    plt.plot(PX, PZ, 'r')
    plt.show()

if __name__ == '__main__':
    reader = DataReader(file_name)
    reader.readData()

    eta = 0.1
    neural = NeuralNet(eta)
    neural.train(reader)

    print('w = %f, b = %f' %(neural.w, neural.b))
    result = neural.forward(0.346)
    print('result = %f' % result)

    show_graph(neural, reader)