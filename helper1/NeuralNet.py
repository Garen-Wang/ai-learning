import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.colors import LogNorm

from helper1.HyperParameters import *
from helper1.TrainingHistory import *
from helper1.DataReader import *

class NeuralNet(object):
    def __init__(self, params):
        self.params = params
        self.w = 0
        self.b = 0
    def forwardBatch(self, xBatch):
        zBatch = np.dot(xBatch, self.w) + self.b
        return zBatch
    def backwardBatch(self, xBatch, yBatch, zBatch):
        m = xBatch.shape[0]
        dZ = zBatch - yBatch
        dB = dZ.sum(axis = 0, keepdims = True) / m
        dW = np.dot(xBatch.T, dZ) / m
        return dW, dB
    def update(self, dW, dB):
        self.w -= self.params.eta * dW
        self.b -= self.params.eta * dB
    def checkLoss(self, reader):
        X, Y = reader.getWholeTrainSamples()
        m = X.shape[0]
        Z = self.forwardBatch(X)
        LOSS = (Z - Y) ** 2
        loss = LOSS.sum() / 2 / m
        return loss
    def loss_contour(self, reader, loss_history, batch_size, iteration):
        last_loss, result_w, result_b = loss_history.getLast()
        len1 = 50
        len2 = 50
        w = np.linspace(result_w - 1, result_w + 1, len1)
        b = np.linspace(result_b - 1, result_b + 1, len2)
        W, B = np.meshgrid(w, b)
        len = len1 * len2
        X, Y = reader.getWholeTrainSamples()
        m = X.shape[0]
        Z = np.dot(X, W.ravel().reshape(1, len)) + B.ravel().reshape(1, len)
        Loss1 = (Z - Y) ** 2
        Loss2 = Loss1.sum(axis = 0, keepdims = True) / m
        Loss3 = Loss2.reshape(len1, len2)
        plt.contour(W, B, Loss3, levels = np.logspace(-5, 5, 100), norm = LogNorm(), cmap = plt.cm.jet)
        plt.plot(loss_history.w_history, loss_history.b_history)
        plt.axis([result_w - 1, result_w + 1, result_b - 1, result_b + 1])
        plt.show()
        pass
    def train(self, reader):
        loss_history = TrainingHistory()

        if self.params.batch_size == -1:
            self.params.batch_size = reader.num_train
        max_iteration = reader.num_train // self.params.batch_size
        for epoch in range(self.params.max_epoch):
            print('epoch = %d' % epoch)
            reader.shuffle()
            for iteration in range(max_iteration):
                xBatch, yBatch = reader.getBatchTrainSamples(self.params.batch_size, iteration)
                zBatch = self.forwardBatch(xBatch)
                dW, dB = self.backwardBatch(xBatch, yBatch, zBatch)
                self.update(dW, dB)
                if iteration % 2 == 0:
                    loss = self.checkLoss(reader)
                    print(epoch, iteration, loss)
                    loss_history.addHistory(epoch * max_iteration + iteration, loss, self.w[0, 0], self.b[0, 0])
                    if loss < self.params.eps:
                        break

            if loss < self.params.eps:
                break

        loss_history.showHistory(self.params)
        print(self.w, self.b)
        self.loss_contour(reader, loss_history, self.params.batch_size, epoch * max_iteration + iteration)

