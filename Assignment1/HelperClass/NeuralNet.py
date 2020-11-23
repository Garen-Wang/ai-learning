import numpy as np
import math
from Assignment1.HelperClass.DataReader import *
from Assignment1.HelperClass.HyperParameters import *
from Assignment1.HelperClass.TrainingHistory import *

class NeuralNet(object):
    def __init__(self, params):
        self.params = params
        self.W = np.zeros((params.input_size, params.output_size))
        self.B = np.zeros((1, params.output_size))
        # print("W shape is ", self.W.shape)
        # print("B shape is ", self.B.shape)

    def __forward(self, X):
        Z = np.dot(X, self.W) + self.B
        # print("Z shape is ", Z.shape)
        return Z

    def __backward(self, X, Y, Z):
        # print("Y shape is ", Y.shape)
        m = X.shape[0]
        dZ = Z - Y
        # print("dZ shape is ", dZ.shape)
        dB = dZ.sum(axis=0, keepdims=True) / m
        dW = np.dot(X.T, dZ) / m
        # print("dB shape is ", dB.shape)
        # print("dW shape is ", dW.shape)
        return dW, dB

    def __update(self, dW, dB):
        self.W -= self.params.eta * dW
        self.B -= self.params.eta * dB

    def __checkLoss(self, reader):
        X, Y = reader.getWholeTrainSamples()
        m = X.shape[0]
        Z = self.__forward(X)
        LOSS = (Z - Y) ** 2
        loss = LOSS.sum() / m / 2
        return loss

    def inference(self, x):
        return self.__forward(x)

    def train(self, reader, checkpoint=0.1):
        history = TrainingHistory()
        if self.params.batch_size == -1:
            self.params.batch_size = self.params.num_train
        max_iteration = int(np.ceil(reader.num_train / self.params.batch_size))
        checkpoint_iteration = int(max_iteration * checkpoint)
        for epoch in range(self.params.max_epoch):
            print("epoch = {}".format(epoch))
            reader.shuffle()
            for iteration in range(max_iteration):
                X, Y = reader.getBatchTrainSamples(self.params.batch_size, iteration)
                Z = self.__forward(X)
                dW, dB = self.__backward(X, Y, Z)
                self.__update(dW, dB)

                total_iteration = epoch * max_iteration + iteration
                if (total_iteration + 1) % checkpoint_iteration == 0:
                    loss = self.__checkLoss(reader)
                    print(epoch, iteration, loss, self.W, self.B)
                    history.append(loss, total_iteration)
                    if loss < self.params.eps:
                        break
            if loss < self.params.eps:
                break
        history.show(self.params)
        print("W = ", self.W)
        print("B = ", self.B)

