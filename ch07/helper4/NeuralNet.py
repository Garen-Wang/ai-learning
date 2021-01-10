import math

from ch07.helper4.DataReader import *
from ch07.helper4.TrainingHistory import *
from ch07.helper4.ClassifierFunction import *


class NeuralNet(object):
    def __init__(self, params):
        self.params = params
        self.W = npy.zeros((params.input_size, params.output_size))
        self.B = npy.zeros((1, params.output_size))

    def forwardBatch(self, xBatch):
        zBatch = np.dot(xBatch, self.W) + self.B
        if self.params.net_type == NetType.BinaryClassifier:
            return Logistic().forward(zBatch)
        elif self.params.net_type == NetType.MultipleClassifier:
            return Softmax().forward(zBatch)
        else:
            return zBatch

    def backwardBatch(self, xBatch, yBatch, zBatch):
        dZ = zBatch - yBatch
        m = xBatch.shape[0]
        dW = npy.dot(xBatch.T, dZ) / m
        dB = dZ.sum(axis=0, keepdims=True) / m
        return dW, dB

    def update(self, dW, dB):
        self.W -= self.params.eta * dW
        self.B -= self.params.eta * dB

    def checkLoss(self, reader):
        X, Y = reader.getWholeTrainSamples()
        m = X.shape[0]
        Z = self.forwardBatch(X)
        LOSS = (Z - Y) ** 2
        loss = LOSS.sum() / 2 / m
        return loss

    def train(self, reader, checkpoint = 0.1):
        loss_history = TrainingHistory()
        loss = 10
        if self.params.batch_size == -1:
            self.params.batch_size = reader.num_train
        max_iteration = math.ceil(reader.num_train / self.params.batch_size)
        checkpoint_iteration = int(max_iteration * checkpoint)

        for epoch in range(self.params.max_epoch):
            print('epoch = {}'.format(epoch))
            reader.shuffle()
            for iteration in range(max_iteration):
                xBatch, yBatch = reader.getBatchTrainSamples(self.params.batch_size, iteration)
                zBatch = self.forwardBatch(xBatch)
                dW, dB = self.backwardBatch(xBatch, yBatch, zBatch)
                self.update(dW, dB)
                print("W = ", self.W)
                print("B = ", self.B)

                total_iteration = epoch * max_iteration + iteration
                if (total_iteration + 1) % checkpoint_iteration == 0:
                    loss = self.checkLoss(reader)
                    loss_history.addLossHistory(epoch * max_iteration + iteration, loss, self.W, self.B)
                    if loss < self.params.eps:
                        break
            if loss < self.params.eps:
                break
        loss_history.showLossHistory(self.params)
        print('W = ', self.W)
        print('B = ', self.B)


