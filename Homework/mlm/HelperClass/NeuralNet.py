import math
from Homework.mlm.HelperClass.TrainingHistory import *
from Homework.mlm.HelperClass.DataReader import *


class NeuralNet(object):
    def __init__(self, params):
        self.params = params
        self.W = np.zeros((params.input_size, params.output_size))
        self.B = np.zeros((1, params.output_size))

    def forward(self, X):
        Z = np.dot(X, self.W) + self.B
        return Z

    def backward(self, X, Y, Z):
        m = X.shape[0]
        dZ = Z - Y
        dB = dZ.sum(axis=0, keepdims=True) / m
        dW = np.dot(X.T, dZ) / m
        return dW, dB

    def update(self, dW, dB):
        self.W -= self.params.eta * dW
        self.B -= self.params.eta * dB

    def checkLoss(self, reader):
        X, Y = reader.getWholeTrainSamples()
        m = X.shape[0]
        Z = self.forward(X)
        LOSS = (Z - Y) ** 2
        loss = LOSS.sum() / m / 2
        return loss

    def train(self, reader, checkpoint):
        history = TrainingHistory()
        max_iteration = int(math.ceil(reader.num_train / self.params.batch_size))
        checkpoint_iteration = int(max_iteration * checkpoint)
        for epoch in range(self.params.max_epoch):
            print("epoch = {}".format(epoch))
            reader.shuffle()
            for iteration in range(max_iteration):
                X, Y = reader.getBatchTrainSamples(self.params.batch_size, iteration)
                Z = self.forward(X)
                dW, dB = self.backward(X, Y, Z)
                self.update(dW, dB)

                total_iteration = epoch * max_iteration + iteration
                if (total_iteration + 1) % checkpoint_iteration == 0:
                    loss = self.checkLoss(reader)
                    print(epoch, iteration, loss, self.W, self.B)
                    history.append(loss, total_iteration)
                    if loss < self.params.eps:
                        break
            if loss < self.params.eps:
                break
        history.show(self.params)
        print("W = ", self.W)
        print("B = ", self.B)

