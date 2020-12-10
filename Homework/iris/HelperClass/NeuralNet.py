import numpy as np
import os
import math

from Homework.iris.HelperClass.ActivatorFunction import *
from Homework.iris.HelperClass.ClassifierFunction import *
from Homework.iris.HelperClass.LossFunction import *
from Homework.iris.HelperClass.TrainingHistory import *
from Homework.iris.HelperClass.HyperParameters import *
from Homework.iris.HelperClass.WeightBias import *


class NeuralNet(object):
    def __init__(self, params, model_name):
        self.params = params
        self.model_name = model_name
        self.wb1 = WeightBias(params.num_input, params.num_hidden, params.init_method, params.eta)
        self.wb2 = WeightBias(params.num_hidden, params.num_output, params.init_method, params.eta)
        self.wb1.InitializeWeights()
        self.wb2.InitializeWeights()

    def forward(self, batch_x):
        self.Z1 = np.dot(batch_x, self.wb1.W) + self.wb1.B
        self.A1 = Sigmoid().forward(self.Z1)
        self.Z2 = np.dot(self.A1, self.wb2.W) + self.wb2.B
        if self.params.net_type == NetType.Fitting:
            self.A2 = self.Z2
        elif self.params.net_type == NetType.BinaryClassifier:
            self.A2 = Logistic().forward(self.Z2)
        elif self.params.net_type == NetType.MultipleClassifier:
            self.A2 = Softmax().forward(self.Z2)

        self.output = self.A2

    def backward(self, batch_x, batch_y, batch_a):
        m = batch_x.shape[0]
        dZ2 = batch_a - batch_y
        self.wb2.dW = np.dot(self.A1.T, dZ2) / m
        self.wb2.dB = np.sum(dZ2, axis=0, keepdims=True) / m
        dA1 = np.dot(dZ2, self.wb2.W.T)
        dZ1, _ = Sigmoid().backward(None, self.A1, dA1)
        self.wb1.dW = np.dot(batch_x.T, dZ1) / m
        self.wb1.dB = np.sum(dZ1, axis=0, keepdims=True) / m

    def update(self):
        self.wb1.update()
        self.wb2.update()

    def inference(self, x):
        self.forward(x)
        return self.output

    def train(self, reader, checkpoint):
        self.loss_history = TrainingHistory()
        self.loss_func = LossFunction(self.params.net_type)
        if self.params.batch_size == -1:
            self.params.batch_size = reader.num_train
        max_iteration = math.ceil(reader.num_train / self.params.batch_size)
        checkpoint_iteration = int(max_iteration * checkpoint)
        flag = False

        for epoch in range(self.params.max_epoch):
            reader.Shuffle()
            for iteration in range(max_iteration):
                batch_x, batch_y = reader.GetBatchTrainSamples(self.params.batch_size, iteration)
                batch_a = self.inference(batch_x)
                self.backward(batch_x, batch_y, batch_a)
                self.update()

                total_iteration = epoch * max_iteration + iteration
                if (total_iteration + 1) % checkpoint_iteration == 0:
                    flag = self.CheckErrorAndLoss(reader, batch_x, batch_y, epoch, total_iteration)
                    if flag:
                        break

            if flag:
                break

    def CheckErrorAndLoss(self, reader, train_x, train_y, epoch, total_iteration):
        print('epoch=%d, total_iteration=%d' % (epoch, total_iteration))
        train_z = self.inference(train_x)
        loss_train = self.loss_func.CheckLoss(train_z, train_y)
        accuracy_train = self.__CalAccuracy(train_z, train_y)
        print('loss_train=%.6f, accuracy_train=%.6f' % (loss_train, accuracy_train))

        vld_x, vld_y = reader.GetValidationSet()
        vld_z = self.inference(vld_x)
        loss_vld = self.loss_func.CheckLoss(vld_z, vld_y)
        accuracy_vld = self.__CalAccuracy(vld_z, vld_y)
        print('loss_vld=%.f, accuracy_vld=%.f' % (loss_vld, accuracy_vld))

        self.loss_history.Add(loss_train, accuracy_train, loss_vld, accuracy_vld, total_iteration, epoch)
        return loss_vld <= self.params.eps

    def __CalAccuracy(self, A, Y):
        m = A.shape[0]
        if self.params.net_type == NetType.Fitting:
            var = np.var(Y)
            mse = np.sum((A - Y) ** 2) / m
            r2 = 1 - mse / var
            return r2
        elif self.params.net_type == NetType.BinaryClassifier:
            B = np.round(A)
            r = (B == Y)
            correct = r.sum()
            return correct / m
        elif self.params.net_type == NetType.MultipleClassifier:
            rA = np.argmax(A, axis=1)
            rY = np.argmax(Y, axis=1)
            r = (rA == rY)
            correct = r.sum()
            return correct / m

    def ShowTrainingHistory(self):
        self.loss_history.ShowLossHistory(self.params)
