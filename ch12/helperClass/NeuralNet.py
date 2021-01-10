import numpy as np
import math

from ch12.helperClass.ImageDataReader import *
from ch12.helperClass.LossFunctions import *
from ch12.helperClass.HyperParameters import *
from ch12.helperClass.WeightsBias import *
from ch12.helperClass.Activators import *
from ch12.helperClass.Classifiers import *
from ch12.helperClass.TrainingHistory import *


class NeuralNet(object):
    def __init__(self, params):
        self.params = params

        self.wb1 = WeightsBias(self.params.num_input, self.params.num_hidden1, self.params.init_method, self.params.eta)
        self.wb2 = WeightsBias(self.params.num_hidden1, self.params.num_hidden2, self.params.init_method, self.params.eta)
        self.wb3 = WeightsBias(self.params.num_hidden2, self.params.num_output, self.params.init_method, self.params.eta)

    def forward(self, batch_x):
        self.Z1 = np.dot(batch_x, self.wb1.W) + self.wb1.B
        self.A1 = Sigmoid().forward(self.Z1)
        self.Z2 = np.dot(self.A1, self.wb2.W) + self.wb2.B
        self.A2 = Tanh().forward(self.Z2)
        self.Z3 = np.dot(self.A2, self.wb3.W) + self.wb3.B

        if self.params.net_type == NetType.Fitting:
            self.A3 = self.Z3
        elif self.params.net_type == NetType.BinaryClassifier:
            self.A3 = Logistic().forward(self.Z3)
        elif self.params.net_type == NetType.MultipleClassifier:
            self.A3 = Softmax().forward(self.Z3)
        return self.A3

    def backward(self, batch_x, batch_y, batch_z):
        m = batch_x.shape[0]

        dZ3 = batch_z - batch_y
        self.wb3.dW = np.dot(self.A2.T, dZ3) / m
        self.wb3.dB = np.sum(dZ3, axis=0, keepdims=True) / m

        dA2 = np.dot(dZ3, self.wb3.W.T)
        dZ2, _ = Tanh().backward(None, self.A2, dA2)
        self.wb2.dW = np.dot(self.A1.T, dZ2) / m
        self.wb2.dB = np.sum(dZ2, axis=0, keepdims=True) / m

        dA1 = np.dot(dZ2, self.wb2.W.T)
        dZ1, _ = Sigmoid().backward(None, self.A1, dA1)
        self.wb1.dW = np.dot(batch_x.T, dZ1) / m
        self.wb1.dB = np.sum(dZ1, axis=0, keepdims=True) / m

    def update(self):
        self.wb1.update()
        self.wb2.update()
        self.wb3.update()

    def train(self, reader, checkpoint, need_test):
        self.history = TrainingHistory()
        self.F = LossFunction(self.params.net_type)
        loss = 10.0
        if self.params.batch_size == -1:
            self.params.batch_size = reader.num_train
        max_iteration = math.ceil(reader.num_train / self.params.batch_size)
        checkpoint_iteration = int(max_iteration * checkpoint)
        flag = False
        for epoch in range(self.params.max_epoch):
            reader.shuffle()
            for iteration in range(max_iteration):
                batch_x, batch_y = reader.get_batch_train_samples(self.params.batch_size, iteration)

                batch_z = self.forward(batch_x)
                self.backward(batch_x, batch_y, batch_z)
                self.update()

                total_iteration = epoch * max_iteration + iteration
                if (total_iteration + 1) % checkpoint_iteration == 0:
                    flag = self.check_error_loss(reader, batch_x, batch_y, epoch, total_iteration)
                    if flag:
                        break
            if flag:
                break
        if need_test:
            print('testing...')
            accuracy = self.test(reader)
            print(accuracy)

    def check_error_loss(self, reader, train_x, train_y, epoch, total_iteration):

        train_z = self.forward(train_x)
        loss_train = self.F.get_loss(train_z, train_y)
        accuracy_train = self.get_accuracy(train_z, train_y)

        valid_x, valid_y = reader.get_validation_set()
        valid_z = self.forward(valid_x)
        loss_valid = self.F.get_loss(valid_z, valid_y)
        accuracy_valid = self.get_accuracy(valid_z, valid_y)

        self.history.add(epoch, total_iteration, loss_train, accuracy_train, loss_valid, accuracy_valid)
        return loss_valid <= self.params.eps

    def test(self, reader):
        batch_x, batch_y = reader.get_test_set()
        batch_z = self.forward(batch_x)
        return self.get_accuracy(batch_z, batch_y)

    def get_accuracy(self, batch_z, batch_y):
        m = batch_y.shape[0]
        if self.params.net_type == NetType.Fitting:
            var = np.var(batch_y)
            mse = np.sum((batch_z - batch_y) ** 2) / m
            return 1 - mse / var
        elif self.params.net_type == NetType.BinaryClassifier:
            batch_zz = np.round(batch_z)
            r = (batch_y == batch_zz)
            return np.sum(r) / m
        elif self.params.net_type == NetType.MultipleClassifier:
            rz = np.argmax(batch_z, axis=1)
            ry = np.argmax(batch_y, axis=1)
            r = (rz == ry)
            return np.sum(r) / m
