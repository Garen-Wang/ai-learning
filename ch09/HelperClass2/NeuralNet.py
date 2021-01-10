import numpy as np
import matplotlib.pyplot as plt
import os

from ch09.HelperClass2.TrainingHistory import *
from ch09.HelperClass2.HyperParameters import *
from ch09.HelperClass2.DataReader_2_0 import *
from ch09.HelperClass2.ClassifierFunction import *
from ch09.HelperClass2.ActivatorFunction import *
from ch09.HelperClass2.LossFunction import *
from ch09.HelperClass2.WeightBias import *


class NeuralNet(object):
    def __init__(self, params, model_name):
        self.params = params
        self.model_name = model_name
        self.sub_folder = os.getcwd() + '/' + self.__create_subfolder()
        print(self.sub_folder)

        self.wb1 = WeightBias(self.params.num_input, self.params.num_hidden, self.params.init_method, self.params.eta)
        self.wb1.InitializeWeights(self.sub_folder, False)
        self.wb2 = WeightBias(self.params.num_hidden, self.params.num_output, self.params.init_method, self.params.eta)
        self.wb2.InitializeWeights(self.sub_folder, False)

    def __create_subfolder(self):
        if self.model_name is not None:
            path = self.model_name.strip()
            path = path.rstrip('/')
            isExists = os.path.exists(path)
            if not isExists:
                os.makedirs(path)
            return path

    def forward(self, batch_x):
        # layer 1
        self.Z1 = np.dot(batch_x, self.wb1.W) + self.wb1.B
        self.A1 = Sigmoid().forward(self.Z1)
        # layer 2
        self.Z2 = np.dot(self.A1, self.wb2.W) + self.wb2.B
        if self.params.net_type == NetType.BinaryClassifier:
            self.A2 = Logistic().forward(self.Z2)
        elif self.params.net_type == NetType.MultipleClassifier:
            self.A2 = Softmax().forward(self.Z2)
        else:
            self.A2 = self.Z2

        self.output = self.A2
        return self.output

    def backward(self, batch_x, batch_y, batch_a):
        m = batch_x.shape[0]
        dZ2 = self.A2 - batch_y
        self.wb2.dW = np.dot(self.A1.T, dZ2) / m
        self.wb2.dB = np.sum(dZ2, axis=0, keepdims=True) / m
        d1 = np.dot(dZ2, self.wb2.W.T)
        dZ1, _ = Sigmoid().backward(None, self.A1, d1)
        self.wb1.dW = np.dot(batch_x.T, dZ1) / m
        self.wb1.dB = np.sum(dZ1, axis=0, keepdims=True) / m

    def update(self):
        self.wb1.update()
        self.wb2.update()

    def inference(self, x):
        self.forward(x)
        return self.output

    def train(self, reader, checkpoint, need_test):
        self.loss_trace = TrainingHistory()
        self.loss_func = LossFunction(self.params.net_type)
        if self.params.batch_size == -1:
            self.params.batch_size = reader.num_train
        max_iteration = int(reader.num_train / self.params.batch_size)
        checkpoint_iteration = int(max_iteration * checkpoint)
        need_stop = False
        for epoch in range(self.params.max_epoch):
            reader.Shuffle()
            for iteration in range(max_iteration):
                batch_x, batch_y = reader.GetBatchTrainSamples(self.params.batch_size, iteration)
                batch_a = self.forward(batch_x)
                self.backward(batch_x, batch_y, batch_a)
                self.update()

                total_iteration = epoch * max_iteration + iteration
                if (total_iteration + 1) % checkpoint_iteration == 0:
                    need_stop = self.CheckErrorAndLoss(reader, batch_x, batch_y, epoch, total_iteration)
                    if need_stop:
                        break

            if need_stop:
                break

        self.SaveResult()
        if need_test:
            print('testing...')
            accuracy = self.Test(reader)
            print(accuracy)

    def CheckErrorAndLoss(self, reader, train_x, train_y, epoch, total_iteration):
        print("epoch=%d, total_iteration=%d" % (epoch, total_iteration))

        self.forward(train_x)
        loss_train = self.loss_func.CheckLoss(self.output, train_y)
        accuracy_train = self.__CalAccuracy(self.output, train_y)
        print("loss_train=%.6f, accuracy_train=%f" % (loss_train, accuracy_train))

        vld_x, vld_y = reader.GetValidationSet()
        self.forward(vld_x)
        loss_vld = self.loss_func.CheckLoss(self.output, vld_y)
        accuracy_vld = self.__CalAccuracy(self.output, vld_y)
        print("loss_valid=%.6f, accuracy_valid=%f" % (loss_vld, accuracy_vld))

        self.loss_trace.Add(epoch, total_iteration, loss_train, accuracy_train, loss_vld, accuracy_vld)
        return loss_vld < self.params.eps

    def Test(self, reader):
        x, y = reader.GetTestSet()
        self.forward(x)
        correct = self.__CalAccuracy(self.output, y)
        return correct

    def __CalAccuracy(self, a, y):
        m = a.shape[0]
        if self.params.net_type == NetType.Fitting:
            var = np.var(y)
            mse = np.sum((a - y) ** 2) / m
            r2 = 1 - mse / var
            return r2
        elif self.params.net_type == NetType.BinaryClassifier:
            b = np.round(a)
            r = (b == y)
            correct = r.sum()
            return correct / m
        elif self.params.net_type == NetType.MultipleClassifier:
            ra = np.argmax(a, axis=1)
            ry = np.argmax(y, axis=1)
            r = (ra == ry)
            correct = r.sum()
            return correct / m

    def SaveResult(self):
        self.wb1.SaveResultValue(self.sub_folder, "wb1")
        self.wb2.SaveResultValue(self.sub_folder, "wb2")

    def LoadResult(self):
        self.wb1.LoadResultValue(self.sub_folder, "wb1")
        self.wb2.LoadResultValue(self.sub_folder, "wb2")

    def ShowTrainingHistory(self):
        self.loss_trace.ShowLossHistory(self.params)

    def GetTrainingHistory(self):
        return self.loss_trace
