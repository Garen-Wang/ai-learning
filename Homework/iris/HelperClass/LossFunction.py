import numpy as np

from Homework.iris.HelperClass.EnumDef import *


class LossFunction(object):
    def __init__(self, net_type):
        self.net_type = net_type

    def MSE(self, A, Y, m):
        LOSS = np.multiply(A - Y, A - Y)
        loss = LOSS.sum() / m / 2
        return loss

    def CE2(self, A, Y, m):
        p1 = np.multiply(Y, np.log(A))
        p2 = np.multiply(1 - Y, np.log(1 - A))
        LOSS = p1 + p2
        loss = np.sum(-LOSS) / m
        return loss

    def CE3(self, A, Y, m):
        LOSS = np.multiply(Y, np.log(A))
        loss = np.sum(-LOSS) / m
        return loss

    def CheckLoss(self, A, Y):
        m = Y.shape[0]
        if self.net_type == NetType.Fitting:
            loss = self.MSE(A, Y, m)
        elif self.net_type == NetType.BinaryClassifier:
            loss = self.CE2(A, Y, m)
        elif self.net_type == NetType.MultipleClassifier:
            loss = self.CE3(A, Y, m)
        else:
            loss = 233
        return loss
