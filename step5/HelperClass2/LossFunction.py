import numpy as np

from step5.HelperClass2.EnumDef import *


class LossFunction(object):
    def __init__(self, net_type):
        self.net_type = net_type

    def MSE(self, A, Y, count):
        p1 = A - Y
        p2 = np.multiply(p1, p1)
        loss = p2.sum() / count / 2
        return loss

    def CE2(self, A, Y, count):
        p1 = 1 - Y
        p2 = np.log(1 - A)
        p3 = np.log(A)
        p4 = np.multiply(p1, p2)
        p5 = np.multiply(Y, p3)
        LOSS = np.sum(-(p4 + p5))
        loss = LOSS.sum() / count
        return loss

    def CE3(self, A, Y, count):
        LOSS = -(np.multiply(Y, np.log(A)))
        loss = LOSS.sum() / count
        return loss

    def CheckLoss(self, A, Y):
        m = Y.shape[0]
        if self.net_type == NetType.Fitting:
            loss = self.MSE(A, Y, m)
        elif self.net_type == NetType.BinaryClassifier:
            loss = self.CE2(A, Y, m)
        elif self.net_type == NetType.MultipleClassifier:
            loss = self.CE3(A, Y, m)
        return loss
