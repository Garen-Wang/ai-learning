import numpy as np
from ch12.helperClass.EnumDef import NetType


class LossFunction(object):
    def __init__(self, net_type):
        self.net_type = net_type

    def get_loss(self, A, Y):
        loss = .0
        m = A.shape[0]
        if self.net_type == NetType.Fitting:
            loss = self.MSE(A, Y, m)
        elif self.net_type == NetType.BinaryClassifier:
            loss = self.CE2(A, Y, m)
        elif self.net_type == NetType.MultipleClassifier:
            loss = self.CE(A, Y, m)
        return loss

    def MSE(self, A, Y, m):
        temp1 = A - Y
        temp2 = temp1 * temp1
        loss = np.sum(temp2) / m / 2.0
        return loss

    def CE2(self, A, Y, m):
        temp1 = Y * np.log(A)
        temp2 = (1 - Y) * np.log(1 - A)
        temp = -(temp1 + temp2)
        return np.sum(temp) / m

    def CE(self, A, Y, m):
        temp = Y * np.log(A)
        loss = np.sum(-temp) / m
        return loss
