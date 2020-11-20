import numpy as npy
from helper4.EnumDef import *


class LossFunction(object):
    def __init__(self, net_type):
        self.net_type = net_type

    def CE2(self, A, Y, count):
        sum1 = npy.dot(Y, npy.log(A))
        sum2 = npy.dot(1 - Y, npy.log(1 - A))
        LOSS = -(sum1 + sum2)
        loss = LOSS.sum() / count
        return loss

    def CE3(self, A, Y, count):
        SUM = npy.multiply(Y, npy.log(A))
        LOSS = -SUM.sum()
        loss = LOSS / count
        return loss
