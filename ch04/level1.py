import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from helper1.DataReader import *

file_name = '../ai-data/data/ch04.npz'

def method1(X, Y, m):
    x_mean = X.mean()
    p = sum(Y  * (X - x_mean))
    q = sum(X * X) - sum(X) ** 2 / m
    return p / q

def method2(X, Y, m):
    x_mean = X.mean()
    p = sum(Y * (X - x_mean))
    q = sum(X ** 2) - x_mean * sum(X)
    return p / q

def method3(X, Y, m):
    p = m * sum(X * Y) - sum(X) * sum(Y)
    q = m * sum(X ** 2) - sum(X) * sum(X)
    return p / q

def method_b1(X, Y, w, m):
    return sum(Y - w * X) / m

def method_b2(X, Y, w, m):
    return Y.mean() - w * X.mean()

if __name__ == '__main__':
    reader = DataReader(file_name)
    reader.readData()
    X, Y = reader.getWholeTrainSamples()
    m = X.shape[0]
    