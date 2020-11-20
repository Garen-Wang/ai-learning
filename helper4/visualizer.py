import numpy as npy
import matplotlib.pyplot as plt
import math


def drawThreeCategoryPoints(X1, X2, Y_onehot, xlabel='x1', ylabel='x2', title=None, show=False, isPredicate=False):
    colors = ['b', 'r', 'g']
    shapes = ['s', 'x', 'o']
    m = X1.shape[0]
    for i in range(m):
        j = npy.argmax(Y_onehot[i])
        if isPredicate:
            plt.scatter(X1[i], X2[i], c=colors[j], marker='^', s=200)
        else:
            plt.scatter(X1[i], X2[i], c=colors[j], marker=shapes[j])

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if show:
        plt.show()
