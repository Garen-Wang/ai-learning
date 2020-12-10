import numpy as np
import matplotlib.pyplot as plt
import math

from step6.HelperClass2.EnumDef import *


def DrawTwoCategoryPoints(X1, X2, Y, title=None, show=False, isPredicate=False):
    colors = ['b', 'r']
    shapes = ['o', 'x']
    for i in range(Y.shape[0]):
        j = int(round(Y[i, 0]))
        if j < 0:
            j = 0
        if isPredicate:
            plt.scatter(X1[i], X2[i], color=colors[j], marker='^', s=200, zorder=10)
        else:
            plt.scatter(X1[i], X2[i], color=colors[j], marker=shapes[j], zorder=10)

    plt.xlabel('x')
    plt.ylabel('y')
    if title is not None:
        plt.title(title)
    if show:
        plt.show()


def DrawThreeCategoryPoints(X1, X2, Y_onehot, title=None, show=False, isPredicate=False):
    colors = ['b', 'r', 'g']
    shapes = ['o', 'x', 's']
    for i in range(Y_onehot.shape[0]):
        j = np.argmax(Y_onehot[i])
        if isPredicate:
            plt.scatter(X1[i], X2[i], color=colors[j], marker='^', s=200, zorder=10)
        else:
            plt.scatter(X1[i], X2[i], color=colors[j], marker=shapes[j], zorder=10)

    plt.xlabel('x')
    plt.ylabel('y')
    if title is not None:
        plt.title(title)
    if show:
        plt.show()


def ShowClassificationResult25D(net, count, title, show=False):
    x = np.linspace(0, 1, count)
    y = np.linspace(0, 1, count)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros((count, count))
    input = np.hstack((X.ravel().reshape(count * count, 1), Y.ravel().reshape(count * count, 1)))
    output = net.inference(input)
    if net.params.net_type == NetType.BinaryClassifier:
        Z = output.reshape(count, count)
    elif net.params.net_type == NetType.MultipleClassifier:
        sm = np.argmax(output, axis=1)
        Z = sm.reshape(count, count)

    plt.contourf(X, Y, Z, cmap=plt.cm.Spectral, zorder=1)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(title)
    if show:
        plt.plot()
