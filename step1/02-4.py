import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def f(x, y):
    return x ** 2 + np.sin(y) ** 2

def f_derivate(theta):
    x, y = theta[0], theta[1]
    return np.array([2 * x, 2 * np.sin(y) * np.cos(y)])

def show_3d(X, Y, Z):
    fig = plt.figure()
    ax = Axes3D(fig)

    u = np.linspace(-3, 3, 100)
    v = np.linspace(-3, 3, 100)
    X, Y = np.meshgrid(u, v)
    R = np.zeros((len(u), len(v)))
    for i in range(len(u)):
        for j in range(len(v)):
            R[i, j] = X[i, j] ** 2 + np.sin(Y[i, j]) ** 2
    
    ax.plot_surface(X, Y, R, cmap = 'rainbow')
    plt.plot(x, y, z, c = 'black')
    plt.show()

if __name__ == '__main__':
    eta = 0.1
    eps = 1e-2
    theta = np.array([3, 1])
    X, Y, Z = [], [], []
    for i in range(100):
        x, y = theta[0], theta[1]
        z = f(x, y)
        X.append(x)
        Y.append(y)
        Z.append(z)
        print('{}: x = {}, y = {}, z = {}'.format(i, x, y, z))
        d_theta = f_derivate(theta)
        theta = theta - eta * d_theta
        if z < eps:
            break
    show_3d(X, Y, Z)

