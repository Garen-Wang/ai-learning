import numpy as np
import matplotlib.pyplot as plt

def draw_function():
    x = np.linspace(-3, 3)
    y = x ** 2
    plt.plot(x, y)
    Y = []
    for i in X:
        Y.append((i) ** 2)
    plt.plot(X, Y)
    plt.show()

if __name__ == '__main__':
    x = 3
    eta = 0.2
    eps = 1e-4
    X = []
    X.append(x)
    y = x ** 2
    while y > eps:
        x = x - eta * 2 * x
        X.append(x)
        y = x ** 2
    
    draw_function()

