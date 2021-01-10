import numpy as np
import matplotlib.pyplot as plt

def forward(x):
    a = x ** 2
    b = np.log(a)
    c = np.sqrt(b)
    return a, b, c

def backward(x, a, b, c, y):
    loss = c - y
    delta_c = c - y
    delta_b = 2 * np.sqrt(b) * delta_c
    delta_a = a * delta_b
    delta_x = delta_a * 0.5 / x
    return loss, delta_x, delta_a, delta_b, delta_c

def update(x, delta_x):
    x = x - delta_x
    if x < 1:
        x = 1.1
    return x

def draw_function(X, Y):
    x = np.linspace(1.2, 10)
    a = x * x
    b = np.log(a)
    c = np.sqrt(b)
    plt.plot(x, c)

    plt.plot(X, Y, 'x')

    d = 1 / (np.sqrt(np.log(x * x)))
    plt.plot(x, d)
    plt.show()

X, Y = [], []

if __name__ == '__main__':
    x = 1.3
    y = 1.8
    eps = 1e-4
    for i in range(20):
        a, b, c = forward(x)
        print('a={}, b={}, c={}'.format(a, b, c))
        X.append(x)
        Y.append(c)
        loss, delta_x, delta_a, delta_b, delta_c = backward(x, a, b, c, y)
        if abs(loss) < eps:
            print('done!')
            break
        x = update(x, delta_x)
        print('delta_c={}, delta_b={}, delta_a={}, delta_x={}'.format(delta_c, delta_b, delta_a, delta_x))
    draw_function(X, Y)