import numpy as np
from pathlib import Path
import pickle
import matplotlib.pyplot as plt
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def read_data():
    path = Path('data/mnist/mnist.pkl')
    if path.exists():
        with open('data/mnist/mnist.pkl', 'rb') as f:
            (XTrain, YTrain), (XTest, YTest), _ = pickle.load(f, encoding='latin-1')
        return XTrain, YTrain, XTest, YTest
    else:
        raise Exception(FileNotFoundError)


def draw(X):
    print(X.shape)
    plt.imshow(X.reshape((28, 28)), cmap='gray')
    plt.show()


class NeuralNet(nn.Module):
    def __init__(self):
        super(NeuralNet, self).__init__()
        self.weights = nn.Parameter(torch.randn(784, 10) / math.sqrt(784))
        self.bias = nn.Parameter(torch.zeros(10))

    def forward(self, batch_x):
        return batch_x @ self.weights + self.bias


def accuracy(batch_z, batch_y):
    temp = torch.argmax(batch_z, dim=1)
    r = (temp == batch_y)
    return r.float().mean()


def get_batch_train_data(batch_size, iteration):
    start = batch_size * iteration
    end = start + iteration
    return XTrain[start:end], YTrain[start:end]


def get_batch_test_data(batch_size, iteration):
    start = batch_size * iteration
    end = start + iteration
    return XTest[start:end], YTest[start:end]


XTrain, YTrain, XTest, YTest = read_data()  # train: 50000, test: 10000
XTrain, YTrain, XTest, YTest = map(torch.tensor, (XTrain, YTrain, XTest, YTest))

loss_func = F.cross_entropy
model = NeuralNet() # I am hanhan!

def train(max_epoch, max_iteration, batch_size, lr):
    print('training...')
    for epoch in range(max_epoch):
        for iteration in range(max_iteration):
            batch_x, batch_y = get_batch_train_data(batch_size, iteration)
            batch_z = model(batch_x)
            loss = loss_func(batch_z, batch_y)

            loss.backward()
            with torch.no_grad():
                for p in model.parameters():
                    p -= p.grad * lr
                model.zero_grad()

    print('training done.')


def test():
    print('testing...')
    ZTest = model(XTest)
    print('loss=%.4f, accuracy=%.4f' % (loss_func(ZTest, YTest), accuracy(ZTest, YTest)))
    print('testing done.')


def main():
    num_train = XTrain.shape[0]
    num_test = XTest.shape[0]
    # batch_x = XTrain[:batch_size]
    # batch_z = forward(batch_x)
    # print(batch_z[0], batch_z.shape)
    #
    # batch_y = YTrain[:batch_size]
    # print(loss_func(batch_z, batch_y))
    #
    # print(accuracy(batch_z, batch_y))

    batch_size = 64
    lr = 0.05
    max_epoch = 20
    max_iteration = math.ceil(num_train / batch_size)
    train(max_epoch, max_iteration, batch_size, lr)
    test()


if __name__ == '__main__':
    main()
