import math
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch import nn
from torch import optim
from torch.utils.data import DataLoader, TensorDataset

def draw(X):
    print(X.shape)
    plt.imshow(X.reshape((28, 28)), cmap='gray')
    plt.show()

def read_data():
    path = Path('data/mnist/mnist.pkl')
    if path.exists():
        with open('data/mnist/mnist.pkl', 'rb') as f:
            (XTrain, YTrain), (XTest, YTest), _ = pickle.load(f, encoding='latin-1')
        return XTrain, YTrain, XTest, YTest
    else:
        raise Exception(FileNotFoundError)

def generate_validation_set(k=10):
    global num_train, XTrain, YTrain
    num_valid = num_train // k
    num_train -= num_valid
    XValid, YValid = XTrain[:num_valid], YTrain[:num_valid]
    XTrain, YTrain = XTrain[num_valid:], YTrain[num_valid:]
    return XValid, YValid, num_valid


XTrain, YTrain, XTest, YTest = read_data()  # train: 50000, test: 10000
num_train = XTrain.shape[0]
num_test = XTest.shape[0]
XValid, YValid, num_valid = generate_validation_set(k=10)
XTrain, YTrain, XValid, YValid, XTest, YTest = map(torch.tensor, (XTrain, YTrain, XValid, YValid, XTest, YTest))


class NeuralNet(nn.Module):
    def __init__(self):
        super(NeuralNet, self).__init__()
        self.linear = nn.Linear(784, 10)

    def forward(self, batch_x):
        return self.linear(batch_x)


def accuracy(batch_z, batch_y):
    temp = torch.argmax(batch_z, dim=1)
    r = (temp == batch_y)
    return r.float().mean()

# hyper-parameter
bs = 64
lr = 0.05
max_epoch = 20
# essential stuff
loss_func = F.cross_entropy
model = NeuralNet()
optimizer = optim.SGD(model.parameters(), lr=lr)
# datasets and dataloaders
train_set = TensorDataset(XTrain, YTrain)
train_loader = DataLoader(train_set, batch_size=bs, shuffle=True)
valid_set = TensorDataset(XValid, YValid)
valid_loader = DataLoader(valid_set, batch_size=bs * 2, shuffle=False)

def train():
    print('training...')
    for epoch in range(max_epoch):
        model.train()
        # training: using training set
        for batch_x, batch_y in train_loader:
            # forward
            batch_z = model(batch_x)
            # backward
            loss = loss_func(batch_z, batch_y)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        model.eval()
        # inference: using validation set
        with torch.no_grad():
            valid_loss = sum(loss_func(model(batch_x), batch_y) for batch_x, batch_y in valid_loader) / num_valid
        print("epoch %d, validation loss=%.4f" % (epoch, valid_loss))

    print('training done.')


def test():
    print('testing...')
    ZTest = model(XTest)
    print('loss=%.4f, accuracy=%.4f' % (loss_func(ZTest, YTest), accuracy(ZTest, YTest)))
    print('testing done.')


train()
test()
