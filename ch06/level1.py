import numpy as npy
from helper3.NeuralNet import *
file_name = '../ai-data/Data/ch06.npz'

if __name__ == '__main__':
    reader = DataReader(file_name)
    reader.readData()
    params = HyperParameters(2, 1, eta=0.1, max_epoch=100, batch_size=10, eps=1e-3, net_type=NetType.BinaryClassifier)
    neural = NeuralNet(params)
    neural.train(reader, checkpoint=1)

    x_predicate = npy.array([0.58,0.92,0.62,0.55,0.39,0.29]).reshape(3, 2)
    ans = neural.forwardBatch(x_predicate)
    print("ans = ", ans)
