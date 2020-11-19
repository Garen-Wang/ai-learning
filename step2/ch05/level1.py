from helper2.DataReader import *
import numpy as npy

file_name = '../../ai-data/Data/ch05.npz'
if __name__ == '__main__':
    reader = DataReader(file_name)
    reader.readData()
    X, Y = reader.getWholeTrainSamples()
    m = X.shape[0]
    one = npy.ones((m, 1))
    x = npy.column_stack((one, (X[0:m, :])))
    a = npy.dot(x.T, x)
    b = npy.asmatrix(a)
    c = npy.linalg.inv(b)
    d = np.dot(c, x.T)
    e = np.dot(d, Y)
    print(e)
    b = e[0, 0]
    w1 = e[1, 0]
    w2 = e[2, 0]
    z = w1 * 15 + w2 * 93 + b
    print("z = %d" % z)

