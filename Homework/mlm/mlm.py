from mpl_toolkits.mplot3d import Axes3D

from Homework.mlm.HelperClass.NeuralNet import *

file_name = 'mlm.csv'

def showResult(reader, neural):
    X, Y = reader.getWholeTrainSamples()
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.scatter(X[:, 0], X[:, 1], Y[:, 0])
    x = np.linspace(0, 1, 50)
    y = np.linspace(0, 1, 50)
    X, Y = np.meshgrid(x, y)
    R = np.hstack((X.ravel().reshape(2500, 1), Y.ravel().reshape(2500, 1)))
    Z = neural.forward(R)
    Z = Z.reshape(50, 50)
    ax.plot_surface(X, Y, Z, cmap="rainbow")
    plt.show()

if __name__ == '__main__':
    reader = DataReader(file_name)
    reader.readData()
    # print(reader.xRaw)
    reader.normalizeX()
    reader.normalizeY()
    # print(reader.xTrain)
    params = HyperParameters(2, 1, eta=0.1, max_epoch=100, batch_size=5, eps=1e-4)
    neural = NeuralNet(params)
    neural.train(reader, 0.1)

    showResult(reader, neural)
    print("W = ", neural.W)
    print("B = ", neural.B)
    print(neural.checkLoss(reader))
