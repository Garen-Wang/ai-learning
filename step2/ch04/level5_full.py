
from helper1.NeuralNet import *
file_name = '../../ai-data/Data/ch04.npz'

def showResult(neural, reader):
    X, Y = reader.getWholeTrainSamples()
    plt.plot(X, Y, '.')
    PX = np.linspace(0, 1, 5).reshape(5, 1)
    PZ = neural.forwardBatch(PX)
    plt.plot(PX, PZ, 'r')
    plt.title('Air Conditioner Power')
    plt.xlabel('Number of Servers(K)')
    plt.ylabel('Power of Air Conditioner(KW)')
    plt.show()

if __name__ == '__main__':
    reader = DataReader(file_name)
    reader.readData()
    params = HyperParameters(1, 1, eta = 0.1, max_epoch = 100, batch_size = 10, eps = 0.02)
    neural = NeuralNet(params)
    neural.train(reader)
    showResult(neural, reader)