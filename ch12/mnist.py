from ch12.helperClass.NeuralNet import *
from ch12.helperClass.ImageDataReader import *


def main():
    reader = ImageDataReader('vector')
    reader.read_data()
    reader.normalize_x()
    reader.normalize_y(NetType.MultipleClassifier, base=0)
    reader.shuffle()
    reader.generate_validation_set(12)

    num_input = reader.num_feature
    num_hidden1 = 64
    num_hidden2 = 16
    num_output = reader.num_category
    eta = 0.2
    eps = 0.01
    batch_size = 128
    max_epoch = 20

    params = HyperParameters(num_input, num_hidden1, num_hidden2, num_output, batch_size, max_epoch, eta, eps, NetType.MultipleClassifier, InitMethod.Xavier)
    net = NeuralNet(params)
    net.train(reader, 0.5, True)
    # show history!!!
    pass


if __name__ == '__main__':
    main()