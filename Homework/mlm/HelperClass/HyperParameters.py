class HyperParameters(object):
    def __init__(self, input_size, output_size, eta=0.1, max_epoch=100, batch_size=10, eps=1e-4):
        self.input_size = input_size
        self.output_size = output_size
        self.eta = eta
        self.max_epoch = max_epoch
        self.batch_size = batch_size
        self.eps = eps

    def getTitle(self):
        return 'eta={}, max_epoch={}, batch_size={}'.format(self.eta, self.max_epoch,  self.batch_size)

