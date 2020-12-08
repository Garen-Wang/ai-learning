from step4.HelperClass2.EnumDef import *

class HyperParameters(object):
    def __init__(self, num_input, num_hidden, num_output,
                 eta=0.1, max_epoch=10000, batch_size=5, eps=1e-1,
                 net_type=NetType.Fitting, init_method=InitialMethod.Xavier):
        self.num_input = num_input
        self.num_hidden = num_hidden
        self.num_output = num_output
        self.eta = eta
        self.max_epoch = max_epoch
        self.batch_size = batch_size
        self.eps = eps
        self.net_type = net_type
        self.init_method = init_method

    def toString(self):
        title = "bz:{}, eta={}, ne:{}".format(self.batch_size, self.eta, self.num_hidden)
        return title
